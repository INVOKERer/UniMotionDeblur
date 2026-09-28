# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

"""
NAFSSR: Stereo Image Super-Resolution Using NAFNet
@InProceedings{Chu2022NAFSSR,
  author    = {Xiaojie Chu and Liangyu Chen and Wenqing Yu},
  title     = {NAFSSR: Stereo Image Super-Resolution Using NAFNet},
  booktitle = {CVPRW},
  year      = {2022},
}
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY

from .arch_util import Local_Base
from .util_win import *
from .util_basicblock import NAFBlock as LocalNAFBlock
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class NAFSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(
            in_channels=c,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True,
        )
        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dw_channel // 2,
                out_channels=dw_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(
            in_channels=c,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size


    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            # print('xxx')
            return self.avgpool(x)
            # return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = int(min(h, self.kernel_size[0])), int(min(w, self.kernel_size[1]))
            # print(self.kernel_size, k1, k2, s.shape)
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')
        # print(n, c, h, w, self.kernel_size, out.shape)
        return out

class NAFBlockX(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., local_avgpool=False, train_size=[256, 256], patch_size=None):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.patch_size = patch_size
        # Simplified Channel Attention
        if local_avgpool:
            base_size = [int(train_size[-2] * 1.), int(train_size[-1] * 1.)]
            # base_size = train_size
            self.sca = nn.Sequential(
                AvgPool2d(kernel_size=base_size, base_size=base_size, train_size=train_size),
                nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                          groups=1, bias=True),
            )
        else:
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                          groups=1, bias=True),
            )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.register_parameter('beta', nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True))
        self.register_parameter('gamma', nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True))
        # self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)

        b, c, H, W = x.shape
        if self.patch_size is not None and (H != self.patch_size[0] or W != self.patch_size[1]):
            x, batch_list = window_partitionx(x, self.patch_size)
        # print(x.shape)
        x = x * self.sca(x)
        if self.patch_size is not None and (H != self.patch_size[0] or W != self.patch_size[1]):
            x = window_reversex(x, self.patch_size, H, W, batch_list)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

@ARCH_REGISTRY.register()
class NAFNetBaseline(nn.Module):
    def __init__(
        self,
        img_channel=3,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=[],
        dec_blk_nums=[],
        window_size=8,
    ):
        super().__init__()

        self.intro = nn.Conv2d(
            in_channels=img_channel,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=img_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        self.encoders = nn.ModuleList()
        # self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        for i, num in enumerate(dec_blk_nums):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            setattr(
                self,
                f"decoder{i}",
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)]),
            )

    def forward(self, inp, gt=None):
        # B, C, H, W = inp.shape
        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for i, (up, enc_skip) in enumerate(zip(self.ups, encs[::-1])):
            x = up(x)
            x = x + enc_skip
            decoder = getattr(self, f"decoder{i}")
            x = decoder(x)

        # if not hook:

        x = self.ending(x)
        x = x + inp

        return x

@ARCH_REGISTRY.register()
class NAFNetX(nn.Module):
    def __init__(
        self,
        img_channel=3,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=[],
        dec_blk_nums=[],
        patch_size=[[64, 64], [32, 32], [16, 16], [8, 8]],
        patch_size_latent=[8, 8],
        window_size=8,
    ):
        super().__init__()

        self.intro = nn.Conv2d(
            in_channels=img_channel,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=img_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        self.encoders = nn.ModuleList()
        # self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for i, num in enumerate(enc_blk_nums):
            self.encoders.append(nn.Sequential(*[NAFBlockX(chan, patch_size=patch_size[i]) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2


        self.middle_blks = nn.Sequential(
            *[NAFBlockX(chan, patch_size=patch_size_latent) for _ in range(middle_blk_num)]
        )
        patch_size_dec = patch_size[::-1]
        # print(patch_size_dec, patch_size)
        for i, num in enumerate(dec_blk_nums):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)
                )
            )
            chan = chan // 2

            setattr(
                self,
                f"decoder{i}",
                nn.Sequential(*[NAFBlockX(chan, patch_size=patch_size_dec[i]) for _ in range(num)]),
            )

    def forward(self, inp, hook=False):
        # B, C, H, W = inp.shape
        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for i, (up, enc_skip) in enumerate(zip(self.ups, encs[::-1])):
            x = up(x)
            x = x + enc_skip
            decoder = getattr(self, f"decoder{i}")
            x = decoder(x)

        if not hook:

            x = self.ending(x)
            x = x + inp

            return x
@ARCH_REGISTRY.register()
class NAFNet(Local_Base, NAFNetBaseline):
    def __init__(self, *args, train_size=(1, 3, 128, 128), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNetBaseline.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)
def disabled_train(self: nn.Module) -> nn.Module:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self
@ARCH_REGISTRY.register()
class DecoupledNAFNetBaseline(nn.Module):
    def __init__(
        self,
        img_channel=3,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=[],
        dec_blk_nums=[],
        frozen_encoder=False,
        window_size=8,
    ):
        super().__init__()

        self.intro = nn.Conv2d(
            in_channels=img_channel,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=img_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        self.encoders = nn.ModuleList()
        # self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        if frozen_encoder:
            for module in [self.intro, self.downs, self.encoders]:
                module.eval()
                module.train = disabled_train
                for p in module.parameters():
                    p.requires_grad = False
        if frozen_encoder:
            self.middle_blks = nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )
        else:
            self.middle_blks = nn.Identity()
        for i, num in enumerate(dec_blk_nums):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            if frozen_encoder:
                setattr(
                    self,
                    f"decoder{i}",
                    nn.Sequential(*[NAFBlock(chan) for _ in range(num)]),
                )
            else:
                setattr(
                    self,
                    f"decoder{i}",
                    nn.Sequential(*[nn.Identity() for _ in range(1)]),
                )

    def forward(self, inp, hook=False):
        # B, C, H, W = inp.shape
        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for i, (up, enc_skip) in enumerate(zip(self.ups, encs[::-1])):
            x = up(x)
            x = x + enc_skip
            decoder = getattr(self, f"decoder{i}")
            x = decoder(x)

        if not hook:

            x = self.ending(x)
            x = x + inp

            return x

@ARCH_REGISTRY.register()
class DecoupledNAFNet(Local_Base, DecoupledNAFNetBaseline):
    def __init__(self, *args, train_size=(1, 3, 128, 128), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        DecoupledNAFNetBaseline.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


class SubScaleDecoupledNAFNetBaseline(nn.Module):
    def __init__(
        self,
        img_channel=3,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=[],
        dec_blk_nums=[],
        scale_train=1,
        frozen_train=False,
        emb_enc=False,
        window_size=8,
    ):
        super().__init__()

        self.intro = nn.Conv2d(
            in_channels=img_channel,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.emb_enc = emb_enc
        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=img_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.enc_embs = nn.ModuleList()
        chan = width
        for i, num in enumerate(enc_blk_nums):
            if i >= scale_train-1:
                self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            else:
                self.encoders.append(nn.Sequential(*[nn.Identity() for _ in range(1)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            if emb_enc:
                self.enc_embs.append(nn.Conv2d(2 * chan, chan,  kernel_size=1,
                                                padding=0,
                                                stride=1,
                                                bias=True,))
            chan = chan * 2



        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        for i, num in enumerate(dec_blk_nums):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            if len(dec_blk_nums)-i >= scale_train:
                self.decoders.append(
                    nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
                )
                # setattr(
                #     self,
                #     f"decoder{i}",
                #     nn.Sequential(*[NAFBlock(chan) for _ in range(num)]),
                # )
            else:
                self.decoders.append(
                    nn.Sequential(*[nn.Identity() for _ in range(1)])
                )
                # setattr(
                #     self,
                #     f"decoder{i}",
                #     nn.Sequential(*[nn.Identity() for _ in range(1)]),
                # )
        self.frozen_train = frozen_train
        # if frozen_train:
        #     for module in [self.intro, self.downs[scale_train:], self.encoders[scale_train:], self.decoders[:-scale_train], self.ups[:-scale_train]]:
        #         module.eval()
        #         module.train = disabled_train
        #         for p in module.parameters():
        #             p.requires_grad = False
    def forward(self, inp, pre_decs=None):
        # B, C, H, W = inp.shape
        x = self.intro(inp)

        encs = []
        decs = []
        if pre_decs is not None and self.emb_enc:
            for encoder, down, enc_emb, pre_dec in zip(self.encoders, self.downs, self.enc_embs, pre_decs):
                x = enc_emb(torch.cat([x, pre_dec], dim=1))
                x = encoder(x)
                encs.append(x)
                x = down(x)
        else:
            for encoder, down in zip(self.encoders, self.downs):
                x = encoder(x)
                encs.append(x)
                x = down(x)

        x = self.middle_blks(x)
        # decs.append(x)
        for i, (up, enc_skip, decoder) in enumerate(zip(self.ups, encs[::-1], self.decoders)):
            x = up(x)
            x = x + enc_skip
            # decoder = getattr(self, f"decoder{i}")
            x = decoder(x)
            decs.append(x)
        if not self.frozen_train:

            x = self.ending(x)
            x = x + inp

            return x
        else:
            return decs[::-1]
@ARCH_REGISTRY.register()
class ScaleDecoupledNAFNetBaseline(nn.Module):
    def __init__(
        self,
        img_channel=3,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=[],
        dec_blk_nums=[],
        scale_train=[3],
        frozen_train=False,
        window_size=8,
    ):
        super().__init__()
        self.frozen_train = frozen_train
        self.sub_nets = nn.ModuleList()
        for i, scale_ in enumerate(scale_train):
            frozen_train = True if i < len(scale_train)-1 else False
            emb_enc = True if i > 0  else False
            self.sub_nets.append(SubScaleDecoupledNAFNetBaseline(img_channel,
                                                                 width,
                                                                 middle_blk_num,
                                                                 enc_blk_nums,
                                                                 dec_blk_nums,
                                                                 scale_train[i],
                                                                 frozen_train,
                                                                 emb_enc,
                                                                 window_size))
        # print(frozen_train, len(scale_train))
        if self.frozen_train and len(scale_train) > 1:
            for module in self.sub_nets[:-1]:
                module.eval()
                module.train = disabled_train
                # print(module)
                for p in module.parameters():
                    # print(p)
                    p.requires_grad = False
    def forward(self, inp, hook=False):
        decs = None
        if len(self.sub_nets) > 1:

            with torch.no_grad():
                for sub_net in self.sub_nets[:-1]:
                    decs = sub_net(inp, decs)

        x = self.sub_nets[-1](inp, decs)
        return x

@ARCH_REGISTRY.register()
class ScaleDecoupledNAFNet(Local_Base, ScaleDecoupledNAFNetBaseline):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        ScaleDecoupledNAFNetBaseline.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


@ARCH_REGISTRY.register()
class KernelEmbEncoderNAFNet(nn.Module):
    def __init__(
        self,
        img_channel=3,
        width=64,
        middle_blk_num=1,
        kernel_blk_num=2,
        enc_blk_nums=[1, 1, 1, 28],
        dec_blk_nums=[1, 1, 1, 1],
        patch_size=[[128, 128], [64, 64], [32, 32], [16, 16]],
        overlap_size=[[16, 16], [8, 8], [4, 4], [2, 2]],
        out_channels=[64, 128, 256],
        ker_start=1,
        window_size=8,
    ):
        super().__init__()
        self.ker_start = ker_start
        self.patch_size = patch_size[::-1]
        self.ker_size = patch_size
        self.out_channels = out_channels
        self.overlap_size = overlap_size
        self.up_scale = 1
        self.overlap_size_up = self.overlap_size  # (self.overlap_size[0] * self.up_scale, self.overlap_size[1] * self.up_scale)
        # self.kernel_size = [train_size, train_size]
        self.kernel_size_up = self.ker_size
        self.nr = [1, 1, 1]
        self.nc = [1, 1, 1]
        self.stride = [-1, -1, -1]
        self.h = [-1, -1, -1]
        self.w = [-1, -1, -1]
        self.ek1, self.ek2 = [0, 0, 0], [0, 0, 0]
        self.num_patchs = [ 1, 1, 1 ]
        self.original_size = [[], [], []]
        self.idxes = [None, None, None]
        self.fuse_matrix_w1 = [None, None, None]
        self.fuse_matrix_w2 = [None, None, None]
        self.fuse_matrix_h1 = [None, None, None]
        self.fuse_matrix_h2 = [None, None, None]

        self.fuse_matrix_eh1 = [None, None, None]
        self.fuse_matrix_eh2 = [None, None, None]

        self.fuse_matrix_ew1 = [None, None, None]
        self.fuse_matrix_ew2 = [None, None, None]

        self.intro = nn.Conv2d(
            in_channels=img_channel,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=img_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.kernel_ins = nn.ModuleList()
        self.kernel_blks = nn.ModuleList()
        self.kernel_outs = nn.ModuleList()
        self.proj_feature_outs = nn.ModuleList()
        self.up_decs = nn.ModuleList()
        self.up_decs.append(nn.Identity())
        self.up_decs.append(nn.Sequential(
                    nn.Conv2d(width*4, width * 4 * 2, 1, bias=False), nn.PixelShuffle(2),
                    nn.Conv2d(width * 2, width * 2 * 2, 1, bias=False), nn.PixelShuffle(2),
                   nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=1, padding=0, stride=1,groups=1, bias=False)
                ))
        self.up_decs.append(nn.Sequential(
            nn.Conv2d(width * 2, width * 2 * 2, 1, bias=False), nn.PixelShuffle(2),
            nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=1, padding=0, stride=1, groups=1,
                      bias=False)
        ))
        self.up_decs.append(nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=1, padding=0, stride=1, groups=1,
                      bias=False)
        ))
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[LocalNAFBlock(chan, local_avgpool=True) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2


        self.middle_blks = nn.Sequential(
            *[LocalNAFBlock(chan, local_avgpool=True) for _ in range(middle_blk_num)]
        )
        # self.kernel_blks = nn.Sequential(
        #     *[NAFBlock(chan*2) for _ in range(kernel_blk_num)]
        # )
        for i, num in enumerate(dec_blk_nums):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            if i >= self.ker_start:
                self.kernel_ins.append(nn.Sequential(
                    nn.Conv2d(chan * 2, chan * 2, 1, bias=False)
                ))
                self.kernel_blks.append(nn.Sequential(
                    *[NAFBlock(chan * 2) for _ in range(kernel_blk_num)]
                ))
                self.kernel_outs.append(nn.Sequential(
                    LayerNorm2d(chan * 2),
                    nn.Conv2d(chan * 2, chan * 2, 1, bias=False)
                ))
                self.proj_feature_outs.append(nn.Sequential(
                    nn.Conv2d(chan * 2, chan, 1, bias=False)
                ))
            else:
                self.kernel_blks.append(nn.Identity())
                self.kernel_ins.append(nn.Identity())
                self.kernel_outs.append(nn.Identity())
                self.proj_feature_outs.append(nn.Identity())
            self.decoders.append(nn.Sequential(*[LocalNAFBlock(chan, local_avgpool=True) for _ in range(num)]))
    def grids(self, x, level=0):
        b, c, h, w = x.shape
        n_level = 2 ** level

        # assert b == 1
        k1, k2 = self.ker_size[level][0] , self.ker_size[level][1]
        # print(h, w, k1, k2)
        k1 = min(h, k1)
        k2 = min(w, k2)
        # print(k1, k2)
        # if self.dynic_overlap and n_level == 1:
        #     num_row = h // self.ker_size[level][0] + 1
        #     num_col = w // self.ker_size[level][1] + 1
        #     self.overlap_size[level][0] = (num_row * k1 - h) // (num_row - 1)
        #     self.overlap_size[level][1] = (num_col * k2 - w) // (num_col - 1)

        overlap_size = [self.overlap_size[level][0], self.overlap_size[level][1]]  # (64, 64)

        stride = (k1 - overlap_size[0], k2 - overlap_size[1])

        num_row = (h - overlap_size[0] - 1) // stride[0] + 1
        num_col = (w - overlap_size[1] - 1) // stride[1] + 1

        # import math
        step_j = k2 if num_col == 1 else stride[1]  # math.ceil((w - stride[1]) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else stride[0]  # math.ceil((h - stride[0]) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        # self.ek1, self.ek2 = None, None
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                # if not self.ek1:
                #     # print(step_i, i, k1, h)
                #     self.ek1 = i + k1 - h # - self.overlap_size[0]
                i = h - k1
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    # if not self.ek2:
                    #     self.ek2 = j + k2 - w # + self.overlap_size[1]
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i * self.up_scale, 'j': j * self.up_scale})
                j = j + step_j
            i = i + step_i

        self.num_patchs[level] = len(parts)
        # print(len(parts))
        parts = torch.stack(parts, dim=1)
        # print(parts.shape)
        parts = parts.view(-1, c, k1, k2)
        # print(parts.shape)
        # parts = parts.view(b*self.num_patchs[level], -1, k1, k2)
        # if level == 0:

        self.original_size[level] = (b, self.out_channels[level], h * self.up_scale , w * self.up_scale)
        self.stride[level] = (stride[0] * self.up_scale, stride[1] * self.up_scale)
        self.nr[level] = num_row
        self.nc[level] = num_col
        self.idxes[level] = idxes
        return parts

    def get_overlap_matrix(self, h, w, level=0):
        # if self.grid:
        # if self.fuse_matrix_h1 is None:
        # print( self.ek1,  self.ek2)
        self.ek1[level] = self.nr[level] * self.stride[level][0] + self.overlap_size[level][0] * 2 - h
        self.ek2[level] = self.nc[level] * self.stride[level][1] + self.overlap_size[level][1] * 2 - w
        # print(self.ek1, self.ek2, h, w, self.nr, self.nc, self.stride[level], self.overlap_size[level], h, w)
        if self.h[level] != h or self.w[level] != w:

            self.fuse_matrix_w1[level] = torch.linspace(1., 0., self.overlap_size[level][1]).view(1, 1, self.overlap_size[level][1]).unsqueeze(0)
            self.fuse_matrix_w2[level] = torch.linspace(0., 1., self.overlap_size[level][1]).view(1, 1, self.overlap_size[level][1]).unsqueeze(0)
            self.fuse_matrix_h1[level] = torch.linspace(1., 0., self.overlap_size[level][0]).view(1, self.overlap_size[level][0], 1).unsqueeze(0)
            self.fuse_matrix_h2[level] = torch.linspace(0., 1., self.overlap_size[level][0]).view(1, self.overlap_size[level][0], 1).unsqueeze(0)
        if self.h[level] != h:
            self.fuse_matrix_eh1[level] = torch.linspace(1., 0., self.ek1[level]).view(1, self.ek1[level], 1).unsqueeze(0)
            self.fuse_matrix_eh2[level] = torch.linspace(0., 1., self.ek1[level]).view(1, self.ek1[level], 1).unsqueeze(0)
        if self.w[level] != w:
            self.fuse_matrix_ew1[level] = torch.linspace(1., 0., self.ek2[level]).view(1, 1, self.ek2[level]).unsqueeze(0)
            self.fuse_matrix_ew2[level] = torch.linspace(0., 1., self.ek2[level]).view(1, 1, self.ek2[level]).unsqueeze(0)
        self.h[level] = h
        self.w[level] = w

    def grids_inverse(self, outs, level=0, out_device=None, pix_max=1., size=None):
        if out_device is None:
            out_device = outs.device
        # print(out_device)
        # print(outs.shape)
        type_out = torch.uint8 if pix_max == 255. else torch.float32
        # type_out = torch.int16
        # print(self.original_size)
        if size is None:
            preds = torch.zeros(self.original_size[level], device=out_device, dtype=type_out)  # .to(out_device)
            b, c, h, w = self.original_size[level]
        else:
            preds = torch.zeros(size, device=out_device, dtype=type_out)  # .to(out_device)
            b, c, h, w = size
        # print(self.original_size)
        # outs = torch.clamp(outs, 0, 1) # * pix_max
        # outs = outs.type(type_out)
        # count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size_up[level]
        k1 = min(h, k1)
        k2 = min(w, k2)
        # print(outs.shape, self.num_patchs[level])
        # print(b, c, h, w)
        # print(outs.shape)
        outs = outs.clone().view(b, self.num_patchs[level], -1, k1, k2)
        # if not self.h or not self.w:
        self.get_overlap_matrix(h, w, level)
        # rounding_mode = 'floor'
        # print(self.idxes)
        for cnt, each_idx in enumerate(self.idxes[level]):
            i = each_idx['i']
            j = each_idx['j']
            if self.overlap_size_up[level][0] > 0:
                if i != 0 and i + k1 != h:
                    # print(1)
                    outs[:, cnt, :, :self.overlap_size_up[level][0], :] = torch.mul(outs[:, cnt, :, :self.overlap_size_up[level][0], :],
                                                                          self.fuse_matrix_h2[level].to(outs.device))
                    # print(outs[cnt, :,  i + k1 - self.overlap_size[0]:i + k1, :].shape,
                    #       self.fuse_matrix_h1.shape)
                if i + k1 * 2 - self.ek1[level] < h:
                    # print(2)
                    # print(self.fuse_matrix_h1.shape, outs[:, cnt, :, -self.overlap_size_up[level][0]:, :].shape)
                    # print(torch.mul(outs[:, cnt, :, -self.overlap_size_up[level][0]:, :],
                    #                                                        self.fuse_matrix_h1.to(outs.device)).shape)
                    outs[:, cnt, :, -self.overlap_size_up[level][0]:, :] = torch.mul(outs[:, cnt, :, -self.overlap_size_up[level][0]:, :],
                                                                           self.fuse_matrix_h1[level].to(outs.device))
            # print(self.fuse_matrix_eh1.dtype)
            if self.ek1[level] > 0:
                if i + k1 == h and i !=0:
                    # print(3)
                    outs[:, cnt, :, :self.ek1[level], :] = torch.mul(outs[:, cnt, :, :self.ek1[level], :], self.fuse_matrix_eh2[level].to(outs.device))
                if i + k1 * 2 - self.ek1[level] == h:
                    # print(4)
                    outs[:, cnt, :, -self.ek1[level]:, :] = torch.mul(outs[:, cnt, :, -self.ek1[level]:, :],
                                                            self.fuse_matrix_eh1[level].to(outs.device))

            if self.overlap_size_up[level][1] > 0:
                if j != 0 and j + k2 != w:
                    # print(1)
                    outs[:, cnt, :, :, :self.overlap_size_up[level][1]] = torch.mul(outs[:, cnt, :, :, :self.overlap_size_up[level][1]],
                                                                          self.fuse_matrix_w2[level].to(outs.device))
                if j + k2 * 2 - self.ek2[level] < w:
                    # print(2)
                    # print(j, j + k2 - self.overlap_size[1], j + k2, self.fuse_matrix_w1.shape)
                    # print(self.fuse_matrix_w1.shape)
                    outs[:, cnt, :, :, -self.overlap_size_up[level][1]:] = torch.mul(outs[:, cnt, :, :, -self.overlap_size_up[level][1]:],
                                                                           self.fuse_matrix_w1[level].to(outs.device))
            if self.ek2[level] > 0:
                if j + k2 == w and j != 0:
                    # print(3)
                    # print('j + k2 == w: ', self.ek2, outs[cnt, :, :, :self.ek2].shape, self.fuse_matrix_ew1.shape)
                    outs[:, cnt, :, :, :self.ek2[level]] = torch.mul(outs[:, cnt, :, :, :self.ek2[level]], self.fuse_matrix_ew2[level].to(outs.device))
                if j + k2 * 2 - self.ek2[level] == w:
                    # print(4)
                    # print('j + k2*2 - self.ek2 == w: ')
                    # print(self.ek2[level], self.fuse_matrix_ew1[level].shape,outs[:, cnt, :, :, -self.ek2[level]:].shape )
                    outs[:, cnt, :, :, -self.ek2[level]:] = torch.mul(outs[:, cnt, :, :, -self.ek2[level]:],
                                                            self.fuse_matrix_ew1[level].to(outs.device))
            # print(i, j)
            # print(preds[0, :, i:i + k1, j:j + k2].shape)
            # pred_win = outs[cnt, :, :, :].clone() * pix_max
            # pred_win = pred_win.type(type_out)
            # print(pred_win.dtype, preds.dtype, type_out, outs.dtype)
            # print(outs[cnt, :, :, :].max())
            # print(preds[:, :, i:i + k1, j:j + k2].shape, j, outs.shape)
            # print(preds.shape, outs.shape)
            preds[:, :, i:i + k1, j:j + k2] += outs[:, cnt, :, :, :].type(type_out).to(out_device)
            # preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :].to(out_device) * pix_max
            # count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds  # / count_mt
    def forward(self, inp, hook=False):
        # B, C, H, W = inp.shape
        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        # x_blur = x
        x = self.middle_blks(x)
        outputs = []
        for i, (decoder, up, enc_skip) in enumerate(zip(self.decoders, self.ups, encs[::-1])):
            x = up(x) + enc_skip
            if i >= self.ker_start:
                level = len(self.decoders) - 1 - i
                b, c, H, W = x.shape
                if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                    x_, batch_list = window_partitionx(x, self.patch_size[i])
                else:
                    # print('11111111111111111')
                    # print('level: ', level)
                    x_ = self.grids(x, level=level)
                    # print(x_.shape)
                x_fft = torch.fft.rfft2(x_)
                x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
                x_ = self.kernel_ins[i](x_)
                x_ = self.kernel_blks[i](x_)
                x_ = self.kernel_outs[i](x_)
                x_real, x_imag = torch.chunk(x_, 2, dim=1)
                otf = torch.complex(x_real, x_imag)
                # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
                x_ = torch.fft.irfft2(otf * x_fft)
                if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                    x_ = window_reversex(x_, self.patch_size[i], H, W, batch_list)
                else:
                    x_ = self.grids_inverse(x_, level=level)
                x = self.proj_feature_outs[i](torch.cat([x, x_], dim=1))
                outputs.append(self.up_decs[i](x)+inp)
            x = decoder(x)
        x = self.ending(x)
        x = x + inp
        outputs.append(x)
        return outputs

