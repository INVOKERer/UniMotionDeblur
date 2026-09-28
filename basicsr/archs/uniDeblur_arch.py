import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange, repeat
import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
# import sys
# sys.path.append('/data/mxt_code/AllinOne')
from basicsr.archs.util_win import *
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
import numpy as np
import kornia
from basicsr.losses.basic_loss import *
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.util_basicblock import get_basicblock, SS2D, LayerNorm2d


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()

        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat * 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat // 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)
@ARCH_REGISTRY.register()
class uniDeblurNorm(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[6, 6, 12],
            num_blocks2=[6, 6, 12],
        baseblock=['localnaf', 'localnaf', 'localnaf'],
        ker_baseblock='EVSS',
        ffn_expansion_factor=3,
        bias=False,
        train_size=[256, 256],
        patch_size=[256, 256],
        ker_size=[[256, 256], [128, 128], [64, 64]],
        overlap_size=[[32, 32], [16, 16], [8, 8]],
        k_channels=[48, 96, 192],
        ker_start=1,
        window_size=8,
        global_fft=[False, False, False],
        patch_size_fft=[128, 128],
        kernel_learning=[False, False, False],
        dec_with_enc=False,
        geo_aug=False,
        flip_transpose_max=False
    ):
        super().__init__()
        self.kernel_learning = kernel_learning
        self.flip_transpose_max = flip_transpose_max
        self.geo_aug = geo_aug
        self.ker_start = ker_start
        self.patch_size = ker_size
        self.ker_size = ker_size
        self.out_channels = [dim, dim*2, dim*4]
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
        self.encoder = True
        self.dec_with_enc = dec_with_enc
        # self.ln0 = LayerNorm(dim)
        # self.ln1 = LayerNorm(dim * 2)
        # self.ln2 = LayerNorm(dim * 4)
        self.ln0 = nn.Tanh()
        self.ln1 = nn.Tanh()
        self.ln2 = nn.Tanh()
        if self.kernel_learning[2]:
            if self.geo_aug:
                self.kernel_ins = nn.ModuleList()
                self.kernel_blks = nn.ModuleList()
                self.kernel_outs = nn.ModuleList()
                for _ in range(4):
                    self.kernel_ins.append(nn.Sequential(
                        nn.Conv2d(dim * 1 * 2, dim * 2 * 2, 1, bias=bias)
                    ))
                    # self.kernel_blks = nn.Sequential(
                    #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
                    # )
                    kernel_blks = nn.Sequential()
                    for i in range(4):
                        block = get_basicblock(ker_baseblock, channels=dim * 2 * 2, num_heads=1, sub_idx=i,
                                               train_size=[train_size[0] // 4, train_size[1] // 4],
                                               patch_size=[patch_size[0] // 4, patch_size[1] // 4])
                        kernel_blks.add_module(f"block{i}", block)
                    self.kernel_blks.append(kernel_blks)
                    self.kernel_outs.append(nn.Sequential(
                        LayerNorm(dim * 2 * 2),
                        nn.Conv2d(dim * 2 * 2, dim * 1 * 2, 1, bias=bias)
                    ))
            else:
                self.kernel_ins = nn.Sequential(
                    nn.Conv2d(dim * 4 * 2, k_channels[2], 1, bias=bias)
                )
                # self.kernel_blks = nn.Sequential(
                #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
                # )
                self.kernel_blks = nn.Sequential()
                for i in range(4):
                    block = get_basicblock(ker_baseblock, channels=k_channels[2], num_heads=1, sub_idx=i,
                                           train_size=[train_size[0]//4, train_size[1]//4],
                                           patch_size=[patch_size[0]//4, patch_size[1]//4])
                    self.kernel_blks.add_module(f"block{i}", block)

                self.kernel_outs = nn.Sequential(
                    LayerNorm(k_channels[2]),
                    nn.Conv2d(k_channels[2], dim * 4 * 2, 1, bias=bias)
                )

            self.proj_feature_outs = nn.Sequential(
                nn.Conv2d(dim * 4 * 2, dim * 4, 1, bias=bias)
            )
            if self.dec_with_enc:
                self.up_decs1 = nn.Sequential(
                    nn.Conv2d(dim * 4 * 2, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2)
                )
                self.up_decs2 = nn.Sequential(
                    nn.Conv2d(dim * 2 * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2)
                )
                self.up_decs3 = nn.Sequential(
                    nn.Conv2d(in_channels=dim * 2, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
            else:
                self.up_decs = nn.Sequential(
                    nn.Conv2d(dim * 4, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
        if self.kernel_learning[1]:
            self.kernel_ins_mid = nn.Sequential(
                nn.Conv2d(dim * 2 * 2, k_channels[1], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks_mid = nn.Sequential()
            for i in range(4):
                block = get_basicblock(ker_baseblock, channels=k_channels[1], num_heads=1, sub_idx=i,
                                       train_size=[train_size[0] // 2, train_size[1] // 2],
                                       patch_size=[patch_size[0] // 2, patch_size[1] // 2])
                self.kernel_blks_mid.add_module(f"block{i}", block)

            self.kernel_outs_mid = nn.Sequential(
                LayerNorm(k_channels[1]),
                nn.Conv2d(k_channels[1], dim * 2 * 2, 1, bias=bias)
            )
            self.up_decs_mid = nn.Sequential(
                nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=bias)
            )
            self.proj_feature_outs_mid = nn.Sequential(
                nn.Conv2d(dim * 2 * 2, dim * 2, 1, bias=bias)
            )
        if self.kernel_learning[0]:
            self.kernel_ins_max = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, k_channels[0], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks_max = nn.Sequential()
            for i in range(4):
                block = get_basicblock(ker_baseblock, channels=k_channels[0], num_heads=1, sub_idx=i,
                                       train_size=[train_size[0] // 2, train_size[1] // 2],
                                       patch_size=[patch_size[0] // 2, patch_size[1] // 2])
                self.kernel_blks_max.add_module(f"block{i}", block)

            self.kernel_outs_max = nn.Sequential(
                LayerNorm(k_channels[0]),
                nn.Conv2d(k_channels[0], dim * 1 * 2, 1, bias=bias)
            )
            self.up_decs_max = nn.Sequential(
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=bias)
            )
            self.proj_feature_outs_max = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, dim * 1, 1, bias=bias)
            )
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.encoder_level1.add_module(f"block{i}", block)

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.encoder_level2.add_module(f"block{i}", block)

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.encoder_level3.add_module(f"block{i}", block)

        self.decoder_level3 = nn.Sequential()
        for i in range(num_blocks2[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.decoder_level3.add_module(f"block{i}", block)

        self.up3_2 = Upsample(int(dim * 2 ** 2))

        self.decoder_level2 = nn.Sequential()
        for i in range(num_blocks2[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.decoder_level2.add_module(f"block{i}", block)

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential()
        for i in range(num_blocks2[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.decoder_level1.add_module(f"block{i}", block)

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
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
        # ks = self.overlap_size[level][0]//4
        # if self.h[level] != h or self.w[level] != w:
        #
        #     self.fuse_matrix_w1[level] = torch.cat([torch.ones(ks), torch.linspace(1., 0., self.overlap_size[level][1]-ks*2), torch.zeros(ks)]).view(1, 1, self.overlap_size[level][1]).unsqueeze(0)
        #     self.fuse_matrix_w2[level] = torch.cat([torch.zeros(ks), torch.linspace(0., 1., self.overlap_size[level][1]-ks*2), torch.ones(ks)]).view(1, 1, self.overlap_size[level][1]).unsqueeze(0)
        #     self.fuse_matrix_h1[level] = torch.cat([torch.ones(ks), torch.linspace(1., 0., self.overlap_size[level][0]-ks*2), torch.zeros(ks)]).view(1, self.overlap_size[level][0], 1).unsqueeze(0)
        #     self.fuse_matrix_h2[level] = torch.cat([torch.zeros(ks), torch.linspace(0., 1., self.overlap_size[level][0]-ks*2), torch.ones(ks)]).view(1, self.overlap_size[level][0], 1).unsqueeze(0)
        # if self.h[level] != h:
        #     self.fuse_matrix_eh1[level] = torch.cat([torch.ones(ks), torch.linspace(1., 0., self.ek1[level]-ks*2), torch.zeros(ks)]).view(1, self.ek1[level], 1).unsqueeze(0)
        #     self.fuse_matrix_eh2[level] = torch.cat([torch.zeros(ks), torch.linspace(0., 1., self.ek1[level]-ks*2), torch.ones(ks)]).view(1, self.ek1[level], 1).unsqueeze(0)
        # if self.w[level] != w:
        #     self.fuse_matrix_ew1[level] = torch.cat([torch.ones(ks), torch.linspace(1., 0., self.ek2[level]-ks*2), torch.zeros(ks)]).view(1, 1, self.ek2[level]).unsqueeze(0)
        #     self.fuse_matrix_ew2[level] = torch.cat([torch.zeros(ks), torch.linspace(0., 1., self.ek2[level]-ks*2), torch.ones(ks)]).view(1, 1, self.ek2[level]).unsqueeze(0)
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
    def forward(self, inp_img, gt=None):
        outputs = []
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)


        out_dec_level3 = self.decoder_level3(out_enc_level3)
        if self.kernel_learning[2]:
            level = 2
            b, c, H, W = out_dec_level3.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_, batch_list = window_partitionx(out_dec_level3, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_ = self.grids(out_dec_level3, level=level)
                # print(x_.shape)
            # pad_size = 7
            # x_ = torch.nn.functional.pad(x_, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
            if self.geo_aug:
                x_s = x_.chunk(4, dim=1)
                x_os = list()
                for j in range(4):
                    x_j = x_s[j]
                    if j % 2 == 1:
                        x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
                    if j % 2 == 0:
                        x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
                    x_fft = torch.fft.rfft2(x_j)
                    x_j = torch.cat([x_fft.real, x_fft.imag], dim=1)
                    x_j = self.kernel_ins[j](x_j)
                    x_j = self.kernel_blks[j](x_j)
                    x_j = self.kernel_outs[j](x_j)
                    x_real, x_imag = torch.chunk(x_j, 2, dim=1)
                    otf = torch.complex(x_real, x_imag)
                    # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
                    # print(x_fft.shape, otf.shape)
                    x_j = torch.fft.irfft2(otf * x_fft)
                    if j % 2 == 1:
                        x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
                    if j % 2 == 0:
                        x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
                    x_os.append(x_j)
                x_ = torch.cat(x_os, dim=1)
            else:
                x_fft = torch.fft.rfft2(x_)
                x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
                x_ = self.kernel_ins(x_)
                x_ = self.kernel_blks(x_)
                x_ = self.kernel_outs(x_)
                x_real, x_imag = torch.chunk(x_, 2, dim=1)
                otf = torch.complex(x_real, x_imag)
                # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
                # print(x_fft.shape, otf.shape)
                x_ = torch.fft.irfft2(otf * x_fft)
            # print(x_.shape)
            # x_ = x_[..., pad_size:-pad_size, pad_size:-pad_size].contiguous()
            x_ = self.ln2(x_)
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_ = window_reversex(x_, self.ker_size[level], H, W, batch_list)
            else:
                x_ = self.grids_inverse(x_, level=level)
            out_dec_level3 = self.proj_feature_outs(torch.cat([out_dec_level3, x_], dim=1))
            if self.dec_with_enc:
                x0 = self.up_decs1(torch.cat([out_enc_level3, x_], dim=1))
                x0 = self.up_decs2(torch.cat([out_enc_level2, x0], dim=1))
                x0 = self.up_decs3(torch.cat([out_enc_level1, x0], dim=1))
                outputs.append(x0 + inp_img)
            else:
                outputs.append(self.up_decs(out_dec_level3) + inp_img)
        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = inp_dec_level2 + out_enc_level2



        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        if self.kernel_learning[1]:
            level = 1
            b, c, H, W = out_dec_level2.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_, batch_list = window_partitionx(out_dec_level2, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_ = self.grids(out_dec_level2, level=level)
                # print(x_.shape)
            # pad_size = 15
            # x_ = torch.nn.functional.pad(x_, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
            x_fft = torch.fft.rfft2(x_)
            x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_ = self.kernel_ins_mid(x_)
            x_ = self.kernel_blks_mid(x_)
            x_ = self.kernel_outs_mid(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_ = torch.fft.irfft2(otf * x_fft)
            # x_ = x_[..., pad_size:-pad_size, pad_size:-pad_size].contiguous()
            x_ = self.ln1(x_)
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_ = window_reversex(x_, self.ker_size[level], H, W, batch_list)
            else:
                x_ = self.grids_inverse(x_, level=level)
            out_dec_level2 = self.proj_feature_outs_mid(torch.cat([out_dec_level2, x_], dim=1))

            outputs.append(self.up_decs_mid(out_dec_level2) + inp_img)
        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = inp_dec_level1 + out_enc_level1


        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        if self.kernel_learning[0]:
            level = 0
            b, c, H, W = out_dec_level1.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_, batch_list = window_partitionx(out_dec_level1, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_ = self.grids(out_dec_level1, level=level)
                # print(x_.shape)
            # pad_size = 33
            # x_ = torch.nn.functional.pad(x_, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
            x_fft = torch.fft.rfft2(x_)
            x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_ = self.kernel_ins_max(x_)
            x_ = self.kernel_blks_max(x_)
            x_ = self.kernel_outs_max(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_ = torch.fft.irfft2(otf * x_fft)
            # x_ = x_[..., pad_size:-pad_size, pad_size:-pad_size].contiguous()
            # print(x_.shape)
            x_ = self.ln0(x_)
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_ = window_reversex(x_, self.ker_size[level], H, W, batch_list)
            else:
                x_ = self.grids_inverse(x_, level=level)

            out_dec_level1 = self.proj_feature_outs_max(torch.cat([out_dec_level1, x_], dim=1))
            # print(inp_img.shape, inp_dec_level1.shape)
            if self.flip_transpose_max:
                out_dec_level1_ = torch.flip(out_dec_level1, dims=(-2, -1)).contiguous()
                out_dec_level1_ = torch.transpose(out_dec_level1_, dim0=-2, dim1=-1).contiguous()
            else:
                out_dec_level1_ = out_enc_level1
            outputs.append(self.up_decs_max(out_dec_level1_) + inp_img)
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        outputs.append(out_dec_level1)
        # print('xxxxxxxxxxxxxxxxxxx')
        return outputs
@ARCH_REGISTRY.register()
class uniDeblur2(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[6, 6, 12],
        baseblock=['localnaf', 'localnaf', 'localnaf'],
        ker_baseblock='EVSS',
        ffn_expansion_factor=3,
        bias=False,
        train_size=[256, 256],
        patch_size=[256, 256],
        ker_size=[[256, 256], [128, 128], [64, 64]],
        overlap_size=[[32, 32], [16, 16], [8, 8]],
        k_channels=[48, 96, 192],
        ker_start=1,
        window_size=8,
        global_fft=[False, False, False],
        patch_size_fft=[128, 128],
        kernel_learning=[False, False, False],
        dec_with_enc=False,
        geo_aug=False,
        flip_transpose_max=False
    ):
        super().__init__()
        self.kernel_learning = kernel_learning
        self.flip_transpose_max = flip_transpose_max
        self.geo_aug = geo_aug
        self.ker_start = ker_start
        self.patch_size = ker_size
        self.ker_size = ker_size
        self.out_channels = [dim, dim*2, dim*4]
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
        self.encoder = True
        self.dec_with_enc = dec_with_enc
        self.ln0 = LayerNorm(dim)
        self.ln1 = LayerNorm(dim*2)
        self.ln2 = LayerNorm(dim*4)
        if self.kernel_learning[2]:
            if self.geo_aug:
                self.kernel_ins = nn.ModuleList()
                self.kernel_blks = nn.ModuleList()
                self.kernel_outs = nn.ModuleList()
                for _ in range(4):
                    self.kernel_ins.append(nn.Sequential(
                        nn.Conv2d(dim * 1 * 2, dim * 2 * 2, 1, bias=bias)
                    ))
                    # self.kernel_blks = nn.Sequential(
                    #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
                    # )
                    kernel_blks = nn.Sequential()
                    for i in range(4):
                        block = get_basicblock(ker_baseblock, channels=dim * 2 * 2, num_heads=1, sub_idx=i,
                                               train_size=[train_size[0] // 4, train_size[1] // 4],
                                               patch_size=[patch_size[0] // 4, patch_size[1] // 4])
                        kernel_blks.add_module(f"block{i}", block)
                    self.kernel_blks.append(kernel_blks)
                    self.kernel_outs.append(nn.Sequential(
                        LayerNorm(dim * 2 * 2),
                        nn.Conv2d(dim * 2 * 2, dim * 1 * 2, 1, bias=bias)
                    ))
            else:
                self.kernel_ins = nn.Sequential(
                    nn.Conv2d(dim * 4 * 2, k_channels[2], 1, bias=bias)
                )
                # self.kernel_blks = nn.Sequential(
                #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
                # )
                self.kernel_blks = nn.Sequential()
                for i in range(4):
                    block = get_basicblock(ker_baseblock, channels=k_channels[2], num_heads=1, sub_idx=i,
                                           train_size=[train_size[0]//4, train_size[1]//4],
                                           patch_size=[patch_size[0]//4, patch_size[1]//4])
                    self.kernel_blks.add_module(f"block{i}", block)

                self.kernel_outs = nn.Sequential(
                    LayerNorm(k_channels[2]),
                    nn.Conv2d(k_channels[2], dim * 4 * 2, 1, bias=bias)
                )

            self.proj_feature_outs = nn.Sequential(
                nn.Conv2d(dim * 4 * 2, dim * 4, 1, bias=bias)
            )
            if self.dec_with_enc:
                self.up_decs1 = nn.Sequential(
                    nn.Conv2d(dim * 4 * 2, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2)
                )
                self.up_decs2 = nn.Sequential(
                    nn.Conv2d(dim * 2 * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2)
                )
                self.up_decs3 = nn.Sequential(
                    nn.Conv2d(in_channels=dim * 2, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
            else:
                self.up_decs = nn.Sequential(
                    nn.Conv2d(dim * 4, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
        if self.kernel_learning[1]:
            self.kernel_ins_mid = nn.Sequential(
                nn.Conv2d(dim * 2 * 2, k_channels[1], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks_mid = nn.Sequential()
            for i in range(4):
                block = get_basicblock(ker_baseblock, channels=k_channels[1], num_heads=1, sub_idx=i,
                                       train_size=[train_size[0] // 2, train_size[1] // 2],
                                       patch_size=[patch_size[0] // 2, patch_size[1] // 2])
                self.kernel_blks_mid.add_module(f"block{i}", block)

            self.kernel_outs_mid = nn.Sequential(
                LayerNorm(k_channels[1]),
                nn.Conv2d(k_channels[1], dim * 2 * 2, 1, bias=bias)
            )
            self.up_decs_mid = nn.Sequential(
                nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=bias)
            )
            self.proj_feature_outs_mid = nn.Sequential(
                nn.Conv2d(dim * 2 * 2, dim * 2, 1, bias=bias)
            )
        if self.kernel_learning[0]:
            self.kernel_ins_max = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, k_channels[0], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks_max = nn.Sequential()
            for i in range(4):
                block = get_basicblock(ker_baseblock, channels=k_channels[0], num_heads=1, sub_idx=i,
                                       train_size=[train_size[0] // 2, train_size[1] // 2],
                                       patch_size=[patch_size[0] // 2, patch_size[1] // 2])
                self.kernel_blks_max.add_module(f"block{i}", block)

            self.kernel_outs_max = nn.Sequential(
                LayerNorm(k_channels[0]),
                nn.Conv2d(k_channels[0], dim * 1 * 2, 1, bias=bias)
            )
            self.up_decs_max = nn.Sequential(
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=bias)
            )
            self.proj_feature_outs_max = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, dim * 1, 1, bias=bias)
            )
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.encoder_level1.add_module(f"block{i}", block)

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.encoder_level2.add_module(f"block{i}", block)

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.encoder_level3.add_module(f"block{i}", block)

        self.decoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.decoder_level3.add_module(f"block{i}", block)

        self.up3_2 = Upsample(int(dim * 2 ** 2))

        self.decoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.decoder_level2.add_module(f"block{i}", block)

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.decoder_level1.add_module(f"block{i}", block)

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
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
    def forward(self, inp_img, gt=None):
        outputs = []
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        if self.kernel_learning[2]:
            level = 2
            b, c, H, W = out_enc_level3.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_, batch_list = window_partitionx(out_enc_level3, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_ = self.grids(out_enc_level3, level=level)
                # print(x_.shape)
            if self.geo_aug:
                x_s = x_.chunk(4, dim=1)
                x_os = list()
                for j in range(4):
                    x_j = x_s[j]
                    if j % 2 == 1:
                        x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
                    if j % 2 == 0:
                        x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
                    x_fft = torch.fft.rfft2(x_j)
                    x_j = torch.cat([x_fft.real, x_fft.imag], dim=1)
                    x_j = self.kernel_ins[j](x_j)
                    x_j = self.kernel_blks[j](x_j)
                    x_j = self.kernel_outs[j](x_j)
                    x_real, x_imag = torch.chunk(x_j, 2, dim=1)
                    otf = torch.complex(x_real, x_imag)
                    # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
                    # print(x_fft.shape, otf.shape)
                    x_j = torch.fft.irfft2(otf * x_fft)
                    if j % 2 == 1:
                        x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
                    if j % 2 == 0:
                        x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
                    x_os.append(x_j)
                x_ = torch.cat(x_os, dim=1)
            else:
                x_fft = torch.fft.rfft2(x_)
                x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
                x_ = self.kernel_ins(x_)
                x_ = self.kernel_blks(x_)
                x_ = self.kernel_outs(x_)
                x_real, x_imag = torch.chunk(x_, 2, dim=1)
                otf = torch.complex(x_real, x_imag)
                # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
                # print(x_fft.shape, otf.shape)
                x_ = torch.fft.irfft2(otf * x_fft)
            # print(x_.shape)
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_ = window_reversex(x_, self.ker_size[level], H, W, batch_list)
            else:
                x_ = self.grids_inverse(x_, level=level)
            out_enc_level3 = self.proj_feature_outs(torch.cat([out_enc_level3, x_], dim=1))
            if self.dec_with_enc:
                x0 = self.up_decs1(torch.cat([out_enc_level3, x_], dim=1))
                x0 = self.up_decs2(torch.cat([out_enc_level2, x0], dim=1))
                x0 = self.up_decs3(torch.cat([out_enc_level1, x0], dim=1))
                outputs.append(x0 + inp_img)
            else:
                outputs.append(self.up_decs(out_enc_level3) + inp_img)

        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = inp_dec_level2 + out_enc_level2

        if self.kernel_learning[1]:
            level = 1
            b, c, H, W = inp_dec_level2.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_, batch_list = window_partitionx(inp_dec_level2, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_ = self.grids(inp_dec_level2, level=level)
                # print(x_.shape)

            x_fft = torch.fft.rfft2(x_)
            x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_ = self.kernel_ins_mid(x_)
            x_ = self.kernel_blks_mid(x_)
            x_ = self.kernel_outs_mid(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_ = torch.fft.irfft2(otf * x_fft)

            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_ = window_reversex(x_, self.ker_size[level], H, W, batch_list)
            else:
                x_ = self.grids_inverse(x_, level=level)
            inp_dec_level2 = self.proj_feature_outs_mid(torch.cat([inp_dec_level2, x_], dim=1))

            outputs.append(self.up_decs_mid(inp_dec_level2) + inp_img)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = inp_dec_level1 + out_enc_level1

        if self.kernel_learning[0]:
            level = 0
            b, c, H, W = inp_dec_level1.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_, batch_list = window_partitionx(inp_dec_level1, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_ = self.grids(inp_dec_level1, level=level)
                # print(x_.shape)

            x_fft = torch.fft.rfft2(x_)
            x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_ = self.kernel_ins_max(x_)
            x_ = self.kernel_blks_max(x_)
            x_ = self.kernel_outs_max(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_ = torch.fft.irfft2(otf * x_fft)
            # print(x_.shape)
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_ = window_reversex(x_, self.ker_size[level], H, W, batch_list)
            else:
                x_ = self.grids_inverse(x_, level=level)

            inp_dec_level1 = self.proj_feature_outs_max(torch.cat([inp_dec_level1, x_], dim=1))
            # print(inp_img.shape, inp_dec_level1.shape)
            if self.flip_transpose_max:
                inp_dec_level1_ = torch.flip(inp_dec_level1, dims=(-2, -1)).contiguous()
                inp_dec_level1_ = torch.transpose(inp_dec_level1_, dim0=-2, dim1=-1).contiguous()
            else:
                inp_dec_level1_ = inp_dec_level1
            outputs.append(self.up_decs_max(inp_dec_level1_) + inp_img)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img
        outputs.append(out_dec_level1)
        # print('xxxxxxxxxxxxxxxxxxx')
        return outputs
@ARCH_REGISTRY.register()
class uniDeblur(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[6, 6, 12],
        num_blocks2=[6, 6, 12],
        baseblock=['localnaf', 'localnaf', 'localnaf'],
        ker_baseblock=['EVSS', 'EVSS', 'EVSS'],
        ker_endblock=['Identity', 'Identity', 'Identity'],
        ker_firstblock=['Identity', 'Identity', 'Identity'],
        ker_num_blocks=[4, 4, 4],
        ker_num_endblocks=[1, 1, 1],
        ker_num_firstblocks=[1, 1, 1],
        ffn_expansion_factor=3,
        bias=False,
        train_size=[256, 256],
        patch_size=[256, 256],
        ker_size=[[256, 256], [128, 128], [64, 64]],
        overlap_size=[[32, 32], [16, 16], [8, 8]],
        k_channels=[48, 96, 192],
        ker_start=1,
        window_size=8,
        global_fft=[False, False, False],
        patch_size_fft=[128, 128],
        kernel_learning=[False, False, False],
        dec_with_enc=False,
        geo_aug=False,
        flip_transpose_max=False,
        phase_only=False,
        FFT_ReLU=False,
        supervised_mode='No',
        loss_weight=[0.5, 0.5, 0.5],
        fft_add=False
    ):
        super().__init__()
        self.fft_add = fft_add
        self.phase_only = phase_only
        self.FFT_ReLU = FFT_ReLU
        self.supervised_mode = supervised_mode
        self.loss_weight = loss_weight
        self.kernel_learning = kernel_learning
        self.flip_transpose_max = flip_transpose_max
        self.geo_aug = geo_aug
        self.ker_start = ker_start
        self.patch_size = ker_size
        self.ker_size = ker_size
        self.out_channels = [dim, dim*2, dim*4]
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
        self.encoder = True
        self.dec_with_enc = dec_with_enc
        # self.n0 = nn.Tanh()
        # self.n1 = nn.Tanh()
        # self.n2 = nn.Tanh()
        self.n0 = nn.Identity()
        self.n1 = nn.Identity()
        self.n2 = nn.Identity()
        # self.n0 = nn.BatchNorm2d(dim)
        # self.n1 = nn.BatchNorm2d(dim*2)
        # self.n2 = nn.BatchNorm2d(dim*4)
        if self.kernel_learning[2]:
            # if self.geo_aug:
            #     self.kernel_ins = nn.ModuleList()
            #     self.kernel_blks = nn.ModuleList()
            #     self.kernel_outs = nn.ModuleList()
            #     for _ in range(4):
            #         self.kernel_ins.append(nn.Sequential(
            #             nn.Conv2d(dim * 1 * 2, dim * 2 * 2, 1, bias=bias)
            #         ))
            #         # self.kernel_blks = nn.Sequential(
            #         #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            #         # )
            #         kernel_blks = nn.Sequential()
            #         for i in range(4):
            #             block = get_basicblock(ker_baseblock, channels=dim * 2 * 2, num_heads=1, sub_idx=i,
            #                                    train_size=[train_size[0] // 4, train_size[1] // 4],
            #                                    patch_size=[patch_size[0] // 4, patch_size[1] // 4])
            #             kernel_blks.add_module(f"block{i}", block)
            #         self.kernel_blks.append(kernel_blks)
            #         self.kernel_outs.append(nn.Sequential(
            #             LayerNorm(dim * 2 * 2),
            #             nn.Conv2d(dim * 2 * 2, dim * 1 * 2, 1, bias=bias)
            #         ))
            # else:
            self.kernel_ins = nn.Sequential(
                nn.Conv2d(dim * 4 * 2, k_channels[2], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks = nn.Sequential()
            for i in range(ker_num_blocks[2]):
                block = get_basicblock(ker_baseblock[2], channels=k_channels[2], num_heads=1, sub_idx=i,
                                       train_size=[ker_size[2][0], ker_size[2][1]],
                                       patch_size=[ker_size[2][0], ker_size[2][1]])
                self.kernel_blks.add_module(f"block{i}", block)
            self.kernel_end_blks = nn.Sequential()
            for i in range(ker_num_endblocks[2]):
                block = get_basicblock(ker_endblock[2], channels=k_channels[2], num_heads=1, sub_idx=i,
                                       train_size=[ker_size[2][0], ker_size[2][1]],
                                       patch_size=[ker_size[2][0], ker_size[2][1]])
                self.kernel_end_blks.add_module(f"block{i}", block)
            self.kernel_first_blks = nn.Sequential()
            for i in range(ker_num_firstblocks[2]):
                block = get_basicblock(ker_firstblock[2], channels=k_channels[2], num_heads=1, sub_idx=i,
                                       train_size=[ker_size[2][0], ker_size[2][1]],
                                       patch_size=[ker_size[2][0], ker_size[2][1]])
                self.kernel_first_blks.add_module(f"block{i}", block)
            self.kernel_outs = nn.Sequential(
                # LayerNorm(k_channels[2]),
                nn.Conv2d(k_channels[2], dim * 4 * 2, 1, bias=bias)
            )

            self.proj_feature_outs = nn.Sequential(
                nn.Conv2d(dim * 4 * 2, dim * 4, 1, bias=bias)
            )
            # if self.dec_with_enc:
            #     self.up_decs1 = nn.Sequential(
            #         nn.Conv2d(dim * 4 * 2, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2)
            #     )
            #     self.up_decs2 = nn.Sequential(
            #         nn.Conv2d(dim * 2 * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2)
            #     )
            #     self.up_decs3 = nn.Sequential(
            #         nn.Conv2d(in_channels=dim * 2, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
            #                   bias=bias)
            #     )
            # else:
            if self.supervised_mode == 'Yes':
                self.up_decs = nn.Sequential(
                    nn.Conv2d(dim * 4, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
            elif self.supervised_mode == 'MultiScale':
                self.up_decs = nn.Sequential(
                    nn.Conv2d(in_channels=dim * 4, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
        if self.kernel_learning[1]:
            self.kernel_ins_mid = nn.Sequential(
                nn.Conv2d(dim * 2 * 2, k_channels[1], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_first_blks_mid = nn.Sequential()
            for i in range(ker_num_firstblocks[1]):
                block = get_basicblock(ker_firstblock[1], channels=k_channels[1], num_heads=1, sub_idx=i,
                                       train_size=[ker_size[1][0], ker_size[1][1]],
                                       patch_size=[ker_size[1][0], ker_size[1][1]])
                self.kernel_first_blks_mid.add_module(f"block{i}", block)
            self.kernel_blks_mid = nn.Sequential()
            for i in range(ker_num_blocks[1]):
                block = get_basicblock(ker_baseblock[1], channels=k_channels[1], num_heads=1, sub_idx=i,
                                       train_size=[ker_size[1][0], ker_size[1][1]],
                                       patch_size=[ker_size[1][0], ker_size[1][1]])
                self.kernel_blks_mid.add_module(f"block{i}", block)
            self.kernel_end_blks_mid = nn.Sequential()
            for i in range(ker_num_endblocks[1]):
                block = get_basicblock(ker_endblock[1], channels=k_channels[1], num_heads=1, sub_idx=i,
                                       train_size=[ker_size[1][0], ker_size[1][1]],
                                       patch_size=[ker_size[1][0], ker_size[1][1]])
                self.kernel_end_blks_mid.add_module(f"block{i}", block)
            self.kernel_outs_mid = nn.Sequential(
                # LayerNorm(k_channels[1]),
                nn.Conv2d(k_channels[1], dim * 2 * 2, 1, bias=bias)
            )
            if self.supervised_mode == 'Yes':
                self.up_decs_mid = nn.Sequential(
                    nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
            elif self.supervised_mode == 'MultiScale':
                self.up_decs_mid = nn.Sequential(
                    nn.Conv2d(in_channels=dim*2, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )

            self.proj_feature_outs_mid = nn.Sequential(
                nn.Conv2d(dim * 2 * 2, dim * 2, 1, bias=bias)
            )
        if self.kernel_learning[0]:
            self.kernel_ins_max = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, k_channels[0], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_first_blks_max = nn.Sequential()
            for i in range(ker_num_firstblocks[0]):
                block = get_basicblock(ker_firstblock[0], channels=k_channels[0], num_heads=1, sub_idx=i,
                                       train_size=[ker_size[0][0], ker_size[0][1]],
                                       patch_size=[ker_size[0][0], ker_size[0][1]])
                self.kernel_first_blks_max.add_module(f"block{i}", block)
            self.kernel_blks_max = nn.Sequential()
            for i in range(ker_num_blocks[0]):
                block = get_basicblock(ker_baseblock[0], channels=k_channels[0], num_heads=1, sub_idx=i,
                                       train_size=[ker_size[0][0], ker_size[0][1]],
                                       patch_size=[ker_size[0][0], ker_size[0][1]])
                self.kernel_blks_max.add_module(f"block{i}", block)
            self.kernel_end_blks_max = nn.Sequential()
            for i in range(ker_num_endblocks[0]):
                block = get_basicblock(ker_endblock[0], channels=k_channels[0], num_heads=1, sub_idx=i,
                                       train_size=[ker_size[0][0], ker_size[0][1]],
                                       patch_size=[ker_size[0][0], ker_size[0][1]])
                self.kernel_end_blks_max.add_module(f"block{i}", block)
            self.kernel_outs_max = nn.Sequential(
                # LayerNorm(k_channels[0]),
                nn.Conv2d(k_channels[0], dim * 1 * 2, 1, bias=bias)
            )
            self.up_decs_max = nn.Sequential(
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=bias)
            )
            self.proj_feature_outs_max = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, dim * 1, 1, bias=bias)
            )
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.encoder_level1.add_module(f"block{i}", block)

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.encoder_level2.add_module(f"block{i}", block)

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.encoder_level3.add_module(f"block{i}", block)

        self.decoder_level3 = nn.Sequential()
        for i in range(num_blocks2[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.decoder_level3.add_module(f"block{i}", block)

        self.up3_2 = Upsample(int(dim * 2 ** 2))

        self.decoder_level2 = nn.Sequential()
        for i in range(num_blocks2[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.decoder_level2.add_module(f"block{i}", block)

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential()
        for i in range(num_blocks2[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.decoder_level1.add_module(f"block{i}", block)

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.loss_l1 = L1Loss()
        self.loss_freq = FourierLoss(0.01)
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
    def forward(self, inp_img, gt=None):
        outputs = []
        out_dict = {}
        outputs_small = []
        outputs_mid = []
        outputs_max = []
        inp_pry = kornia.geometry.transform.build_pyramid(inp_img, 3)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        if self.kernel_learning[2]:
            level = 2
            b, c, H, W = out_enc_level3.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_0, batch_list = window_partitionx(out_enc_level3, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_0 = self.grids(out_enc_level3, level=level)
                # print(x_.shape)
            # if self.geo_aug:
            #     x_s = x_0.chunk(4, dim=1)
            #     x_os = list()
            #     for j in range(4):
            #         x_j = x_s[j]
            #         if j % 2 == 1:
            #             x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
            #         if j % 2 == 0:
            #             x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
            #         x_fft = torch.fft.rfft2(x_j)
            #         x_j = torch.cat([x_fft.real, x_fft.imag], dim=1)
            #         x_j = self.kernel_ins[j](x_j)
            #         x_j = self.kernel_blks[j](x_j)
            #         x_j = self.kernel_outs[j](x_j)
            #         x_real, x_imag = torch.chunk(x_j, 2, dim=1)
            #         otf = torch.complex(x_real, x_imag)
            #         # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            #         # print(x_fft.shape, otf.shape)
            #         x_j = torch.fft.irfft2(otf * x_fft)
            #         if j % 2 == 1:
            #             x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
            #         if j % 2 == 0:
            #             x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
            #         x_os.append(x_j)
            #     x_1 = torch.cat(x_os, dim=1)
            # else:
            x_fft = torch.fft.rfft2(x_0)
            x_ = x_fft
            if self.phase_only:
                x_ = x_ / (torch.abs(x_)+1e-6)
            x_ = torch.cat([x_.real, x_.imag], dim=1)
            if self.FFT_ReLU:
                x_ = torch.relu(x_) - 0.5 * x_
            x_ = self.kernel_ins(x_)
            x_ = self.kernel_first_blks(x_)
            x_ = self.kernel_blks(x_)
            x_ = self.kernel_end_blks(x_)
            x_ = self.kernel_outs(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            if self.fft_add:
                x_1 = torch.fft.irfft2(otf + x_fft)
            else:
                x_1 = torch.fft.irfft2(otf * x_fft)
            if self.geo_aug:
                x_s = x_1.chunk(4, dim=1)
                x_os = list()
                for j in range(4):
                    x_j = x_s[j]
                    if j % 2 == 1:
                        x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
                    if j % 2 == 0:
                        x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
                    x_os.append(x_j)
                x_1 = torch.cat(x_os, dim=1)
            # print(x_.shape)
            x_1 = self.n2(x_1)
            # out_enc_level3 = self.proj_feature_outs(torch.cat([x_0, x_1], dim=1))
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_1 = window_reversex(x_1, self.ker_size[level], H, W, batch_list)
            else:
                x_1 = self.grids_inverse(x_1, level=level)
            out_enc_level3 = self.proj_feature_outs(torch.cat([out_enc_level3, x_1], dim=1))
            # if self.dec_with_enc:
            #     x0 = self.up_decs1(torch.cat([out_enc_level3, x_1], dim=1))
            #     x0 = self.up_decs2(torch.cat([out_enc_level2, x0], dim=1))
            #     x0 = self.up_decs3(torch.cat([out_enc_level1, x0], dim=1))
            #     outputs.append(x0 + inp_img)
            # else:
            if self.training:
                if self.supervised_mode == 'Yes':
                    outputs_small.append(self.up_decs(out_enc_level3) + inp_img)
                elif self.supervised_mode == 'MultiScale':
                    outputs_small.append(self.up_decs(out_enc_level3) + inp_pry[2])

        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = inp_dec_level2 + out_enc_level2

        if self.kernel_learning[1]:
            level = 1
            b, c, H, W = inp_dec_level2.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_0, batch_list = window_partitionx(inp_dec_level2, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_0 = self.grids(inp_dec_level2, level=level)
                # print(x_.shape)

            x_fft = torch.fft.rfft2(x_0)
            x_ = x_fft
            if self.phase_only:
                x_ = x_ / (torch.abs(x_)+1e-6)
            x_ = torch.cat([x_.real, x_.imag], dim=1)
            if self.FFT_ReLU:
                x_ = torch.relu(x_) - 0.5 * x_
            x_ = self.kernel_ins_mid(x_)
            x_ = self.kernel_first_blks_mid(x_)
            x_ = self.kernel_blks_mid(x_)
            x_ = self.kernel_end_blks_mid(x_)
            x_ = self.kernel_outs_mid(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            if self.fft_add:
                x_1 = torch.fft.irfft2(otf + x_fft)
            else:
                x_1 = torch.fft.irfft2(otf * x_fft)
            if self.geo_aug:
                x_s = x_1.chunk(4, dim=1)
                x_os = list()
                for j in range(4):
                    x_j = x_s[j]
                    if j % 2 == 1:
                        x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
                    if j % 2 == 0:
                        x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
                    x_os.append(x_j)
                x_1 = torch.cat(x_os, dim=1)
            x_1 = self.n1(x_1)

            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_1 = window_reversex(x_1, self.ker_size[level], H, W, batch_list)
            else:
                x_1 = self.grids_inverse(x_1, level=level)
            inp_dec_level2 = self.proj_feature_outs_mid(torch.cat([inp_dec_level2, x_1], dim=1))
            if self.training:
                if self.supervised_mode == 'Yes':
                    outputs_mid.append(self.up_decs_mid(inp_dec_level2) + inp_img)
                elif self.supervised_mode == 'MultiScale':
                    outputs_mid.append(self.up_decs_mid(inp_dec_level2) + inp_pry[1])

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = inp_dec_level1 + out_enc_level1

        if self.kernel_learning[0]:
            level = 0
            b, c, H, W = inp_dec_level1.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_0, batch_list = window_partitionx(inp_dec_level1, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_0 = self.grids(inp_dec_level1, level=level)
                # print(x_.shape)

            x_fft = torch.fft.rfft2(x_0)
            x_ = x_fft
            if self.phase_only:
                x_ = x_ / (torch.abs(x_)+1e-6)
            x_ = torch.cat([x_.real, x_.imag], dim=1)
            if self.FFT_ReLU:
                x_ = torch.relu(x_) - 0.5 * x_
            x_ = self.kernel_ins_max(x_)
            x_ = self.kernel_first_blks_max(x_)
            x_ = self.kernel_blks_max(x_)
            x_ = self.kernel_end_blks_max(x_)
            x_ = self.kernel_outs_max(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            if self.fft_add:
                x_1 = torch.fft.irfft2(otf + x_fft)
            else:
                x_1 = torch.fft.irfft2(otf * x_fft)
            # print(x_.shape)
            if self.geo_aug:
                x_s = x_1.chunk(4, dim=1)
                x_os = list()
                for j in range(4):
                    x_j = x_s[j]
                    if j % 2 == 1:
                        x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
                    if j % 2 == 0:
                        x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
                    x_os.append(x_j)
                x_1 = torch.cat(x_os, dim=1)
            x_1 = self.n0(x_1)
            # print(self.fft_add)
            # print(inp_img.shape, inp_dec_level1.shape)
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_1 = window_reversex(x_1, self.ker_size[level], H, W, batch_list)
            else:
                x_1 = self.grids_inverse(x_1, level=level)


            inp_dec_level1 = self.proj_feature_outs_max(torch.cat([inp_dec_level1, x_1], dim=1))
            if self.training:
                if self.flip_transpose_max:
                    inp_dec_level1_ = torch.flip(inp_dec_level1, dims=(-2, -1)).contiguous()
                    inp_dec_level1_ = torch.transpose(inp_dec_level1_, dim0=-2, dim1=-1).contiguous()
                else:
                    inp_dec_level1_ = inp_dec_level1
                outputs_max.append(self.up_decs_max(inp_dec_level1_) + inp_img)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img
        outputs.append(out_dec_level1)
        # print('xxxxxxxxxxxxxxxxxxx')
        out_dict['restored'] = outputs
        if self.training and gt is not None and len(outputs_max) + len(outputs_mid) + len(outputs_small) > 0:
            if self.supervised_mode == 'Yes':
                gt_pry = [gt, gt, gt]
            elif self.supervised_mode == 'MultiScale':
                gt_pry = kornia.geometry.transform.build_pyramid(gt, 3)
            if len(outputs_small) > 0:
                loss_small = self.loss_l1(outputs_small, gt_pry[2]) + self.loss_freq(outputs_small, gt_pry[2])
            else:
                loss_small = 0.
            if len(outputs_mid) > 0:
                loss_mid = self.loss_l1(outputs_mid, gt_pry[1]) + self.loss_freq(outputs_mid, gt_pry[1])
            else:
                loss_mid = 0.
            if len(outputs_max) > 0:
                loss_max = self.loss_l1(outputs_max, gt) + self.loss_freq(outputs_max, gt)
            else:
                loss_max = 0.
            loss = loss_max * self.loss_weight[0] + loss_mid * self.loss_weight[1] + loss_small * self.loss_weight[2]
            out_dict['loss_model'] = loss
        return out_dict

@ARCH_REGISTRY.register()
class uniDeblur_vis(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[6, 6, 12],
        num_blocks2=[6, 6, 12],
        baseblock=['localnaf', 'localnaf', 'localnaf'],
        ker_baseblock=['EVSS', 'EVSS', 'EVSS'],
        ker_endblock=['Identity', 'Identity', 'Identity'],
        ker_firstblock=['Identity', 'Identity', 'Identity'],
        ker_num_blocks=[4, 4, 4],
        ker_num_endblocks=[1, 1, 1],
        ker_num_firstblocks=[1, 1, 1],
        ffn_expansion_factor=3,
        bias=False,
        train_size=[256, 256],
        patch_size=[256, 256],
        ker_size=[[256, 256], [128, 128], [64, 64]],
        overlap_size=[[32, 32], [16, 16], [8, 8]],
        k_channels=[48, 96, 192],
        ker_start=1,
        window_size=8,
        global_fft=[False, False, False],
        patch_size_fft=[128, 128],
        kernel_learning=[False, False, False],
        dec_with_enc=False,
        geo_aug=False,
        flip_transpose_max=False,
        phase_only=False,
        FFT_ReLU=False,
        supervised_mode='No',
        loss_weight=[0.5, 0.5, 0.5],
        fft_add=False
    ):
        super().__init__()
        self.fft_add = fft_add
        self.phase_only = phase_only
        self.FFT_ReLU = FFT_ReLU
        self.supervised_mode = supervised_mode
        self.loss_weight = loss_weight
        self.kernel_learning = kernel_learning
        self.flip_transpose_max = flip_transpose_max
        self.geo_aug = geo_aug
        self.ker_start = ker_start
        self.patch_size = ker_size
        self.ker_size = ker_size
        self.out_channels = [dim, dim*2, dim*4]
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
        self.encoder = True
        self.dec_with_enc = dec_with_enc
        # self.n0 = nn.Tanh()
        # self.n1 = nn.Tanh()
        # self.n2 = nn.Tanh()
        self.n0 = nn.Identity()
        self.n1 = nn.Identity()
        self.n2 = nn.Identity()
        # self.n0 = nn.BatchNorm2d(dim)
        # self.n1 = nn.BatchNorm2d(dim*2)
        # self.n2 = nn.BatchNorm2d(dim*4)
        if self.kernel_learning[2]:
            # if self.geo_aug:
            #     self.kernel_ins = nn.ModuleList()
            #     self.kernel_blks = nn.ModuleList()
            #     self.kernel_outs = nn.ModuleList()
            #     for _ in range(4):
            #         self.kernel_ins.append(nn.Sequential(
            #             nn.Conv2d(dim * 1 * 2, dim * 2 * 2, 1, bias=bias)
            #         ))
            #         # self.kernel_blks = nn.Sequential(
            #         #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            #         # )
            #         kernel_blks = nn.Sequential()
            #         for i in range(4):
            #             block = get_basicblock(ker_baseblock, channels=dim * 2 * 2, num_heads=1, sub_idx=i,
            #                                    train_size=[train_size[0] // 4, train_size[1] // 4],
            #                                    patch_size=[patch_size[0] // 4, patch_size[1] // 4])
            #             kernel_blks.add_module(f"block{i}", block)
            #         self.kernel_blks.append(kernel_blks)
            #         self.kernel_outs.append(nn.Sequential(
            #             LayerNorm(dim * 2 * 2),
            #             nn.Conv2d(dim * 2 * 2, dim * 1 * 2, 1, bias=bias)
            #         ))
            # else:
            self.kernel_ins = nn.Sequential(
                nn.Conv2d(dim * 4 * 2, k_channels[2], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks = nn.Sequential()
            for i in range(ker_num_blocks[2]):
                block = get_basicblock(ker_baseblock[2], channels=k_channels[2], num_heads=1, sub_idx=i,
                                       train_size=[ker_size[2][0], ker_size[2][1]],
                                       patch_size=[ker_size[2][0], ker_size[2][1]])
                self.kernel_blks.add_module(f"block{i}", block)
            self.kernel_end_blks = nn.Sequential()
            for i in range(ker_num_endblocks[2]):
                block = get_basicblock(ker_endblock[2], channels=k_channels[2], num_heads=1, sub_idx=i,
                                       train_size=[ker_size[2][0], ker_size[2][1]],
                                       patch_size=[ker_size[2][0], ker_size[2][1]])
                self.kernel_end_blks.add_module(f"block{i}", block)
            self.kernel_first_blks = nn.Sequential()
            for i in range(ker_num_firstblocks[2]):
                block = get_basicblock(ker_firstblock[2], channels=k_channels[2], num_heads=1, sub_idx=i,
                                       train_size=[ker_size[2][0], ker_size[2][1]],
                                       patch_size=[ker_size[2][0], ker_size[2][1]])
                self.kernel_first_blks.add_module(f"block{i}", block)
            self.kernel_outs = nn.Sequential(
                # LayerNorm(k_channels[2]),
                nn.Conv2d(k_channels[2], dim * 4 * 2, 1, bias=bias)
            )

            self.proj_feature_outs = nn.Sequential(
                nn.Conv2d(dim * 4 * 2, dim * 4, 1, bias=bias)
            )
            # if self.dec_with_enc:
            #     self.up_decs1 = nn.Sequential(
            #         nn.Conv2d(dim * 4 * 2, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2)
            #     )
            #     self.up_decs2 = nn.Sequential(
            #         nn.Conv2d(dim * 2 * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2)
            #     )
            #     self.up_decs3 = nn.Sequential(
            #         nn.Conv2d(in_channels=dim * 2, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
            #                   bias=bias)
            #     )
            # else:
            if self.supervised_mode == 'Yes':
                self.up_decs = nn.Sequential(
                    nn.Conv2d(dim * 4, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
            elif self.supervised_mode == 'MultiScale':
                self.up_decs = nn.Sequential(
                    nn.Conv2d(in_channels=dim * 4, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
        if self.kernel_learning[1]:
            self.kernel_ins_mid = nn.Sequential(
                nn.Conv2d(dim * 2 * 2, k_channels[1], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_first_blks_mid = nn.Sequential()
            for i in range(ker_num_firstblocks[1]):
                block = get_basicblock(ker_firstblock[1], channels=k_channels[1], num_heads=1, sub_idx=i,
                                       train_size=[ker_size[1][0], ker_size[1][1]],
                                       patch_size=[ker_size[1][0], ker_size[1][1]])
                self.kernel_first_blks_mid.add_module(f"block{i}", block)
            self.kernel_blks_mid = nn.Sequential()
            for i in range(ker_num_blocks[1]):
                block = get_basicblock(ker_baseblock[1], channels=k_channels[1], num_heads=1, sub_idx=i,
                                       train_size=[ker_size[1][0], ker_size[1][1]],
                                       patch_size=[ker_size[1][0], ker_size[1][1]])
                self.kernel_blks_mid.add_module(f"block{i}", block)
            self.kernel_end_blks_mid = nn.Sequential()
            for i in range(ker_num_endblocks[1]):
                block = get_basicblock(ker_endblock[1], channels=k_channels[1], num_heads=1, sub_idx=i,
                                       train_size=[ker_size[1][0], ker_size[1][1]],
                                       patch_size=[ker_size[1][0], ker_size[1][1]])
                self.kernel_end_blks_mid.add_module(f"block{i}", block)
            self.kernel_outs_mid = nn.Sequential(
                # LayerNorm(k_channels[1]),
                nn.Conv2d(k_channels[1], dim * 2 * 2, 1, bias=bias)
            )
            if self.supervised_mode == 'Yes':
                self.up_decs_mid = nn.Sequential(
                    nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
            elif self.supervised_mode == 'MultiScale':
                self.up_decs_mid = nn.Sequential(
                    nn.Conv2d(in_channels=dim*2, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )

            self.proj_feature_outs_mid = nn.Sequential(
                nn.Conv2d(dim * 2 * 2, dim * 2, 1, bias=bias)
            )
        if self.kernel_learning[0]:
            self.kernel_ins_max = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, k_channels[0], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_first_blks_max = nn.Sequential()
            for i in range(ker_num_firstblocks[0]):
                block = get_basicblock(ker_firstblock[0], channels=k_channels[0], num_heads=1, sub_idx=i,
                                       train_size=[ker_size[0][0], ker_size[0][1]],
                                       patch_size=[ker_size[0][0], ker_size[0][1]])
                self.kernel_first_blks_max.add_module(f"block{i}", block)
            self.kernel_blks_max = nn.Sequential()
            for i in range(ker_num_blocks[0]):
                block = get_basicblock(ker_baseblock[0], channels=k_channels[0], num_heads=1, sub_idx=i,
                                       train_size=[ker_size[0][0], ker_size[0][1]],
                                       patch_size=[ker_size[0][0], ker_size[0][1]])
                self.kernel_blks_max.add_module(f"block{i}", block)
            self.kernel_end_blks_max = nn.Sequential()
            for i in range(ker_num_endblocks[0]):
                block = get_basicblock(ker_endblock[0], channels=k_channels[0], num_heads=1, sub_idx=i,
                                       train_size=[ker_size[0][0], ker_size[0][1]],
                                       patch_size=[ker_size[0][0], ker_size[0][1]])
                self.kernel_end_blks_max.add_module(f"block{i}", block)
            self.kernel_outs_max = nn.Sequential(
                # LayerNorm(k_channels[0]),
                nn.Conv2d(k_channels[0], dim * 1 * 2, 1, bias=bias)
            )
            self.up_decs_max = nn.Sequential(
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=bias)
            )
            self.proj_feature_outs_max = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, dim * 1, 1, bias=bias)
            )
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.encoder_level1.add_module(f"block{i}", block)

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.encoder_level2.add_module(f"block{i}", block)

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.encoder_level3.add_module(f"block{i}", block)

        self.decoder_level3 = nn.Sequential()
        for i in range(num_blocks2[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.decoder_level3.add_module(f"block{i}", block)

        self.up3_2 = Upsample(int(dim * 2 ** 2))

        self.decoder_level2 = nn.Sequential()
        for i in range(num_blocks2[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.decoder_level2.add_module(f"block{i}", block)

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential()
        for i in range(num_blocks2[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.decoder_level1.add_module(f"block{i}", block)

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.loss_l1 = L1Loss()
        self.loss_freq = FourierLoss(0.01)
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
    def forward(self, inp_img, gt=None):
        outputs = []
        out_dict = {}
        outputs_small = []
        outputs_mid = []
        outputs_max = []
        otfs = []

        x_ins = []
        x_outs = []

        inp_pry = kornia.geometry.transform.build_pyramid(inp_img, 3)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        if self.kernel_learning[2]:
            level = 2
            b, c, H, W = out_enc_level3.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_0, batch_list = window_partitionx(out_enc_level3, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_0 = self.grids(out_enc_level3, level=level)
                # print(x_.shape)
            # if self.geo_aug:
            #     x_s = x_0.chunk(4, dim=1)
            #     x_os = list()
            #     for j in range(4):
            #         x_j = x_s[j]
            #         if j % 2 == 1:
            #             x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
            #         if j % 2 == 0:
            #             x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
            #         x_fft = torch.fft.rfft2(x_j)
            #         x_j = torch.cat([x_fft.real, x_fft.imag], dim=1)
            #         x_j = self.kernel_ins[j](x_j)
            #         x_j = self.kernel_blks[j](x_j)
            #         x_j = self.kernel_outs[j](x_j)
            #         x_real, x_imag = torch.chunk(x_j, 2, dim=1)
            #         otf = torch.complex(x_real, x_imag)
            #         # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            #         # print(x_fft.shape, otf.shape)
            #         x_j = torch.fft.irfft2(otf * x_fft)
            #         if j % 2 == 1:
            #             x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
            #         if j % 2 == 0:
            #             x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
            #         x_os.append(x_j)
            #     x_1 = torch.cat(x_os, dim=1)
            # else:
            x_fft = torch.fft.rfft2(x_0)
            x_ = x_fft
            if self.phase_only:
                x_ = x_ / (torch.abs(x_)+1e-6)
            x_ = torch.cat([x_.real, x_.imag], dim=1)
            if self.FFT_ReLU:
                x_ = torch.relu(x_) - 0.5 * x_
            x_ = self.kernel_ins(x_)
            x_ = self.kernel_first_blks(x_)
            x_ = self.kernel_blks(x_)
            x_ = self.kernel_end_blks(x_)
            x_ = self.kernel_outs(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            if self.fft_add:
                x_1 = torch.fft.irfft2(otf + x_fft)
            else:
                x_1 = torch.fft.irfft2(otf * x_fft)
            if self.geo_aug:
                x_s = x_1.chunk(4, dim=1)
                x_os = list()
                for j in range(4):
                    x_j = x_s[j]
                    if j % 2 == 1:
                        x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
                    if j % 2 == 0:
                        x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
                    x_os.append(x_j)
                x_1 = torch.cat(x_os, dim=1)
            # print(x_.shape)
            x_1 = self.n2(x_1)
            # out_enc_level3 = self.proj_feature_outs(torch.cat([x_0, x_1], dim=1))
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_1 = window_reversex(x_1, self.ker_size[level], H, W, batch_list)
            else:
                x_1 = self.grids_inverse(x_1, level=level)
            out_enc_level3 = self.proj_feature_outs(torch.cat([out_enc_level3, x_1], dim=1))
            # if self.dec_with_enc:
            #     x0 = self.up_decs1(torch.cat([out_enc_level3, x_1], dim=1))
            #     x0 = self.up_decs2(torch.cat([out_enc_level2, x0], dim=1))
            #     x0 = self.up_decs3(torch.cat([out_enc_level1, x0], dim=1))
            #     outputs.append(x0 + inp_img)
            # else:
            if self.training:
                if self.supervised_mode == 'Yes':
                    outputs_small.append(self.up_decs(out_enc_level3) + inp_img)
                elif self.supervised_mode == 'MultiScale':
                    outputs_small.append(self.up_decs(out_enc_level3) + inp_pry[2])
            x_ins.append(x_0)
            x_outs.append(x_1)
            otfs.append(otf)
        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = inp_dec_level2 + out_enc_level2

        if self.kernel_learning[1]:
            level = 1
            b, c, H, W = inp_dec_level2.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_0, batch_list = window_partitionx(inp_dec_level2, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_0 = self.grids(inp_dec_level2, level=level)
                # print(x_.shape)

            x_fft = torch.fft.rfft2(x_0)
            x_ = x_fft
            if self.phase_only:
                x_ = x_ / (torch.abs(x_)+1e-6)
            x_ = torch.cat([x_.real, x_.imag], dim=1)
            if self.FFT_ReLU:
                x_ = torch.relu(x_) - 0.5 * x_
            x_ = self.kernel_ins_mid(x_)
            x_ = self.kernel_first_blks_mid(x_)
            x_ = self.kernel_blks_mid(x_)
            x_ = self.kernel_end_blks_mid(x_)
            x_ = self.kernel_outs_mid(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            if self.fft_add:
                x_1 = torch.fft.irfft2(otf + x_fft)
            else:
                x_1 = torch.fft.irfft2(otf * x_fft)
            if self.geo_aug:
                x_s = x_1.chunk(4, dim=1)
                x_os = list()
                for j in range(4):
                    x_j = x_s[j]
                    if j % 2 == 1:
                        x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
                    if j % 2 == 0:
                        x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
                    x_os.append(x_j)
                x_1 = torch.cat(x_os, dim=1)
            x_1 = self.n1(x_1)

            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_1 = window_reversex(x_1, self.ker_size[level], H, W, batch_list)
            else:
                x_1 = self.grids_inverse(x_1, level=level)
            inp_dec_level2 = self.proj_feature_outs_mid(torch.cat([inp_dec_level2, x_1], dim=1))
            if self.training:
                if self.supervised_mode == 'Yes':
                    outputs_mid.append(self.up_decs_mid(inp_dec_level2) + inp_img)
                elif self.supervised_mode == 'MultiScale':
                    outputs_mid.append(self.up_decs_mid(inp_dec_level2) + inp_pry[1])
            x_ins.append(x_0)
            x_outs.append(x_1)
            otfs.append(otf)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = inp_dec_level1 + out_enc_level1

        if self.kernel_learning[0]:
            level = 0
            b, c, H, W = inp_dec_level1.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_0, batch_list = window_partitionx(inp_dec_level1, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_0 = self.grids(inp_dec_level1, level=level)
                # print(x_.shape)

            x_fft = torch.fft.rfft2(x_0)
            x_ = x_fft
            if self.phase_only:
                x_ = x_ / (torch.abs(x_)+1e-6)
            x_ = torch.cat([x_.real, x_.imag], dim=1)
            if self.FFT_ReLU:
                x_ = torch.relu(x_) - 0.5 * x_
            x_ = self.kernel_ins_max(x_)
            x_ = self.kernel_first_blks_max(x_)
            x_ = self.kernel_blks_max(x_)
            x_ = self.kernel_end_blks_max(x_)
            x_ = self.kernel_outs_max(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            if self.fft_add:
                x_1 = torch.fft.irfft2(otf + x_fft)
            else:
                x_1 = torch.fft.irfft2(otf * x_fft)
            # print(x_.shape)
            if self.geo_aug:
                x_s = x_1.chunk(4, dim=1)
                x_os = list()
                for j in range(4):
                    x_j = x_s[j]
                    if j % 2 == 1:
                        x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
                    if j % 2 == 0:
                        x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
                    x_os.append(x_j)
                x_1 = torch.cat(x_os, dim=1)
            x_1 = self.n0(x_1)

            # print(inp_img.shape, inp_dec_level1.shape)
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_1 = window_reversex(x_1, self.ker_size[level], H, W, batch_list)
            else:
                x_1 = self.grids_inverse(x_1, level=level)


            inp_dec_level1 = self.proj_feature_outs_max(torch.cat([inp_dec_level1, x_1], dim=1))
            if self.training:
                if self.flip_transpose_max:
                    inp_dec_level1_ = torch.flip(inp_dec_level1, dims=(-2, -1)).contiguous()
                    inp_dec_level1_ = torch.transpose(inp_dec_level1_, dim0=-2, dim1=-1).contiguous()
                else:
                    inp_dec_level1_ = inp_dec_level1
                outputs_max.append(self.up_decs_max(inp_dec_level1_) + inp_img)
            if self.flip_transpose_max:
                x_0 = torch.flip(x_0, dims=(-2, -1)).contiguous()
                x_0 = torch.transpose(x_0, dim0=-2, dim1=-1).contiguous()
                x_1 = torch.flip(x_1, dims=(-2, -1)).contiguous()
                x_1 = torch.transpose(x_1, dim0=-2, dim1=-1).contiguous()
            x_ins.append(x_0)
            x_outs.append(x_1)
            otfs.append(otf)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img
        outputs.append(out_dec_level1)
        # print('xxxxxxxxxxxxxxxxxxx')
        out_dict['restored'] = outputs
        out_dict['x_enc'] = [out_enc_level1, out_enc_level2]
        out_dict['x_dec'] = [inp_dec_level1, inp_dec_level2]
        out_dict['x_in'] = x_ins[::-1]
        out_dict['x_out'] = x_outs[::-1]
        kernels = []
        kernels_fft = []

        for i, otf in enumerate(otfs):
            psf = torch.fft.irfft2(otf)
            kernel_fft = torch.fft.fft2(psf)
            kernel_fft = torch.log(torch.fft.fftshift(torch.abs(kernel_fft) + 1e-7, dim=[-2, -1]))

            psf = torch.fft.ifftshift(psf, dim=[-2, -1])

            psf = torch.nn.functional.pad(psf, (0, 1, 0, 1))
            ker = kornia.geometry.center_crop(psf, [65, 65])
            kernels.append(ker)
            kernels_fft.append(kernel_fft)
        out_dict['kernel'] = kernels[::-1]
        out_dict['kernel_fft'] = kernels_fft[::-1]
        if self.training and gt is not None and len(outputs_max) + len(outputs_mid) + len(outputs_small) > 0:
            if self.supervised_mode == 'Yes':
                gt_pry = [gt, gt, gt]
            elif self.supervised_mode == 'MultiScale':
                gt_pry = kornia.geometry.transform.build_pyramid(gt, 3)
            if len(outputs_small) > 0:
                loss_small = self.loss_l1(outputs_small, gt_pry[2]) + self.loss_freq(outputs_small, gt_pry[2])
            else:
                loss_small = 0.
            if len(outputs_mid) > 0:
                loss_mid = self.loss_l1(outputs_mid, gt_pry[1]) + self.loss_freq(outputs_mid, gt_pry[1])
            else:
                loss_mid = 0.
            if len(outputs_max) > 0:
                loss_max = self.loss_l1(outputs_max, gt) + self.loss_freq(outputs_max, gt)
            else:
                loss_max = 0.
            loss = loss_max * self.loss_weight[0] + loss_mid * self.loss_weight[1] + loss_small * self.loss_weight[2]
            out_dict['loss_model'] = loss
        return out_dict
@ARCH_REGISTRY.register()
class uniDeblurLog(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[6, 6, 12],
        num_blocks2=[6, 6, 12],
        baseblock=['localnaf', 'localnaf', 'localnaf'],
        ker_baseblock='EVSS',
        ffn_expansion_factor=3,
        bias=False,
        train_size=[256, 256],
        patch_size=[256, 256],
        ker_size=[[256, 256], [128, 128], [64, 64]],
        overlap_size=[[32, 32], [16, 16], [8, 8]],
        k_channels=[48, 96, 192],
        ker_start=1,
        window_size=8,
        global_fft=[False, False, False],
        patch_size_fft=[128, 128],
        kernel_learning=[False, False, False],
        dec_with_enc=False,
        geo_aug=False,
        flip_transpose_max=False
    ):
        super().__init__()
        self.kernel_learning = kernel_learning
        self.flip_transpose_max = flip_transpose_max
        self.geo_aug = geo_aug
        self.ker_start = ker_start
        self.patch_size = ker_size
        self.ker_size = ker_size
        self.out_channels = [dim, dim*2, dim*4]
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
        self.encoder = True
        self.dec_with_enc = dec_with_enc
        # self.n0 = nn.Tanh()
        # self.n1 = nn.Tanh()
        # self.n2 = nn.Tanh()
        self.n0 = nn.Identity()
        self.n1 = nn.Identity()
        self.n2 = nn.Identity()
        # self.n0 = nn.BatchNorm2d(dim)
        # self.n1 = nn.BatchNorm2d(dim*2)
        # self.n2 = nn.BatchNorm2d(dim*4)
        if self.kernel_learning[2]:
            # if self.geo_aug:
            #     self.kernel_ins = nn.ModuleList()
            #     self.kernel_blks = nn.ModuleList()
            #     self.kernel_outs = nn.ModuleList()
            #     for _ in range(4):
            #         self.kernel_ins.append(nn.Sequential(
            #             nn.Conv2d(dim * 1 * 2, dim * 2 * 2, 1, bias=bias)
            #         ))
            #         # self.kernel_blks = nn.Sequential(
            #         #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            #         # )
            #         kernel_blks = nn.Sequential()
            #         for i in range(4):
            #             block = get_basicblock(ker_baseblock, channels=dim * 2 * 2, num_heads=1, sub_idx=i,
            #                                    train_size=[train_size[0] // 4, train_size[1] // 4],
            #                                    patch_size=[patch_size[0] // 4, patch_size[1] // 4])
            #             kernel_blks.add_module(f"block{i}", block)
            #         self.kernel_blks.append(kernel_blks)
            #         self.kernel_outs.append(nn.Sequential(
            #             LayerNorm(dim * 2 * 2),
            #             nn.Conv2d(dim * 2 * 2, dim * 1 * 2, 1, bias=bias)
            #         ))
            # else:
            self.kernel_ins = nn.Sequential(
                nn.Conv2d(dim * 4 * 2, k_channels[2], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks = nn.Sequential()
            for i in range(4):
                block = get_basicblock(ker_baseblock, channels=k_channels[2], num_heads=1, sub_idx=i,
                                       train_size=[ker_size[2][0], ker_size[2][1]],
                                       patch_size=[ker_size[2][0], ker_size[2][1]])
                self.kernel_blks.add_module(f"block{i}", block)

            self.kernel_outs = nn.Sequential(
                # LayerNorm(k_channels[2]),
                nn.Conv2d(k_channels[2], dim * 4 * 2, 1, bias=bias)
            )

            self.proj_feature_outs = nn.Sequential(
                nn.Conv2d(dim * 4 * 2, dim * 4, 1, bias=bias)
            )
            if self.dec_with_enc:
                self.up_decs1 = nn.Sequential(
                    nn.Conv2d(dim * 4 * 2, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2)
                )
                self.up_decs2 = nn.Sequential(
                    nn.Conv2d(dim * 2 * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2)
                )
                self.up_decs3 = nn.Sequential(
                    nn.Conv2d(in_channels=dim * 2, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
            else:
                self.up_decs = nn.Sequential(
                    nn.Conv2d(dim * 4, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
        if self.kernel_learning[1]:
            self.kernel_ins_mid = nn.Sequential(
                nn.Conv2d(dim * 2 * 2, k_channels[1], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks_mid = nn.Sequential()
            for i in range(4):
                block = get_basicblock(ker_baseblock, channels=k_channels[1], num_heads=1, sub_idx=i,
                                       train_size=[ker_size[1][0], ker_size[1][1]],
                                       patch_size=[ker_size[1][0], ker_size[1][1]])
                self.kernel_blks_mid.add_module(f"block{i}", block)

            self.kernel_outs_mid = nn.Sequential(
                # LayerNorm(k_channels[1]),
                nn.Conv2d(k_channels[1], dim * 2 * 2, 1, bias=bias)
            )
            self.up_decs_mid = nn.Sequential(
                nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=bias)
            )
            self.proj_feature_outs_mid = nn.Sequential(
                nn.Conv2d(dim * 2 * 2, dim * 2, 1, bias=bias)
            )
        if self.kernel_learning[0]:
            self.kernel_ins_max = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, k_channels[0], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks_max = nn.Sequential()
            for i in range(4):
                block = get_basicblock(ker_baseblock, channels=k_channels[0], num_heads=1, sub_idx=i,
                                       train_size=[ker_size[0][0], ker_size[0][1]],
                                       patch_size=[ker_size[0][0], ker_size[0][1]])
                self.kernel_blks_max.add_module(f"block{i}", block)

            self.kernel_outs_max = nn.Sequential(
                # LayerNorm(k_channels[0]),
                nn.Conv2d(k_channels[0], dim * 1 * 2, 1, bias=bias)
            )
            self.up_decs_max = nn.Sequential(
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=bias)
            )
            self.proj_feature_outs_max = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, dim * 1, 1, bias=bias)
            )
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.encoder_level1.add_module(f"block{i}", block)

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.encoder_level2.add_module(f"block{i}", block)

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.encoder_level3.add_module(f"block{i}", block)

        self.decoder_level3 = nn.Sequential()
        for i in range(num_blocks2[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.decoder_level3.add_module(f"block{i}", block)

        self.up3_2 = Upsample(int(dim * 2 ** 2))

        self.decoder_level2 = nn.Sequential()
        for i in range(num_blocks2[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.decoder_level2.add_module(f"block{i}", block)

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential()
        for i in range(num_blocks2[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.decoder_level1.add_module(f"block{i}", block)

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
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

    def log_fft(self, image, eps=math.exp(-7), mode='rfft'):  # 1e-6
        if mode == 'fft':
            image_fft = torch.fft.fft2(image)
        elif mode == 'rfft':
            image_fft = torch.fft.rfft2(image)
        else:
            image_fft = image
        # image_fft_shift = torch.fft.fftshift(image_fft, dim=[-2, -1])
        # image_fft_shift = torch.cat([image_fft_shift.real, image_fft_shift.imag], dim=1)
        mag = torch.clamp(torch.abs(image_fft), eps)
        phase = torch.angle(image_fft)
        # image_fft_shift_s = image_fft_shift / (image_fft_shift_abs+1e-7)
        mag_log = torch.log(mag)
        return mag_log, phase


    def inverse_log_fft(self, x_mag, x_phase, mode='irfft'):

        I_mag = torch.exp(x_mag)

        I_phase = torch.exp(1j * x_phase)
        Image = I_mag * I_phase
        if mode == 'ifft':
            image = torch.fft.ifft2(Image)
        elif mode == 'irfft':
            image = torch.fft.irfft2(Image)
        else:
            image = Image
        return image
    def forward(self, inp_img, gt=None):
        outputs = []
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        if self.kernel_learning[2]:
            level = 2
            b, c, H, W = out_enc_level3.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_0, batch_list = window_partitionx(out_enc_level3, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_0 = self.grids(out_enc_level3, level=level)
                # print(x_.shape)
            # if self.geo_aug:
            #     x_s = x_0.chunk(4, dim=1)
            #     x_os = list()
            #     for j in range(4):
            #         x_j = x_s[j]
            #         if j % 2 == 1:
            #             x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
            #         if j % 2 == 0:
            #             x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
            #         x_fft = torch.fft.rfft2(x_j)
            #         x_j = torch.cat([x_fft.real, x_fft.imag], dim=1)
            #         x_j = self.kernel_ins[j](x_j)
            #         x_j = self.kernel_blks[j](x_j)
            #         x_j = self.kernel_outs[j](x_j)
            #         x_real, x_imag = torch.chunk(x_j, 2, dim=1)
            #         otf = torch.complex(x_real, x_imag)
            #         # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            #         # print(x_fft.shape, otf.shape)
            #         x_j = torch.fft.irfft2(otf * x_fft)
            #         if j % 2 == 1:
            #             x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
            #         if j % 2 == 0:
            #             x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
            #         x_os.append(x_j)
            #     x_1 = torch.cat(x_os, dim=1)
            # else:
            # x_fft = torch.fft.rfft2(x_0)
            x_mag, x_phase = self.log_fft(x_0)
            x_ = torch.cat([x_mag, x_phase], dim=1)
            x_ = self.kernel_ins(x_)
            x_ = self.kernel_blks(x_)
            x_ = self.kernel_outs(x_)
            y_mag, y_phase = torch.chunk(x_, 2, dim=1)
            # otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_1 = self.inverse_log_fft(x_mag-y_mag, x_phase-y_phase)
            if self.geo_aug:
                x_s = x_1.chunk(4, dim=1)
                x_os = list()
                for j in range(4):
                    x_j = x_s[j]
                    if j % 2 == 1:
                        x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
                    if j % 2 == 0:
                        x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
                    x_os.append(x_j)
                x_1 = torch.cat(x_os, dim=1)
            # print(x_.shape)
            x_1 = self.n2(x_1)
            out_enc_level3 = self.proj_feature_outs(torch.cat([x_0, x_1], dim=1))
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                out_enc_level3 = window_reversex(out_enc_level3, self.ker_size[level], H, W, batch_list)
            else:
                out_enc_level3 = self.grids_inverse(out_enc_level3, level=level)
            if self.dec_with_enc:
                x0 = self.up_decs1(torch.cat([out_enc_level3, x_1], dim=1))
                x0 = self.up_decs2(torch.cat([out_enc_level2, x0], dim=1))
                x0 = self.up_decs3(torch.cat([out_enc_level1, x0], dim=1))
                outputs.append(x0 + inp_img)
            else:
                outputs.append(self.up_decs(out_enc_level3) + inp_img)

        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = inp_dec_level2 + out_enc_level2

        if self.kernel_learning[1]:
            level = 1
            b, c, H, W = inp_dec_level2.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_0, batch_list = window_partitionx(inp_dec_level2, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_0 = self.grids(inp_dec_level2, level=level)
                # print(x_.shape)

            # x_fft = torch.fft.rfft2(x_0)
            # x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_mag, x_phase = self.log_fft(x_0)
            x_ = torch.cat([x_mag, x_phase], dim=1)
            x_ = self.kernel_ins_mid(x_)
            x_ = self.kernel_blks_mid(x_)
            x_ = self.kernel_outs_mid(x_)
            y_mag, y_phase = torch.chunk(x_, 2, dim=1)
            x_1 = self.inverse_log_fft(x_mag - y_mag, x_phase - y_phase)
            # otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            # x_1 = torch.fft.irfft2(otf * x_fft)
            if self.geo_aug:
                x_s = x_1.chunk(4, dim=1)
                x_os = list()
                for j in range(4):
                    x_j = x_s[j]
                    if j % 2 == 1:
                        x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
                    if j % 2 == 0:
                        x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
                    x_os.append(x_j)
                x_1 = torch.cat(x_os, dim=1)
            x_1 = self.n1(x_1)
            inp_dec_level2 = self.proj_feature_outs_mid(torch.cat([x_0, x_1], dim=1))
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                inp_dec_level2 = window_reversex(inp_dec_level2, self.ker_size[level], H, W, batch_list)
            else:
                inp_dec_level2 = self.grids_inverse(inp_dec_level2, level=level)
            outputs.append(self.up_decs_mid(inp_dec_level2) + inp_img)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = inp_dec_level1 + out_enc_level1

        if self.kernel_learning[0]:
            level = 0
            b, c, H, W = inp_dec_level1.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_0, batch_list = window_partitionx(inp_dec_level1, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_0 = self.grids(inp_dec_level1, level=level)
                # print(x_.shape)

            # x_fft = torch.fft.rfft2(x_0)
            # x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_mag, x_phase = self.log_fft(x_0)
            x_ = torch.cat([x_mag, x_phase], dim=1)
            x_ = self.kernel_ins_max(x_)
            x_ = self.kernel_blks_max(x_)
            x_ = self.kernel_outs_max(x_)
            # x_real, x_imag = torch.chunk(x_, 2, dim=1)
            # otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            y_mag, y_phase = torch.chunk(x_, 2, dim=1)
            x_1 = self.inverse_log_fft(x_mag - y_mag, x_phase - y_phase)
            # print(x_.shape)
            if self.geo_aug:
                x_s = x_1.chunk(4, dim=1)
                x_os = list()
                for j in range(4):
                    x_j = x_s[j]
                    if j % 2 == 1:
                        x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
                    if j % 2 == 0:
                        x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
                    x_os.append(x_j)
                x_1 = torch.cat(x_os, dim=1)
            x_1 = self.n0(x_1)
            inp_dec_level1 = self.proj_feature_outs_max(torch.cat([x_0, x_1], dim=1))
            # print(inp_img.shape, inp_dec_level1.shape)
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                inp_dec_level1 = window_reversex(inp_dec_level1, self.ker_size[level], H, W, batch_list)
            else:
                inp_dec_level1 = self.grids_inverse(inp_dec_level1, level=level)
            if self.flip_transpose_max:
                inp_dec_level1_ = torch.flip(inp_dec_level1, dims=(-2, -1)).contiguous()
                inp_dec_level1_ = torch.transpose(inp_dec_level1_, dim0=-2, dim1=-1).contiguous()
            else:
                inp_dec_level1_ = inp_dec_level1
            outputs.append(self.up_decs_max(inp_dec_level1_) + inp_img)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img
        outputs.append(out_dec_level1)
        # print('xxxxxxxxxxxxxxxxxxx')
        return outputs
@ARCH_REGISTRY.register()
class uniDeblur3(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[6, 6, 12],
        num_blocks2=[6, 6, 12],
        baseblock=['localnaf', 'localnaf', 'localnaf'],
        ker_baseblock='EVSS',
        ffn_expansion_factor=3,
        bias=False,
        train_size=[256, 256],
        patch_size=[256, 256],
        ker_size=[[256, 256], [128, 128], [64, 64]],
        overlap_size=[[32, 32], [16, 16], [8, 8]],
        k_channels=[48, 96, 192],
        ker_start=1,
        window_size=8,
        global_fft=[False, False, False],
        patch_size_fft=[128, 128],
        kernel_learning=[False, False, False],
        dec_with_enc=False,
        geo_aug=False,
        flip_transpose_max=False
    ):
        super().__init__()
        self.kernel_learning = kernel_learning
        self.flip_transpose_max = flip_transpose_max
        self.geo_aug = geo_aug
        self.ker_start = ker_start
        self.patch_size = ker_size
        self.ker_size = ker_size
        self.out_channels = k_channels
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
        self.encoder = True
        self.dec_with_enc = dec_with_enc
        # self.n0 = nn.Tanh()
        # self.n1 = nn.Tanh()
        # self.n2 = nn.Tanh()
        self.n0 = nn.Identity()
        self.n1 = nn.Identity()
        self.n2 = nn.Identity()
        # self.n0 = nn.BatchNorm2d(dim)
        # self.n1 = nn.BatchNorm2d(dim*2)
        # self.n2 = nn.BatchNorm2d(dim*4)
        if self.kernel_learning[2]:
            # if self.geo_aug:
            #     self.kernel_ins = nn.ModuleList()
            #     self.kernel_blks = nn.ModuleList()
            #     self.kernel_outs = nn.ModuleList()
            #     for _ in range(4):
            #         self.kernel_ins.append(nn.Sequential(
            #             nn.Conv2d(dim * 1 * 2, dim * 2 * 2, 1, bias=bias)
            #         ))
            #         # self.kernel_blks = nn.Sequential(
            #         #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            #         # )
            #         kernel_blks = nn.Sequential()
            #         for i in range(4):
            #             block = get_basicblock(ker_baseblock, channels=dim * 2 * 2, num_heads=1, sub_idx=i,
            #                                    train_size=[train_size[0] // 4, train_size[1] // 4],
            #                                    patch_size=[patch_size[0] // 4, patch_size[1] // 4])
            #             kernel_blks.add_module(f"block{i}", block)
            #         self.kernel_blks.append(kernel_blks)
            #         self.kernel_outs.append(nn.Sequential(
            #             LayerNorm(dim * 2 * 2),
            #             nn.Conv2d(dim * 2 * 2, dim * 1 * 2, 1, bias=bias)
            #         ))
            # else:
            self.kernel_ins = nn.Sequential(
                nn.Conv2d(dim * 4 * 2, k_channels[2], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks = nn.Sequential()
            for i in range(4):
                block = get_basicblock(ker_baseblock, channels=k_channels[2], num_heads=1, sub_idx=i,
                                       train_size=[ker_size[2][0], ker_size[2][1]],
                                       patch_size=[ker_size[2][0], ker_size[2][1]])
                self.kernel_blks.add_module(f"block{i}", block)

            self.kernel_outs = nn.Sequential(
                LayerNorm(k_channels[2]),
                nn.Conv2d(k_channels[2], dim * 4 * 2, 1, bias=bias)
            )

            self.proj_feature_outs = nn.Sequential(
                nn.Conv2d(dim * 4 * 2, dim * 4, 1, bias=bias)
            )
            if self.dec_with_enc:
                self.up_decs1 = nn.Sequential(
                    nn.Conv2d(dim * 4 * 2, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2)
                )
                self.up_decs2 = nn.Sequential(
                    nn.Conv2d(dim * 2 * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2)
                )
                self.up_decs3 = nn.Sequential(
                    nn.Conv2d(in_channels=dim * 2, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
            else:
                self.up_decs = nn.Sequential(
                    nn.Conv2d(dim * 4, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
        if self.kernel_learning[1]:
            self.kernel_ins_mid = nn.Sequential(
                nn.Conv2d(dim * 4 * 2, k_channels[1], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks_mid = nn.Sequential()
            for i in range(4):
                block = get_basicblock(ker_baseblock, channels=k_channels[1], num_heads=1, sub_idx=i,
                                       train_size=[ker_size[1][0], ker_size[1][1]],
                                       patch_size=[ker_size[1][0], ker_size[1][1]])
                self.kernel_blks_mid.add_module(f"block{i}", block)

            self.kernel_outs_mid = nn.Sequential(
                LayerNorm(k_channels[1]),
                nn.Conv2d(k_channels[1], dim * 4 * 2, 1, bias=bias)
            )
            self.up_decs_mid = nn.Sequential(
                    nn.Conv2d(dim * 4, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
            self.proj_feature_outs_mid = nn.Sequential(
                nn.Conv2d(dim * 4 * 2, dim * 4, 1, bias=bias)
            )
        if self.kernel_learning[0]:
            self.kernel_ins_max = nn.Sequential(
                nn.Conv2d(dim * 4 * 2, k_channels[0], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks_max = nn.Sequential()
            for i in range(4):
                block = get_basicblock(ker_baseblock, channels=k_channels[0], num_heads=1, sub_idx=i,
                                       train_size=[ker_size[0][0], ker_size[0][1]],
                                       patch_size=[ker_size[0][0], ker_size[0][1]])
                self.kernel_blks_max.add_module(f"block{i}", block)

            self.kernel_outs_max = nn.Sequential(
                LayerNorm(k_channels[0]),
                nn.Conv2d(k_channels[0], dim * 4 * 2, 1, bias=bias)
            )
            self.up_decs_max = nn.Sequential(
                    nn.Conv2d(dim * 4, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
            self.proj_feature_outs_max = nn.Sequential(
                nn.Conv2d(dim * 4 * 2, dim * 4, 1, bias=bias)
            )
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.encoder_level1.add_module(f"block{i}", block)

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.encoder_level2.add_module(f"block{i}", block)

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.encoder_level3.add_module(f"block{i}", block)

        self.decoder_level3 = nn.Sequential()
        for i in range(num_blocks2[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.decoder_level3.add_module(f"block{i}", block)

        self.up3_2 = Upsample(int(dim * 2 ** 2))

        self.decoder_level2 = nn.Sequential()
        for i in range(num_blocks2[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.decoder_level2.add_module(f"block{i}", block)

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential()
        for i in range(num_blocks2[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.decoder_level1.add_module(f"block{i}", block)

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
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
    def forward(self, inp_img, gt=None):
        outputs = []
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        if self.kernel_learning[0]:
            level = 0
            b, c, H, W = inp_enc_level3.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_0, batch_list = window_partitionx(inp_enc_level3, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_0 = self.grids(inp_enc_level3, level=level)
                # print(x_.shape)

            x_fft = torch.fft.rfft2(x_0)
            x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_ = self.kernel_ins_max(x_)
            x_ = self.kernel_blks_max(x_)
            x_ = self.kernel_outs_max(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_1 = torch.fft.irfft2(otf * x_fft)
            # print(x_.shape)
            if self.geo_aug:
                x_s = x_1.chunk(4, dim=1)
                x_os = list()
                for j in range(4):
                    x_j = x_s[j]
                    if j % 2 == 1:
                        x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
                    if j % 2 == 0:
                        x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
                    x_os.append(x_j)
                x_1 = torch.cat(x_os, dim=1)
            x_1 = self.n0(x_1)

            # print(inp_img.shape, inp_dec_level1.shape)
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_1 = window_reversex(x_1, self.ker_size[level], H, W, batch_list)
            else:
                x_1 = self.grids_inverse(x_1, level=level)

            inp_enc_level3 = self.proj_feature_outs_max(torch.cat([inp_enc_level3, x_1], dim=1))
            outputs.append(self.up_decs_max(inp_enc_level3) + inp_img)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        if self.kernel_learning[1]:
            level = 1
            b, c, H, W = out_enc_level3.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_0, batch_list = window_partitionx(out_enc_level3, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_0 = self.grids(out_enc_level3, level=level)
                # print(x_.shape)

            x_fft = torch.fft.rfft2(x_0)
            x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_ = self.kernel_ins_mid(x_)
            x_ = self.kernel_blks_mid(x_)
            x_ = self.kernel_outs_mid(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_1 = torch.fft.irfft2(otf * x_fft)
            if self.geo_aug:
                x_s = x_1.chunk(4, dim=1)
                x_os = list()
                for j in range(4):
                    x_j = x_s[j]
                    if j % 2 == 1:
                        x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
                    if j % 2 == 0:
                        x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
                    x_os.append(x_j)
                x_1 = torch.cat(x_os, dim=1)
            x_1 = self.n1(x_1)

            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_1 = window_reversex(x_1, self.ker_size[level], H, W, batch_list)
            else:
                x_1 = self.grids_inverse(x_1, level=level)
            out_enc_level3 = self.proj_feature_outs_mid(torch.cat([out_enc_level3, x_1], dim=1))
            outputs.append(self.up_decs_mid(out_enc_level3) + inp_img)

        out_dec_level3 = self.decoder_level3(out_enc_level3)
        if self.kernel_learning[2]:
            level = 2
            b, c, H, W = out_dec_level3.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_0, batch_list = window_partitionx(out_dec_level3, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_0 = self.grids(out_dec_level3, level=level)
                # print(x_.shape)
            # if self.geo_aug:
            #     x_s = x_0.chunk(4, dim=1)
            #     x_os = list()
            #     for j in range(4):
            #         x_j = x_s[j]
            #         if j % 2 == 1:
            #             x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
            #         if j % 2 == 0:
            #             x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
            #         x_fft = torch.fft.rfft2(x_j)
            #         x_j = torch.cat([x_fft.real, x_fft.imag], dim=1)
            #         x_j = self.kernel_ins[j](x_j)
            #         x_j = self.kernel_blks[j](x_j)
            #         x_j = self.kernel_outs[j](x_j)
            #         x_real, x_imag = torch.chunk(x_j, 2, dim=1)
            #         otf = torch.complex(x_real, x_imag)
            #         # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            #         # print(x_fft.shape, otf.shape)
            #         x_j = torch.fft.irfft2(otf * x_fft)
            #         if j % 2 == 1:
            #             x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
            #         if j % 2 == 0:
            #             x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
            #         x_os.append(x_j)
            #     x_1 = torch.cat(x_os, dim=1)
            # else:
            x_fft = torch.fft.rfft2(x_0)
            x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_ = self.kernel_ins(x_)
            x_ = self.kernel_blks(x_)
            x_ = self.kernel_outs(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_1 = torch.fft.irfft2(otf * x_fft)
            if self.geo_aug:
                x_s = x_1.chunk(4, dim=1)
                x_os = list()
                for j in range(4):
                    x_j = x_s[j]
                    if j % 2 == 1:
                        x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
                    if j % 2 == 0:
                        x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
                    x_os.append(x_j)
                x_1 = torch.cat(x_os, dim=1)
            # print(x_.shape)
            x_1 = self.n2(x_1)

            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_1 = window_reversex(x_1, self.ker_size[level], H, W, batch_list)
            else:
                x_1 = self.grids_inverse(x_1, level=level)
            out_dec_level3 = self.proj_feature_outs(torch.cat([out_dec_level3, x_1], dim=1))
            outputs.append(self.up_decs(out_dec_level3) + inp_img)

        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = inp_dec_level2 + out_enc_level2



        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = inp_dec_level1 + out_enc_level1


        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img
        outputs.append(out_dec_level1)
        # print('xxxxxxxxxxxxxxxxxxx')
        return outputs

class prompt_attention(nn.Module):
    def __init__(self, prompt_ch, in_ch, out_ch):
        super(prompt_attention, self).__init__()

        self.conv_1 = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU()
                        )
        self.conv_kernel = nn.Sequential(
                        LayerNorm2d(prompt_ch),
                        nn.Conv2d(prompt_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU(),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.GELU()
                        )
        self.conv_2 = nn.Sequential(
                        nn.Conv2d(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.Sigmoid()
                        )

    def forward(self, input, prompt):
        x = self.conv_1(input)
        kernel = self.conv_kernel(prompt)
        att = torch.cat([x, kernel], dim=1)
        att = self.conv_2(att)
        x = x * att
        output = x + input

        return output

@ARCH_REGISTRY.register()
class uniDeblurM(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[6, 6, 12],
        num_blocks2=[6, 6, 12],
        baseblock=['localnaf', 'localnaf', 'localnaf'],
        ker_baseblock='EVSS',
        ffn_expansion_factor=3,
        bias=False,
        train_size=[256, 256],
        patch_size=[256, 256],
        ker_size=[[256, 256], [128, 128], [64, 64]],
        overlap_size=[[32, 32], [16, 16], [8, 8]],
        k_channels=[48, 96, 192],
        ker_start=1,
        window_size=8,
        global_fft=[False, False, False],
        patch_size_fft=[128, 128],
        kernel_learning=[False, False, False],
        dec_with_enc=False,
        geo_aug=False,
        flip_transpose_max=False
    ):
        super().__init__()
        self.kernel_learning = kernel_learning
        self.flip_transpose_max = flip_transpose_max
        self.geo_aug = geo_aug
        self.ker_start = ker_start
        self.patch_size = ker_size
        self.ker_size = ker_size
        self.out_channels = [dim, dim*2, dim*4]
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
        self.encoder = True
        self.dec_with_enc = dec_with_enc
        if self.kernel_learning[2]:
            # if self.geo_aug:
            #     self.kernel_ins = nn.ModuleList()
            #     self.kernel_blks = nn.ModuleList()
            #     self.kernel_outs = nn.ModuleList()
            #     for _ in range(4):
            #         self.kernel_ins.append(nn.Sequential(
            #             nn.Conv2d(dim * 1 * 2, dim * 2 * 2, 1, bias=bias)
            #         ))
            #         # self.kernel_blks = nn.Sequential(
            #         #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            #         # )
            #         kernel_blks = nn.Sequential()
            #         for i in range(4):
            #             block = get_basicblock(ker_baseblock, channels=dim * 2 * 2, num_heads=1, sub_idx=i,
            #                                    train_size=[train_size[0] // 4, train_size[1] // 4],
            #                                    patch_size=[patch_size[0] // 4, patch_size[1] // 4])
            #             kernel_blks.add_module(f"block{i}", block)
            #         self.kernel_blks.append(kernel_blks)
            #         self.kernel_outs.append(nn.Sequential(
            #             LayerNorm(dim * 2 * 2),
            #             nn.Conv2d(dim * 2 * 2, dim * 1 * 2, 1, bias=bias)
            #         ))
            # else:
            self.kernel_ins = nn.Sequential(
                nn.Conv2d(dim * 4 * 2, k_channels[2], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks = nn.Sequential()
            for i in range(4):
                block = get_basicblock(ker_baseblock, channels=k_channels[2], num_heads=1, sub_idx=i,
                                       train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
                self.kernel_blks.add_module(f"block{i}", block)

            self.kernel_outs = nn.Sequential(
                LayerNorm(k_channels[2]),
                nn.Conv2d(k_channels[2], dim * 4 * 2, 1, bias=bias)
            )

            # self.proj_feature_outs = nn.Sequential(
            #     nn.Conv2d(dim * 4 * 1, dim * 4, 1, bias=bias)
            # )
            self.proj_feature_outs = prompt_attention(dim * 4, dim * 4, dim * 4)
            if self.dec_with_enc:
                self.up_decs1 = nn.Sequential(
                    nn.Conv2d(dim * 4 * 2, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2)
                )
                self.up_decs2 = nn.Sequential(
                    nn.Conv2d(dim * 2 * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2)
                )
                self.up_decs3 = nn.Sequential(
                    nn.Conv2d(in_channels=dim * 2, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
            else:
                self.up_decs = nn.Sequential(
                    # nn.Conv2d(dim * 4, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    # nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(in_channels=dim*4, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
        if self.kernel_learning[1]:
            self.kernel_ins_mid = nn.Sequential(
                nn.Conv2d(dim * 2 * 2, k_channels[1], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks_mid = nn.Sequential()
            for i in range(4):
                block = get_basicblock(ker_baseblock, channels=k_channels[1], num_heads=1, sub_idx=i,
                                       train_size=[train_size[0] // 2, train_size[1] // 2],
                                       patch_size=[patch_size[0] // 2, patch_size[1] // 2])
                self.kernel_blks_mid.add_module(f"block{i}", block)

            self.kernel_outs_mid = nn.Sequential(
                LayerNorm(k_channels[1]),
                nn.Conv2d(k_channels[1], dim * 2 * 2, 1, bias=bias)
            )
            self.up_decs_mid = nn.Sequential(
                # nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                nn.Conv2d(in_channels=dim*2, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=bias)
            )
            # self.proj_feature_outs_mid = nn.Sequential(
            #     # LayerNorm(dim * 2),
            #     nn.Conv2d(dim * 2 * 1, dim * 2, 1, bias=bias)
            # )
            self.proj_feature_outs_mid = prompt_attention(dim * 2, dim * 2, dim * 2)
        if self.kernel_learning[0]:
            self.kernel_ins_max = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, k_channels[0], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks_max = nn.Sequential()
            for i in range(4):
                block = get_basicblock(ker_baseblock, channels=k_channels[0], num_heads=1, sub_idx=i,
                                       train_size=[train_size[0] // 2, train_size[1] // 2],
                                       patch_size=[patch_size[0] // 2, patch_size[1] // 2])
                self.kernel_blks_max.add_module(f"block{i}", block)

            self.kernel_outs_max = nn.Sequential(
                LayerNorm(k_channels[0]),
                nn.Conv2d(k_channels[0], dim * 1 * 2, 1, bias=bias)
            )
            self.up_decs_max = nn.Sequential(
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=bias)
            )
            # self.proj_feature_outs_max = nn.Sequential(
            #     nn.Conv2d(dim * 1 * 1, dim * 1, 1, bias=bias)
            # )
            self.proj_feature_outs_max = prompt_attention(dim * 1, dim * 1, dim * 1)
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.encoder_level1.add_module(f"block{i}", block)

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.encoder_level2.add_module(f"block{i}", block)

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.encoder_level3.add_module(f"block{i}", block)

        self.decoder_level3 = nn.Sequential()
        for i in range(num_blocks2[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.decoder_level3.add_module(f"block{i}", block)

        self.up3_2 = Upsample(int(dim * 2 ** 2))

        self.decoder_level2 = nn.Sequential()
        for i in range(num_blocks2[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.decoder_level2.add_module(f"block{i}", block)

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential()
        for i in range(num_blocks2[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.decoder_level1.add_module(f"block{i}", block)

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.loss_l1 = L1Loss()
        self.loss_freq = FourierLoss(0.01)
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
    def forward(self, inp_img, gt=None):
        outputs = []
        out_dict = {}
        outputs_small = []
        outputs_mid = []
        outputs_max = []
        inp_pry = kornia.geometry.transform.build_pyramid(inp_img, 3)
        if gt is not None:
            gt_pry = kornia.geometry.transform.build_pyramid(gt, 3)
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        if self.kernel_learning[2]:
            level = 2
            b, c, H, W = out_enc_level3.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_0, batch_list = window_partitionx(out_enc_level3, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_0 = self.grids(out_enc_level3, level=level)
                # print(x_.shape)
            # if self.geo_aug:
            #     x_s = x_0.chunk(4, dim=1)
            #     x_os = list()
            #     for j in range(4):
            #         x_j = x_s[j]
            #         if j % 2 == 1:
            #             x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
            #         if j % 2 == 0:
            #             x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
            #         x_fft = torch.fft.rfft2(x_j)
            #         x_j = torch.cat([x_fft.real, x_fft.imag], dim=1)
            #         x_j = self.kernel_ins[j](x_j)
            #         x_j = self.kernel_blks[j](x_j)
            #         x_j = self.kernel_outs[j](x_j)
            #         x_real, x_imag = torch.chunk(x_j, 2, dim=1)
            #         otf = torch.complex(x_real, x_imag)
            #         # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            #         # print(x_fft.shape, otf.shape)
            #         x_j = torch.fft.irfft2(otf * x_fft)
            #         if j % 2 == 1:
            #             x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
            #         if j % 2 == 0:
            #             x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
            #         x_os.append(x_j)
            #     x_1 = torch.cat(x_os, dim=1)
            # else:
            x_fft = torch.fft.rfft2(x_0)
            x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_ = self.kernel_ins(x_)
            x_ = self.kernel_blks(x_)
            x_ = self.kernel_outs(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_1 = torch.fft.irfft2(otf * x_fft)
            if self.geo_aug:
                x_s = x_1.chunk(4, dim=1)
                x_os = list()
                for j in range(4):
                    x_j = x_s[j]
                    if j % 2 == 1:
                        x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
                    if j % 2 == 0:
                        x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
                    x_os.append(x_j)
                x_1 = torch.cat(x_os, dim=1)
            # print(x_.shape)

             # (torch.cat([x_0, x_1], dim=1))
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_1 = window_reversex(x_1, self.ker_size[level], H, W, batch_list)
            else:
                x_1 = self.grids_inverse(x_1, level=level)
            # if self.dec_with_enc:
            #     x0 = self.up_decs1(torch.cat([out_enc_level3, x_1], dim=1))
            #     x0 = self.up_decs2(torch.cat([out_enc_level2, x0], dim=1))
            #     x0 = self.up_decs3(torch.cat([out_enc_level1, x0], dim=1))
            #     outputs.append(x0 + inp_img)
            # else:
            outputs_small.append(self.up_decs(x_1) + inp_pry[2])
            out_enc_level3 = self.proj_feature_outs(out_enc_level3, x_1)
        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = inp_dec_level2 + out_enc_level2

        if self.kernel_learning[1]:
            level = 1
            b, c, H, W = inp_dec_level2.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_0, batch_list = window_partitionx(inp_dec_level2, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_0 = self.grids(inp_dec_level2, level=level)
                # print(x_.shape)

            x_fft = torch.fft.rfft2(x_0)
            x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_ = self.kernel_ins_mid(x_)
            x_ = self.kernel_blks_mid(x_)
            x_ = self.kernel_outs_mid(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_1 = torch.fft.irfft2(otf * x_fft)
            if self.geo_aug:
                x_s = x_1.chunk(4, dim=1)
                x_os = list()
                for j in range(4):
                    x_j = x_s[j]
                    if j % 2 == 1:
                        x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
                    if j % 2 == 0:
                        x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
                    x_os.append(x_j)
                x_1 = torch.cat(x_os, dim=1)

             # (torch.cat([x_0, x_1], dim=1))
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_1 = window_reversex(x_1, self.ker_size[level], H, W, batch_list)
            else:
                x_1 = self.grids_inverse(x_1, level=level)
            outputs_mid.append(self.up_decs_mid(x_1) + inp_pry[1])
            inp_dec_level2 = self.proj_feature_outs_mid(inp_dec_level2, x_1)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = inp_dec_level1 + out_enc_level1

        if self.kernel_learning[0]:
            level = 0
            b, c, H, W = inp_dec_level1.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_0, batch_list = window_partitionx(inp_dec_level1, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_0 = self.grids(inp_dec_level1, level=level)
                # print(x_.shape)

            x_fft = torch.fft.rfft2(x_0)
            x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_ = self.kernel_ins_max(x_)
            x_ = self.kernel_blks_max(x_)
            x_ = self.kernel_outs_max(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_1 = torch.fft.irfft2(otf * x_fft)
            # print(x_.shape)
            if self.geo_aug:
                x_s = x_1.chunk(4, dim=1)
                x_os = list()
                for j in range(4):
                    x_j = x_s[j]
                    if j % 2 == 1:
                        x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
                    if j % 2 == 0:
                        x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
                    x_os.append(x_j)
                x_1 = torch.cat(x_os, dim=1)

            # (torch.cat([x_0, x_1], dim=1))
            # print(inp_img.shape, inp_dec_level1.shape)
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_1 = window_reversex(x_1, self.ker_size[level], H, W, batch_list)
            else:
                x_1 = self.grids_inverse(x_1, level=level)
            if self.flip_transpose_max:
                inp_dec_level1_ = torch.flip(x_1, dims=(-2, -1)).contiguous()
                inp_dec_level1_ = torch.transpose(inp_dec_level1_, dim0=-2, dim1=-1).contiguous()
            else:
                inp_dec_level1_ = x_1
            outputs_max.append(self.up_decs_max(inp_dec_level1_) + inp_img)
            inp_dec_level1 = self.proj_feature_outs_max(inp_dec_level1, x_1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img
        outputs.append(out_dec_level1)
        # print('xxxxxxxxxxxxxxxxxxx')

        out_dict['restored'] = outputs
        if self.training and gt is not None:
            loss_small = self.loss_l1(outputs_small, gt_pry[2]) + self.loss_freq(outputs_small, gt_pry[2])
            loss_mid = self.loss_l1(outputs_mid, gt_pry[1]) + self.loss_freq(outputs_mid, gt_pry[1])
            loss_max = self.loss_l1(outputs_max, gt) + self.loss_freq(outputs_max, gt)
            loss = loss_max * 0.5 + loss_mid * 0.3 + loss_small * 0.2
            out_dict['loss_model'] = loss
        return out_dict
@ARCH_REGISTRY.register()
class uniDeblurInf(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[6, 6, 12],
        num_blocks2=[6, 6, 12],
        baseblock=['localnaf', 'localnaf', 'localnaf'],
        ker_baseblock='EVSS',
        ffn_expansion_factor=3,
        bias=False,
        train_size=[256, 256],
        patch_size=[256, 256],
        ker_size=[[256, 256], [128, 128], [64, 64]],
        overlap_size=[[32, 32], [16, 16], [8, 8]],
        k_channels=[48, 96, 192],
        ker_start=1,
        window_size=8,
        global_fft=[False, False, False],
        patch_size_fft=[128, 128],
        kernel_learning=[False, False, False],
        dec_with_enc=False,
        geo_aug=False,
        flip_transpose_max=False
    ):
        super().__init__()
        self.kernel_learning = kernel_learning
        self.flip_transpose_max = flip_transpose_max
        self.geo_aug = geo_aug
        self.ker_start = ker_start
        self.patch_size = ker_size
        self.ker_size = ker_size
        self.out_channels = [dim, dim*2, dim*4]
        self.overlap_size = overlap_size
        # self.overlap_size = [0, 0]
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
        self.encoder = True
        self.dec_with_enc = dec_with_enc
        if self.kernel_learning[2]:
            if self.geo_aug:
                self.kernel_ins = nn.ModuleList()
                self.kernel_blks = nn.ModuleList()
                self.kernel_outs = nn.ModuleList()
                for _ in range(4):
                    self.kernel_ins.append(nn.Sequential(
                        nn.Conv2d(dim * 1 * 2, dim * 2 * 2, 1, bias=bias)
                    ))
                    # self.kernel_blks = nn.Sequential(
                    #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
                    # )
                    kernel_blks = nn.Sequential()
                    for i in range(4):
                        block = get_basicblock(ker_baseblock, channels=dim * 2 * 2, num_heads=1, sub_idx=i,
                                               train_size=[train_size[0] // 4, train_size[1] // 4],
                                               patch_size=[patch_size[0] // 4, patch_size[1] // 4])
                        kernel_blks.add_module(f"block{i}", block)
                    self.kernel_blks.append(kernel_blks)
                    self.kernel_outs.append(nn.Sequential(
                        LayerNorm(dim * 2 * 2),
                        nn.Conv2d(dim * 2 * 2, dim * 1 * 2, 1, bias=bias)
                    ))
            else:
                self.kernel_ins = nn.Sequential(
                    nn.Conv2d(dim * 4 * 2, k_channels[2], 1, bias=bias)
                )
                # self.kernel_blks = nn.Sequential(
                #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
                # )
                self.kernel_blks = nn.Sequential()
                for i in range(4):
                    block = get_basicblock(ker_baseblock, channels=k_channels[2], num_heads=1, sub_idx=i,
                                           train_size=[train_size[0]//4, train_size[1]//4],
                                           patch_size=[patch_size[0]//4, patch_size[1]//4])
                    self.kernel_blks.add_module(f"block{i}", block)

                self.kernel_outs = nn.Sequential(
                    LayerNorm(k_channels[2]),
                    nn.Conv2d(k_channels[2], dim * 4 * 2, 1, bias=bias)
                )

            self.proj_feature_outs = nn.Sequential(
                nn.Conv2d(dim * 4 * 2, dim * 4, 1, bias=bias)
            )
            if self.dec_with_enc:
                self.up_decs1 = nn.Sequential(
                    nn.Conv2d(dim * 4 * 2, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2)
                )
                self.up_decs2 = nn.Sequential(
                    nn.Conv2d(dim * 2 * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2)
                )
                self.up_decs3 = nn.Sequential(
                    nn.Conv2d(in_channels=dim * 2, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
            else:
                self.up_decs = nn.Sequential(
                    nn.Conv2d(dim * 4, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
        if self.kernel_learning[1]:
            self.kernel_ins_mid = nn.Sequential(
                nn.Conv2d(dim * 2 * 2, k_channels[1], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks_mid = nn.Sequential()
            for i in range(4):
                block = get_basicblock(ker_baseblock, channels=k_channels[1], num_heads=1, sub_idx=i,
                                       train_size=[train_size[0] // 2, train_size[1] // 2],
                                       patch_size=[patch_size[0] // 2, patch_size[1] // 2])
                self.kernel_blks_mid.add_module(f"block{i}", block)

            self.kernel_outs_mid = nn.Sequential(
                LayerNorm(k_channels[1]),
                nn.Conv2d(k_channels[1], dim * 2 * 2, 1, bias=bias)
            )
            self.up_decs_mid = nn.Sequential(
                nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=bias)
            )
            self.proj_feature_outs_mid = nn.Sequential(
                nn.Conv2d(dim * 2 * 2, dim * 2, 1, bias=bias)
            )
        if self.kernel_learning[0]:
            self.kernel_ins_max = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, k_channels[0], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks_max = nn.Sequential()
            for i in range(4):
                block = get_basicblock(ker_baseblock, channels=k_channels[0], num_heads=1, sub_idx=i,
                                       train_size=[train_size[0] // 2, train_size[1] // 2],
                                       patch_size=[patch_size[0] // 2, patch_size[1] // 2])
                self.kernel_blks_max.add_module(f"block{i}", block)

            self.kernel_outs_max = nn.Sequential(
                LayerNorm(k_channels[0]),
                nn.Conv2d(k_channels[0], dim * 1 * 2, 1, bias=bias)
            )
            self.up_decs_max = nn.Sequential(
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=bias)
            )
            self.proj_feature_outs_max = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, dim * 1, 1, bias=bias)
            )
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.encoder_level1.add_module(f"block{i}", block)

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.encoder_level2.add_module(f"block{i}", block)

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.encoder_level3.add_module(f"block{i}", block)

        self.decoder_level3 = nn.Sequential()
        for i in range(num_blocks2[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.decoder_level3.add_module(f"block{i}", block)

        self.up3_2 = Upsample(int(dim * 2 ** 2))

        self.decoder_level2 = nn.Sequential()
        for i in range(num_blocks2[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.decoder_level2.add_module(f"block{i}", block)

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential()
        for i in range(num_blocks2[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.decoder_level1.add_module(f"block{i}", block)

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
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
    def forward(self, inp_img, gt=None):
        outputs = []

        level = 0

        b, c, H, W = inp_img.shape
        if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
            inp_enc_level1, batch_list = window_partitionx(inp_img, self.ker_size[level])
        else:
            # print('11111111111111111')
            # print('level: ', level)
            inp_enc_level1 = self.grids(inp_img, level=level)
        inp_enc_level1 = self.patch_embed(inp_enc_level1)

        # level = 0
        #
        # b, c, H, W = inp_enc_level1.shape
        # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
        #     inp_enc_level1, batch_list = window_partitionx(inp_enc_level1, self.ker_size[level])
        # else:
        #     # print('11111111111111111')
        #     # print('level: ', level)
        #     inp_enc_level1 = self.grids(inp_enc_level1, level=level)
            # print(x_.shape)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        # level = 0
        #
        # b, c, H, W = inp_enc_level1.shape
        # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
        #     out_enc_level1, batch_list = window_partitionx(out_enc_level1, self.ker_size[level])
        # else:
        #     # print('11111111111111111')
        #     # print('level: ', level)
        #     out_enc_level1 = self.grids(out_enc_level1, level=level)
        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        level = 2

        # b, c, H, W = out_enc_level3.shape
        # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
        #     out_enc_level3, batch_list = window_partitionx(out_enc_level3, self.ker_size[level])
        # else:
        #     # print('11111111111111111')
        #     # print('level: ', level)
        #     out_enc_level3 = self.grids(out_enc_level3, level=level)
        if self.kernel_learning[2]:


            x_fft = torch.fft.rfft2(out_enc_level3)
            x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_ = self.kernel_ins(x_)
            x_ = self.kernel_blks(x_)
            x_ = self.kernel_outs(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_ = torch.fft.irfft2(otf * x_fft)
            # print(x_.shape)
            # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
            #     x_ = window_reversex(x_, self.ker_size[level], H, W, batch_list)
            # else:
            #     x_ = self.grids_inverse(x_, level=level)
            out_enc_level3 = self.proj_feature_outs(torch.cat([out_enc_level3, x_], dim=1))
            # if self.dec_with_enc:
            #     x0 = self.up_decs1(torch.cat([out_enc_level3, x_], dim=1))
            #     x0 = self.up_decs2(torch.cat([out_enc_level2, x0], dim=1))
            #     x0 = self.up_decs3(torch.cat([out_enc_level1, x0], dim=1))
            #     outputs.append(x0 + inp_img)
            # else:
            #     outputs.append(self.up_decs(out_enc_level3) + inp_img)

        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)

        level = 1

        # b, c, H, W = inp_enc_level1.shape
        # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
        #     out_enc_level2, batch_list = window_partitionx(out_enc_level2, self.ker_size[level])
        # else:
        #     # print('11111111111111111')
        #     # print('level: ', level)
        #     out_enc_level2 = self.grids(out_enc_level2, level=level)
        inp_dec_level2 = inp_dec_level2 + out_enc_level2

        if self.kernel_learning[1]:
            level = 1
            # b, c, H, W = inp_dec_level2.shape
            # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
            #     x_, batch_list = window_partitionx(inp_dec_level2, self.ker_size[level])
            # else:
            #     # print('11111111111111111')
            #     # print('level: ', level)
            #     x_ = self.grids(inp_dec_level2, level=level)
                # print(x_.shape)

            x_fft = torch.fft.rfft2(inp_dec_level2)
            x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_ = self.kernel_ins_mid(x_)
            x_ = self.kernel_blks_mid(x_)
            x_ = self.kernel_outs_mid(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_ = torch.fft.irfft2(otf * x_fft)

            # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
            #     x_ = window_reversex(x_, self.ker_size[level], H, W, batch_list)
            # else:
            #     x_ = self.grids_inverse(x_, level=level)
            inp_dec_level2 = self.proj_feature_outs_mid(torch.cat([inp_dec_level2, x_], dim=1))

            # outputs.append(self.up_decs_mid(inp_dec_level2) + inp_img)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        level = 0

        # b, c, H, W = inp_enc_level1.shape
        # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
        #     out_enc_level1, batch_list = window_partitionx(out_enc_level1, self.ker_size[level])
        # else:
        #     # print('11111111111111111')
        #     # print('level: ', level)
        #     out_enc_level1 = self.grids(out_enc_level1, level=level)
        inp_dec_level1 = inp_dec_level1 + out_enc_level1

        if self.kernel_learning[0]:
            # level = 0
            # b, c, H, W = inp_dec_level1.shape
            # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
            #     x_, batch_list = window_partitionx(inp_dec_level1, self.ker_size[level])
            # else:
            #     # print('11111111111111111')
            #     # print('level: ', level)
            #     x_ = self.grids(inp_dec_level1, level=level)
                # print(x_.shape)

            x_fft = torch.fft.rfft2(inp_dec_level1)
            x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_ = self.kernel_ins_max(x_)
            x_ = self.kernel_blks_max(x_)
            x_ = self.kernel_outs_max(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_ = torch.fft.irfft2(otf * x_fft)
            # print(x_.shape)
            # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
            #     x_ = window_reversex(x_, self.ker_size[level], H, W, batch_list)
            # else:
            #     x_ = self.grids_inverse(x_, level=level)

            inp_dec_level1 = self.proj_feature_outs_max(torch.cat([inp_dec_level1, x_], dim=1))
            # print(inp_img.shape, inp_dec_level1.shape)
            # if self.flip_transpose_max:
            #     inp_dec_level1_ = torch.flip(inp_dec_level1, dims=(-2, -1)).contiguous()
            #     inp_dec_level1_ = torch.transpose(inp_dec_level1_, dim0=-2, dim1=-1).contiguous()
            # else:
            #     inp_dec_level1_ = inp_dec_level1
            # outputs.append(self.up_decs_max(inp_dec_level1_) + inp_img)
        level = 0
        if self.flip_transpose_max:
            inp_dec_level1 = torch.flip(inp_dec_level1, dims=(-2, -1)).contiguous()
            inp_dec_level1 = torch.transpose(inp_dec_level1, dim0=-2, dim1=-1).contiguous()
        else:
            inp_dec_level1 = inp_dec_level1
        if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
            inp_dec_level1 = window_reversex(inp_dec_level1, self.ker_size[level], H, W, batch_list, mask=True)
        else:
            inp_dec_level1 = self.grids_inverse(inp_dec_level1, level=level)
        if self.flip_transpose_max:
            inp_dec_level1 = torch.flip(inp_dec_level1, dims=(-2, -1)).contiguous()
            inp_dec_level1 = torch.transpose(inp_dec_level1, dim0=-2, dim1=-1).contiguous()
        else:
            inp_dec_level1 = inp_dec_level1
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        # level = 0
        # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
        #     out_dec_level1 = window_reversex(out_dec_level1, self.ker_size[level], H, W, batch_list, mask=True)
        # else:
        #     out_dec_level1 = self.grids_inverse(out_dec_level1, level=level)

        out_img = self.output(out_dec_level1) + inp_img
        # outputs.append(out_dec_level1)
        # print('xxxxxxxxxxxxxxxxxxx')
        out_dict = {}
        out_dict['feature'] = out_dec_level1
        out_dict['restored'] = out_img
        return out_dict


@ARCH_REGISTRY.register()
class uniDeblurG(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[6, 6, 12],
        num_blocks2=[6, 6, 12],
        baseblock=['localnaf', 'localnaf', 'localnaf'],
        ker_baseblock='EVSS',
        ffn_expansion_factor=3,
        bias=False,
        train_size=[256, 256],
        patch_size=[256, 256],
        ker_size=[[256, 256], [128, 128], [64, 64]],
        overlap_size=[[32, 32], [16, 16], [8, 8]],
        k_channels=[48, 96, 192],
        ker_start=1,
        window_size=8,
        global_fft=[False, False, False],
        patch_size_fft=[128, 128],
        kernel_learning=[False, False, False],
        dec_with_enc=False,
        geo_aug=False,
        flip_transpose_max=False
    ):
        super().__init__()
        self.kernel_learning = kernel_learning
        self.flip_transpose_max = flip_transpose_max
        self.geo_aug = geo_aug
        self.ker_start = ker_start
        self.patch_size = ker_size
        self.ker_size = ker_size
        self.out_channels = [dim, dim*2, dim*4]
        self.overlap_size = overlap_size
        # self.overlap_size = [0, 0]
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
        self.encoder = True
        self.dec_with_enc = dec_with_enc
        if self.kernel_learning[2]:
            if self.geo_aug:
                self.kernel_ins = nn.ModuleList()
                self.kernel_blks = nn.ModuleList()
                self.kernel_outs = nn.ModuleList()
                for _ in range(4):
                    self.kernel_ins.append(nn.Sequential(
                        nn.Conv2d(dim * 1 * 2, dim * 2 * 2, 1, bias=bias)
                    ))
                    # self.kernel_blks = nn.Sequential(
                    #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
                    # )
                    kernel_blks = nn.Sequential()
                    for i in range(4):
                        block = get_basicblock(ker_baseblock, channels=dim * 2 * 2, num_heads=1, sub_idx=i,
                                               train_size=[train_size[0] // 4, train_size[1] // 4],
                                               patch_size=[patch_size[0] // 4, patch_size[1] // 4])
                        kernel_blks.add_module(f"block{i}", block)
                    self.kernel_blks.append(kernel_blks)
                    self.kernel_outs.append(nn.Sequential(
                        LayerNorm(dim * 2 * 2),
                        nn.Conv2d(dim * 2 * 2, dim * 1 * 2, 1, bias=bias)
                    ))
            else:
                self.kernel_ins = nn.Sequential(
                    nn.Conv2d(dim * 4 * 2, k_channels[2], 1, bias=bias)
                )
                # self.kernel_blks = nn.Sequential(
                #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
                # )
                self.kernel_blks = nn.Sequential()
                for i in range(4):
                    block = get_basicblock(ker_baseblock, channels=k_channels[2], num_heads=1, sub_idx=i,
                                           train_size=[train_size[0]//4, train_size[1]//4],
                                           patch_size=[patch_size[0]//4, patch_size[1]//4])
                    self.kernel_blks.add_module(f"block{i}", block)

                self.kernel_outs = nn.Sequential(
                    LayerNorm(k_channels[2]),
                    nn.Conv2d(k_channels[2], dim * 4 * 2, 1, bias=bias)
                )

            self.proj_feature_outs = nn.Sequential(
                nn.Conv2d(dim * 4 * 2, dim * 4, 1, bias=bias)
            )
            if self.dec_with_enc:
                self.up_decs1 = nn.Sequential(
                    nn.Conv2d(dim * 4 * 2, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2)
                )
                self.up_decs2 = nn.Sequential(
                    nn.Conv2d(dim * 2 * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2)
                )
                self.up_decs3 = nn.Sequential(
                    nn.Conv2d(in_channels=dim * 2, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
            else:
                self.up_decs = nn.Sequential(
                    nn.Conv2d(dim * 4, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
        if self.kernel_learning[1]:
            self.kernel_ins_mid = nn.Sequential(
                nn.Conv2d(dim * 2 * 2, k_channels[1], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks_mid = nn.Sequential()
            for i in range(4):
                block = get_basicblock(ker_baseblock, channels=k_channels[1], num_heads=1, sub_idx=i,
                                       train_size=[train_size[0] // 2, train_size[1] // 2],
                                       patch_size=[patch_size[0] // 2, patch_size[1] // 2])
                self.kernel_blks_mid.add_module(f"block{i}", block)

            self.kernel_outs_mid = nn.Sequential(
                LayerNorm(k_channels[1]),
                nn.Conv2d(k_channels[1], dim * 2 * 2, 1, bias=bias)
            )
            self.up_decs_mid = nn.Sequential(
                nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=bias)
            )
            self.proj_feature_outs_mid = nn.Sequential(
                nn.Conv2d(dim * 2 * 2, dim * 2, 1, bias=bias)
            )
        if self.kernel_learning[0]:
            self.kernel_ins_max = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, k_channels[0], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks_max = nn.Sequential()
            for i in range(4):
                block = get_basicblock(ker_baseblock, channels=k_channels[0], num_heads=1, sub_idx=i,
                                       train_size=[train_size[0] // 2, train_size[1] // 2],
                                       patch_size=[patch_size[0] // 2, patch_size[1] // 2])
                self.kernel_blks_max.add_module(f"block{i}", block)

            self.kernel_outs_max = nn.Sequential(
                LayerNorm(k_channels[0]),
                nn.Conv2d(k_channels[0], dim * 1 * 2, 1, bias=bias)
            )
            self.up_decs_max = nn.Sequential(
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=bias)
            )
            self.proj_feature_outs_max = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, dim * 1, 1, bias=bias)
            )
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.encoder_level1.add_module(f"block{i}", block)

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.encoder_level2.add_module(f"block{i}", block)

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.encoder_level3.add_module(f"block{i}", block)

        self.decoder_level3 = nn.Sequential()
        for i in range(num_blocks2[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.decoder_level3.add_module(f"block{i}", block)

        self.up3_2 = Upsample(int(dim * 2 ** 2))

        self.decoder_level2 = nn.Sequential()
        for i in range(num_blocks2[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.decoder_level2.add_module(f"block{i}", block)

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential()
        for i in range(num_blocks2[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.decoder_level1.add_module(f"block{i}", block)

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
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
    def forward(self, inp_img, gt=None):
        outputs = []

        level = 0

        b, c, H, W = inp_img.shape
        # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
        #     inp_enc_level1, batch_list = window_partitionx(inp_img, self.ker_size[level])
        # else:
        #     # print('11111111111111111')
        #     # print('level: ', level)
        #     inp_enc_level1 = self.grids(inp_img, level=level)
        inp_enc_level1 = self.patch_embed(inp_img)

        # level = 0
        #
        # b, c, H, W = inp_enc_level1.shape
        # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
        #     inp_enc_level1, batch_list = window_partitionx(inp_enc_level1, self.ker_size[level])
        # else:
        #     # print('11111111111111111')
        #     # print('level: ', level)
        #     inp_enc_level1 = self.grids(inp_enc_level1, level=level)
            # print(x_.shape)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        # level = 0
        #
        # b, c, H, W = inp_enc_level1.shape
        # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
        #     out_enc_level1, batch_list = window_partitionx(out_enc_level1, self.ker_size[level])
        # else:
        #     # print('11111111111111111')
        #     # print('level: ', level)
        #     out_enc_level1 = self.grids(out_enc_level1, level=level)
        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        level = 2

        # b, c, H, W = out_enc_level3.shape
        # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
        #     out_enc_level3, batch_list = window_partitionx(out_enc_level3, self.ker_size[level])
        # else:
        #     # print('11111111111111111')
        #     # print('level: ', level)
        #     out_enc_level3 = self.grids(out_enc_level3, level=level)
        if self.kernel_learning[2]:


            x_fft = torch.fft.rfft2(out_enc_level3)
            x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_ = self.kernel_ins(x_)
            x_ = self.kernel_blks(x_)
            x_ = self.kernel_outs(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_ = torch.fft.irfft2(otf * x_fft)
            # print(x_.shape)
            # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
            #     x_ = window_reversex(x_, self.ker_size[level], H, W, batch_list)
            # else:
            #     x_ = self.grids_inverse(x_, level=level)
            out_enc_level3 = self.proj_feature_outs(torch.cat([out_enc_level3, x_], dim=1))
            # if self.dec_with_enc:
            #     x0 = self.up_decs1(torch.cat([out_enc_level3, x_], dim=1))
            #     x0 = self.up_decs2(torch.cat([out_enc_level2, x0], dim=1))
            #     x0 = self.up_decs3(torch.cat([out_enc_level1, x0], dim=1))
            #     outputs.append(x0 + inp_img)
            # else:
            #     outputs.append(self.up_decs(out_enc_level3) + inp_img)

        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)

        level = 1

        # b, c, H, W = inp_enc_level1.shape
        # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
        #     out_enc_level2, batch_list = window_partitionx(out_enc_level2, self.ker_size[level])
        # else:
        #     # print('11111111111111111')
        #     # print('level: ', level)
        #     out_enc_level2 = self.grids(out_enc_level2, level=level)
        inp_dec_level2 = inp_dec_level2 + out_enc_level2

        if self.kernel_learning[1]:
            level = 1
            # b, c, H, W = inp_dec_level2.shape
            # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
            #     x_, batch_list = window_partitionx(inp_dec_level2, self.ker_size[level])
            # else:
            #     # print('11111111111111111')
            #     # print('level: ', level)
            #     x_ = self.grids(inp_dec_level2, level=level)
                # print(x_.shape)

            x_fft = torch.fft.rfft2(inp_dec_level2)
            x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_ = self.kernel_ins_mid(x_)
            x_ = self.kernel_blks_mid(x_)
            x_ = self.kernel_outs_mid(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_ = torch.fft.irfft2(otf * x_fft)

            # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
            #     x_ = window_reversex(x_, self.ker_size[level], H, W, batch_list)
            # else:
            #     x_ = self.grids_inverse(x_, level=level)
            inp_dec_level2 = self.proj_feature_outs_mid(torch.cat([inp_dec_level2, x_], dim=1))

            # outputs.append(self.up_decs_mid(inp_dec_level2) + inp_img)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        level = 0

        # b, c, H, W = inp_enc_level1.shape
        # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
        #     out_enc_level1, batch_list = window_partitionx(out_enc_level1, self.ker_size[level])
        # else:
        #     # print('11111111111111111')
        #     # print('level: ', level)
        #     out_enc_level1 = self.grids(out_enc_level1, level=level)
        inp_dec_level1 = inp_dec_level1 + out_enc_level1

        if self.kernel_learning[0]:
            # level = 0
            # b, c, H, W = inp_dec_level1.shape
            # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
            #     x_, batch_list = window_partitionx(inp_dec_level1, self.ker_size[level])
            # else:
            #     # print('11111111111111111')
            #     # print('level: ', level)
            #     x_ = self.grids(inp_dec_level1, level=level)
                # print(x_.shape)

            x_fft = torch.fft.rfft2(inp_dec_level1)
            x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_ = self.kernel_ins_max(x_)
            x_ = self.kernel_blks_max(x_)
            x_ = self.kernel_outs_max(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_ = torch.fft.irfft2(otf * x_fft)
            # print(x_.shape)
            # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
            #     x_ = window_reversex(x_, self.ker_size[level], H, W, batch_list)
            # else:
            #     x_ = self.grids_inverse(x_, level=level)

            inp_dec_level1 = self.proj_feature_outs_max(torch.cat([inp_dec_level1, x_], dim=1))
            # print(inp_img.shape, inp_dec_level1.shape)
            # if self.flip_transpose_max:
            #     inp_dec_level1_ = torch.flip(inp_dec_level1, dims=(-2, -1)).contiguous()
            #     inp_dec_level1_ = torch.transpose(inp_dec_level1_, dim0=-2, dim1=-1).contiguous()
            # else:
            #     inp_dec_level1_ = inp_dec_level1
            # outputs.append(self.up_decs_max(inp_dec_level1_) + inp_img)
        level = 0
        if self.flip_transpose_max:
            inp_dec_level1 = torch.flip(inp_dec_level1, dims=(-2, -1)).contiguous()
            inp_dec_level1 = torch.transpose(inp_dec_level1, dim0=-2, dim1=-1).contiguous()
        else:
            inp_dec_level1 = inp_dec_level1
        # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
        #     inp_dec_level1 = window_reversex(inp_dec_level1, self.ker_size[level], H, W, batch_list, mask=True)
        # else:
        #     inp_dec_level1 = self.grids_inverse(inp_dec_level1, level=level)
        if self.flip_transpose_max:
            inp_dec_level1 = torch.flip(inp_dec_level1, dims=(-2, -1)).contiguous()
            inp_dec_level1 = torch.transpose(inp_dec_level1, dim0=-2, dim1=-1).contiguous()
        else:
            inp_dec_level1 = inp_dec_level1
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        # level = 0
        # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
        #     out_dec_level1 = window_reversex(out_dec_level1, self.ker_size[level], H, W, batch_list, mask=True)
        # else:
        #     out_dec_level1 = self.grids_inverse(out_dec_level1, level=level)

        out_img = self.output(out_dec_level1) + inp_img
        # outputs.append(out_dec_level1)
        # print('xxxxxxxxxxxxxxxxxxx')
        out_dict = {}
        out_dict['feature'] = [out_enc_level1, out_enc_level2, out_enc_level3, out_dec_level3, out_dec_level2, out_dec_level1]
        out_dict['restored'] = out_img

        return out_dict
##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv3 = nn.Conv2d(3, n_feat, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        x2 = torch.sigmoid(self.conv3(x_img))
        x1 = x1*x2
        x1 = x1+x
        return x1
@ARCH_REGISTRY.register()
class uniDeblurS2(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[6, 6, 12],
        num_blocks2=[6, 6, 12],
        baseblock=['naf', 'naf', 'naf'],
        ker_baseblock='EVSS',
        ffn_expansion_factor=3,
        bias=False,
        train_size=[256, 256],
        patch_size=[256, 256],
        ker_size=[[256, 256], [128, 128], [64, 64]],
        overlap_size=[[32, 32], [16, 16], [8, 8]],
        k_channels=[48, 96, 192],
        ker_start=1,
        window_size=8,
        global_fft=[False, False, False],
        patch_size_fft=[128, 128],
        kernel_learning=[False, False, False],
        dec_with_enc=False,
        geo_aug=False,
        flip_transpose_max=False,
        prior_yml='',
        model_weight=''
    ):
        super().__init__()
        self.kernel_learning = kernel_learning
        self.flip_transpose_max = flip_transpose_max
        self.geo_aug = geo_aug
        self.ker_start = ker_start
        self.patch_size = ker_size
        self.ker_size = ker_size
        self.out_channels = [dim, dim*2, dim*4]
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
        self.encoder = True
        self.dec_with_enc = dec_with_enc
        self.n0 = nn.Tanh()
        self.n1 = nn.Tanh()
        self.n2 = nn.Tanh()
        # self.n0 = nn.Identity()
        # self.n1 = nn.Identity()
        # self.n2 = nn.Identity()
        import yaml

        try:
            from yaml import CLoader as Loader
        except ImportError:
            from yaml import Loader

        yaml_file = prior_yml  # '/data/mxt_code/DeepMXT/Motion_Deblurring/Options/Abalation-DeepRFTX-200k-8gpu.yml'
        x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

        s = x['network_g'].pop('type')
        ##########################
        # print(x['network_g'])

        self.model_global = uniDeblurG(**x['network_g'])
        self.csff_enc1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.csff_enc2 = nn.Conv2d(dim*2, dim*2, kernel_size=1, bias=bias)
        self.csff_enc3 = nn.Conv2d(dim*4, dim*4, kernel_size=1, bias=bias)

        self.csff_dec1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.csff_dec2 = nn.Conv2d(dim*2, dim*2, kernel_size=1, bias=bias)
        self.csff_dec3 = nn.Conv2d(dim*4, dim*4, kernel_size=1, bias=bias)
        checkpoint = torch.load(model_weight)
        self.model_global.load_state_dict(checkpoint['params_ema'], strict=True)
        self.model_global.eval()
        for p in self.model_global.parameters(): p.requires_grad = False
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.sam = SAM(dim, 3, bias)
        self.concat_conv = nn.Conv2d(dim*2, dim, kernel_size=1, stride=1, padding=0, bias=bias)

        if self.kernel_learning[2]:
            # if self.geo_aug:
            #     self.kernel_ins = nn.ModuleList()
            #     self.kernel_blks = nn.ModuleList()
            #     self.kernel_outs = nn.ModuleList()
            #     for _ in range(4):
            #         self.kernel_ins.append(nn.Sequential(
            #             nn.Conv2d(dim * 1 * 2, dim * 2 * 2, 1, bias=bias)
            #         ))
            #         # self.kernel_blks = nn.Sequential(
            #         #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            #         # )
            #         kernel_blks = nn.Sequential()
            #         for i in range(4):
            #             block = get_basicblock(ker_baseblock, channels=dim * 2 * 2, num_heads=1, sub_idx=i,
            #                                    train_size=[train_size[0] // 4, train_size[1] // 4],
            #                                    patch_size=[patch_size[0] // 4, patch_size[1] // 4])
            #             kernel_blks.add_module(f"block{i}", block)
            #         self.kernel_blks.append(kernel_blks)
            #         self.kernel_outs.append(nn.Sequential(
            #             LayerNorm(dim * 2 * 2),
            #             nn.Conv2d(dim * 2 * 2, dim * 1 * 2, 1, bias=bias)
            #         ))
            # else:
            self.kernel_ins = nn.Sequential(
                nn.Conv2d(dim * 4 * 2, k_channels[2], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks = nn.Sequential()
            for i in range(4):
                block = get_basicblock(ker_baseblock, channels=k_channels[2], num_heads=1, sub_idx=i,
                                       train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
                self.kernel_blks.add_module(f"block{i}", block)

            self.kernel_outs = nn.Sequential(
                LayerNorm(k_channels[2]),
                nn.Conv2d(k_channels[2], dim * 4 * 2, 1, bias=bias)
            )

            self.proj_feature_outs = nn.Sequential(
                nn.Conv2d(dim * 4 * 2, dim * 4, 1, bias=bias)
            )
            if self.dec_with_enc:
                self.up_decs1 = nn.Sequential(
                    nn.Conv2d(dim * 4 * 2, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2)
                )
                self.up_decs2 = nn.Sequential(
                    nn.Conv2d(dim * 2 * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2)
                )
                self.up_decs3 = nn.Sequential(
                    nn.Conv2d(in_channels=dim * 2, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
            else:
                self.up_decs = nn.Sequential(
                    nn.Conv2d(dim * 4, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
        if self.kernel_learning[1]:
            self.kernel_ins_mid = nn.Sequential(
                nn.Conv2d(dim * 2 * 2, k_channels[1], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks_mid = nn.Sequential()
            for i in range(4):
                block = get_basicblock(ker_baseblock, channels=k_channels[1], num_heads=1, sub_idx=i,
                                       train_size=[train_size[0] // 2, train_size[1] // 2],
                                       patch_size=[patch_size[0] // 2, patch_size[1] // 2])
                self.kernel_blks_mid.add_module(f"block{i}", block)

            self.kernel_outs_mid = nn.Sequential(
                LayerNorm(k_channels[1]),
                nn.Conv2d(k_channels[1], dim * 2 * 2, 1, bias=bias)
            )
            self.up_decs_mid = nn.Sequential(
                nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=bias)
            )
            self.proj_feature_outs_mid = nn.Sequential(
                nn.Conv2d(dim * 2 * 2, dim * 2, 1, bias=bias)
            )
        if self.kernel_learning[0]:
            self.kernel_ins_max = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, k_channels[0], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks_max = nn.Sequential()
            for i in range(4):
                block = get_basicblock(ker_baseblock, channels=k_channels[0], num_heads=1, sub_idx=i,
                                       train_size=[train_size[0] // 2, train_size[1] // 2],
                                       patch_size=[patch_size[0] // 2, patch_size[1] // 2])
                self.kernel_blks_max.add_module(f"block{i}", block)

            self.kernel_outs_max = nn.Sequential(
                LayerNorm(k_channels[0]),
                nn.Conv2d(k_channels[0], dim * 1 * 2, 1, bias=bias)
            )
            self.up_decs_max = nn.Sequential(
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=bias)
            )
            self.proj_feature_outs_max = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, dim * 1, 1, bias=bias)
            )
        self.encoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.encoder_level1.add_module(f"block{i}", block)

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.encoder_level2.add_module(f"block{i}", block)

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.encoder_level3.add_module(f"block{i}", block)

        self.decoder_level3 = nn.Sequential()
        for i in range(num_blocks2[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.decoder_level3.add_module(f"block{i}", block)

        self.up3_2 = Upsample(int(dim * 2 ** 2))

        self.decoder_level2 = nn.Sequential()
        for i in range(num_blocks2[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.decoder_level2.add_module(f"block{i}", block)

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential()
        for i in range(num_blocks2[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.decoder_level1.add_module(f"block{i}", block)

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
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
    def forward(self, inp_img, gt=None):
        outputs = []
        with torch.no_grad():
            out_dict = self.model_global(inp_img)
        enc1, enc2, enc3, dec3, dec2, dec1 = out_dict['feature']


        x = self.sam(dec1, out_dict['restored'])

        inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_level1 = self.concat_conv(torch.cat([inp_enc_level1, x], dim=1))
        out_enc_level1 = self.encoder_level1(inp_enc_level1) + self.csff_enc1(enc1) + self.csff_dec1(dec1)

        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2) + self.csff_enc2(enc2) + self.csff_dec2(dec2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3) + self.csff_enc3(enc3) + self.csff_dec3(dec3)

        if self.kernel_learning[2]:
            level = 2
            b, c, H, W = out_enc_level3.shape
            if (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_0, batch_list = window_partitionx(out_enc_level3, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_0 = self.grids(out_enc_level3, level=level)
                # print(x_.shape)
            # if self.geo_aug:
            #     x_s = x_0.chunk(4, dim=1)
            #     x_os = list()
            #     for j in range(4):
            #         x_j = x_s[j]
            #         if j % 2 == 1:
            #             x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
            #         if j % 2 == 0:
            #             x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
            #         x_fft = torch.fft.rfft2(x_j)
            #         x_j = torch.cat([x_fft.real, x_fft.imag], dim=1)
            #         x_j = self.kernel_ins[j](x_j)
            #         x_j = self.kernel_blks[j](x_j)
            #         x_j = self.kernel_outs[j](x_j)
            #         x_real, x_imag = torch.chunk(x_j, 2, dim=1)
            #         otf = torch.complex(x_real, x_imag)
            #         # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            #         # print(x_fft.shape, otf.shape)
            #         x_j = torch.fft.irfft2(otf * x_fft)
            #         if j % 2 == 1:
            #             x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
            #         if j % 2 == 0:
            #             x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
            #         x_os.append(x_j)
            #     x_1 = torch.cat(x_os, dim=1)
            # else:
            x_fft = torch.fft.rfft2(x_0)
            x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_ = self.kernel_ins(x_)
            x_ = self.kernel_blks(x_)
            x_ = self.kernel_outs(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_1 = torch.fft.irfft2(otf * x_fft)
            if self.geo_aug:
                x_s = x_1.chunk(4, dim=1)
                x_os = list()
                for j in range(4):
                    x_j = x_s[j]
                    if j % 2 == 1:
                        x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
                    if j % 2 == 0:
                        x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
                    x_os.append(x_j)
                x_1 = torch.cat(x_os, dim=1)
            # print(x_.shape)
            x_1 = self.n2(x_1)
            out_enc_level3 = self.proj_feature_outs(torch.cat([x_0, x_1], dim=1))
            if (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                out_enc_level3 = window_reversex(out_enc_level3, self.ker_size[level], H, W, batch_list)
            else:
                out_enc_level3 = self.grids_inverse(out_enc_level3, level=level)
            if self.dec_with_enc:
                x0 = self.up_decs1(torch.cat([out_enc_level3, x_1], dim=1))
                x0 = self.up_decs2(torch.cat([out_enc_level2, x0], dim=1))
                x0 = self.up_decs3(torch.cat([out_enc_level1, x0], dim=1))
                outputs.append(x0 + inp_img)
            else:
                outputs.append(self.up_decs(out_enc_level3) + inp_img)

        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = inp_dec_level2 + out_enc_level2

        if self.kernel_learning[1]:
            level = 1
            b, c, H, W = inp_dec_level2.shape
            if (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_0, batch_list = window_partitionx(inp_dec_level2, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_0 = self.grids(inp_dec_level2, level=level)
                # print(x_.shape)

            x_fft = torch.fft.rfft2(x_0)
            x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_ = self.kernel_ins_mid(x_)
            x_ = self.kernel_blks_mid(x_)
            x_ = self.kernel_outs_mid(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_1 = torch.fft.irfft2(otf * x_fft)
            if self.geo_aug:
                x_s = x_1.chunk(4, dim=1)
                x_os = list()
                for j in range(4):
                    x_j = x_s[j]
                    if j % 2 == 1:
                        x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
                    if j % 2 == 0:
                        x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
                    x_os.append(x_j)
                x_1 = torch.cat(x_os, dim=1)
            x_1 = self.n1(x_1)
            inp_dec_level2 = self.proj_feature_outs_mid(torch.cat([x_0, x_1], dim=1))
            if (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                inp_dec_level2 = window_reversex(inp_dec_level2, self.ker_size[level], H, W, batch_list)
            else:
                inp_dec_level2 = self.grids_inverse(inp_dec_level2, level=level)
            outputs.append(self.up_decs_mid(inp_dec_level2) + inp_img)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = inp_dec_level1 + out_enc_level1

        if self.kernel_learning[0]:
            level = 0
            b, c, H, W = inp_dec_level1.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_0, batch_list = window_partitionx(inp_dec_level1, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_0 = self.grids(inp_dec_level1, level=level)
                # print(x_.shape)

            x_fft = torch.fft.rfft2(x_0)
            x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_ = self.kernel_ins_max(x_)
            x_ = self.kernel_blks_max(x_)
            x_ = self.kernel_outs_max(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_1 = torch.fft.irfft2(otf * x_fft)
            # print(x_.shape)
            if self.geo_aug:
                x_s = x_1.chunk(4, dim=1)
                x_os = list()
                for j in range(4):
                    x_j = x_s[j]
                    if j % 2 == 1:
                        x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
                    if j % 2 == 0:
                        x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
                    x_os.append(x_j)
                x_1 = torch.cat(x_os, dim=1)
            x_1 = self.n0(x_1)
            inp_dec_level1 = self.proj_feature_outs_max(torch.cat([x_0, x_1], dim=1))
            # print(inp_img.shape, inp_dec_level1.shape)
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                inp_dec_level1 = window_reversex(inp_dec_level1, self.ker_size[level], H, W, batch_list)
            else:
                inp_dec_level1 = self.grids_inverse(inp_dec_level1, level=level)
            if self.flip_transpose_max:
                inp_dec_level1_ = torch.flip(inp_dec_level1, dims=(-2, -1)).contiguous()
                inp_dec_level1_ = torch.transpose(inp_dec_level1_, dim0=-2, dim1=-1).contiguous()
            else:
                inp_dec_level1_ = inp_dec_level1
            outputs.append(self.up_decs_max(inp_dec_level1_) + inp_img)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img
        outputs.append(out_dec_level1)
        # outputs.append(out_dict['restored'])
        # print('xxxxxxxxxxxxxxxxxxx')
        return outputs
@ARCH_REGISTRY.register()
class uniDeblurS(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[1, 1, 1],
        baseblock=['localnaf', 'localnaf', 'localnaf'],
        ker_baseblock='EVSS',
        ffn_expansion_factor=3,
        bias=False,
        train_size=[256, 256],
        patch_size=[256, 256],
        window_size=8,
        prior_yml='',
        model_weight=''
    ):
        super().__init__()

        import yaml

        try:
            from yaml import CLoader as Loader
        except ImportError:
            from yaml import Loader

        yaml_file = prior_yml  # '/data/mxt_code/DeepMXT/Motion_Deblurring/Options/Abalation-DeepRFTX-200k-8gpu.yml'
        x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

        s = x['network_g'].pop('type')
        ##########################
        # print(x['network_g'])

        self.model_patch = uniDeblurInf(**x['network_g'])

        checkpoint = torch.load(model_weight)
        self.model_patch.load_state_dict(checkpoint['params_ema'], strict=True)
        self.model_patch.eval()
        for p in self.model_patch.parameters(): p.requires_grad = False
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.sam = SAM(dim, 3, bias)
        self.concat_conv = nn.Conv2d(dim*2, dim, kernel_size=1, stride=1, padding=0, bias=bias)
        self.encoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.encoder_level1.add_module(f"block{i}", block)

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.encoder_level2.add_module(f"block{i}", block)

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.encoder_level3.add_module(f"block{i}", block)

        self.decoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.decoder_level3.add_module(f"block{i}", block)

        self.up3_2 = Upsample(int(dim * 2 ** 2))

        self.decoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.decoder_level2.add_module(f"block{i}", block)

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.decoder_level1.add_module(f"block{i}", block)

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, gt=None):
        outputs = []
        with torch.no_grad():
            out_dict = self.model_patch(inp_img)
        x = self.sam(out_dict['feature'], out_dict['restored'])

        inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_level1 = self.concat_conv(torch.cat([inp_enc_level1, x], dim=1))
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = inp_dec_level2 + out_enc_level2

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = inp_dec_level1 + out_enc_level1

        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img
        outputs.append(out_dec_level1)
        outputs.append(out_dict['restored'])
        # print('xxxxxxxxxxxxxxxxxxx')
        return outputs
@ARCH_REGISTRY.register()
class uniDeblurE(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[6, 6, 12],
        baseblock=['localnaf', 'localnaf', 'localnaf'],
        ker_baseblock='EVSS',
        ffn_expansion_factor=3,
        bias=False,
        train_size=[256, 256],
        patch_size=[256, 256],
        ker_size=[[256, 256], [128, 128], [64, 64]],
        overlap_size=[[32, 32], [16, 16], [8, 8]],
        k_channels=[48, 96, 192],
        ker_start=1,
        window_size=8,
        global_fft=[False, False, False],
        patch_size_fft=[128, 128],
        kernel_learning=[False, False, False],
        dec_with_enc=False,
        geo_aug=False,
        flip_transpose_max=False
    ):
        super().__init__()
        self.kernel_learning = kernel_learning
        self.flip_transpose_max = flip_transpose_max
        self.geo_aug = geo_aug
        self.ker_start = ker_start
        self.patch_size = ker_size
        self.ker_size = ker_size
        self.out_channels = [dim, dim*2, dim*4]
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
        self.encoder = True
        self.dec_with_enc = dec_with_enc
        if self.kernel_learning[2]:
            if self.geo_aug:
                self.kernel_ins = nn.ModuleList()
                self.kernel_blks = nn.ModuleList()
                self.kernel_outs = nn.ModuleList()
                for _ in range(4):
                    self.kernel_ins.append(nn.Sequential(
                        nn.Conv2d(dim * 1 * 2, dim * 2 * 2, 1, bias=bias)
                    ))
                    # self.kernel_blks = nn.Sequential(
                    #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
                    # )
                    kernel_blks = nn.Sequential()
                    for i in range(4):
                        block = get_basicblock(ker_baseblock, channels=dim * 2 * 2, num_heads=1, sub_idx=i,
                                               train_size=[train_size[0] // 4, train_size[1] // 4],
                                               patch_size=[patch_size[0] // 4, patch_size[1] // 4])
                        kernel_blks.add_module(f"block{i}", block)
                    self.kernel_blks.append(kernel_blks)
                    self.kernel_outs.append(nn.Sequential(
                        LayerNorm(dim * 2 * 2),
                        nn.Conv2d(dim * 2 * 2, dim * 1 * 2, 1, bias=bias)
                    ))
            else:
                self.kernel_ins = nn.Sequential(
                    nn.Conv2d(dim * 4 * 2, k_channels[2], 1, bias=bias)
                )
                # self.kernel_blks = nn.Sequential(
                #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
                # )
                self.kernel_blks = nn.Sequential()
                for i in range(4):
                    block = get_basicblock(ker_baseblock, channels=k_channels[2], num_heads=1, sub_idx=i,
                                           train_size=[train_size[0]//4, train_size[1]//4],
                                           patch_size=[patch_size[0]//4, patch_size[1]//4])
                    self.kernel_blks.add_module(f"block{i}", block)

                self.kernel_outs = nn.Sequential(
                    LayerNorm(k_channels[2]),
                    nn.Conv2d(k_channels[2], dim * 4 * 2, 1, bias=bias)
                )

            self.proj_feature_outs = nn.Sequential(
                nn.Conv2d(dim * 4 * 2, dim * 4, 1, bias=bias)
            )
            if self.dec_with_enc:
                self.up_decs1 = nn.Sequential(
                    nn.Conv2d(dim * 4 * 2, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2)
                )
                self.up_decs2 = nn.Sequential(
                    nn.Conv2d(dim * 2 * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2)
                )
                self.up_decs3 = nn.Sequential(
                    nn.Conv2d(in_channels=dim * 2, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
            else:
                self.up_decs = nn.Sequential(
                    nn.Conv2d(dim * 4, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
        if self.kernel_learning[1]:
            self.kernel_ins_mid = nn.Sequential(
                nn.Conv2d(dim * 2 * 2, k_channels[1], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks_mid = nn.Sequential()
            for i in range(4):
                block = get_basicblock(ker_baseblock, channels=k_channels[1], num_heads=1, sub_idx=i,
                                       train_size=[train_size[0] // 2, train_size[1] // 2],
                                       patch_size=[patch_size[0] // 2, patch_size[1] // 2])
                self.kernel_blks_mid.add_module(f"block{i}", block)

            self.kernel_outs_mid = nn.Sequential(
                LayerNorm(k_channels[1]),
                nn.Conv2d(k_channels[1], dim * 2 * 2, 1, bias=bias)
            )
            self.up_decs_mid = nn.Sequential(
                nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=bias)
            )
            self.proj_feature_outs_mid = nn.Sequential(
                nn.Conv2d(dim * 2 * 2, dim * 2, 1, bias=bias)
            )
        if self.kernel_learning[0]:
            self.kernel_ins_max = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, k_channels[0], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks_max = nn.Sequential()
            for i in range(4):
                block = get_basicblock(ker_baseblock, channels=k_channels[0], num_heads=1, sub_idx=i,
                                       train_size=[train_size[0] // 2, train_size[1] // 2],
                                       patch_size=[patch_size[0] // 2, patch_size[1] // 2])
                self.kernel_blks_max.add_module(f"block{i}", block)

            self.kernel_outs_max = nn.Sequential(
                LayerNorm(k_channels[0]),
                nn.Conv2d(k_channels[0], dim * 1 * 2, 1, bias=bias)
            )
            self.up_decs_max = nn.Sequential(
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=bias)
            )
            self.proj_feature_outs_max = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, dim * 1, 1, bias=bias)
            )
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.encoder_level1.add_module(f"block{i}", block)

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.encoder_level2.add_module(f"block{i}", block)

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.encoder_level3.add_module(f"block{i}", block)

        self.decoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.decoder_level3.add_module(f"block{i}", block)

        self.up3_2 = Upsample(int(dim * 2 ** 2))

        self.decoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.decoder_level2.add_module(f"block{i}", block)

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.decoder_level1.add_module(f"block{i}", block)

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        for p in self.patch_embed.parameters(): p.requires_grad = False
        for p in self.encoder_level1.parameters(): p.requires_grad = False
        for p in self.encoder_level2.parameters(): p.requires_grad = False
        for p in self.encoder_level3.parameters(): p.requires_grad = False
        for p in self.down1_2.parameters(): p.requires_grad = False
        for p in self.down2_3.parameters(): p.requires_grad = False
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
    def forward(self, inp_img, gt=None):
        outputs = []
        with torch.no_grad():
            inp_enc_level1 = self.patch_embed(inp_img)
            out_enc_level1 = self.encoder_level1(inp_enc_level1)

            inp_enc_level2 = self.down1_2(out_enc_level1)

            out_enc_level2 = self.encoder_level2(inp_enc_level2)

            inp_enc_level3 = self.down2_3(out_enc_level2)

            out_enc_level3 = self.encoder_level3(inp_enc_level3)
        if self.kernel_learning[2]:
            level = 2
            b, c, H, W = out_enc_level3.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_, batch_list = window_partitionx(out_enc_level3, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_ = self.grids(out_enc_level3, level=level)
                # print(x_.shape)
            if self.geo_aug:
                x_s = x_.chunk(4, dim=1)
                x_os = list()
                for j in range(4):
                    x_j = x_s[j]
                    if j % 2 == 1:
                        x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
                    if j % 2 == 0:
                        x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
                    x_fft = torch.fft.rfft2(x_j)
                    x_j = torch.cat([x_fft.real, x_fft.imag], dim=1)
                    x_j = self.kernel_ins[j](x_j)
                    x_j = self.kernel_blks[j](x_j)
                    x_j = self.kernel_outs[j](x_j)
                    x_real, x_imag = torch.chunk(x_j, 2, dim=1)
                    otf = torch.complex(x_real, x_imag)
                    # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
                    # print(x_fft.shape, otf.shape)
                    x_j = torch.fft.irfft2(otf * x_fft)
                    if j % 2 == 1:
                        x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
                    if j % 2 == 0:
                        x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
                    x_os.append(x_j)
                x_ = torch.cat(x_os, dim=1)
            else:
                x_fft = torch.fft.rfft2(x_)
                x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
                x_ = self.kernel_ins(x_)
                x_ = self.kernel_blks(x_)
                x_ = self.kernel_outs(x_)
                x_real, x_imag = torch.chunk(x_, 2, dim=1)
                otf = torch.complex(x_real, x_imag)
                # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
                # print(x_fft.shape, otf.shape)
                x_ = torch.fft.irfft2(otf * x_fft)
            # print(x_.shape)
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_ = window_reversex(x_, self.ker_size[level], H, W, batch_list)
            else:
                x_ = self.grids_inverse(x_, level=level)
            out_enc_level3 = self.proj_feature_outs(torch.cat([out_enc_level3, x_], dim=1))
            if self.dec_with_enc:
                x0 = self.up_decs1(torch.cat([out_enc_level3, x_], dim=1))
                x0 = self.up_decs2(torch.cat([out_enc_level2, x0], dim=1))
                x0 = self.up_decs3(torch.cat([out_enc_level1, x0], dim=1))
                outputs.append(x0 + inp_img)
            else:
                outputs.append(self.up_decs(out_enc_level3) + inp_img)

        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = inp_dec_level2 + out_enc_level2

        if self.kernel_learning[1]:
            level = 1
            b, c, H, W = inp_dec_level2.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_, batch_list = window_partitionx(inp_dec_level2, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_ = self.grids(inp_dec_level2, level=level)
                # print(x_.shape)

            x_fft = torch.fft.rfft2(x_)
            x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_ = self.kernel_ins_mid(x_)
            x_ = self.kernel_blks_mid(x_)
            x_ = self.kernel_outs_mid(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_ = torch.fft.irfft2(otf * x_fft)

            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_ = window_reversex(x_, self.ker_size[level], H, W, batch_list)
            else:
                x_ = self.grids_inverse(x_, level=level)
            inp_dec_level2 = self.proj_feature_outs_mid(torch.cat([inp_dec_level2, x_], dim=1))

            outputs.append(self.up_decs_mid(inp_dec_level2) + inp_img)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = inp_dec_level1 + out_enc_level1

        if self.kernel_learning[0]:
            level = 0
            b, c, H, W = inp_dec_level1.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_, batch_list = window_partitionx(inp_dec_level1, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_ = self.grids(inp_dec_level1, level=level)
                # print(x_.shape)

            x_fft = torch.fft.rfft2(x_)
            x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_ = self.kernel_ins_max(x_)
            x_ = self.kernel_blks_max(x_)
            x_ = self.kernel_outs_max(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_ = torch.fft.irfft2(otf * x_fft)
            # print(x_.shape)
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_ = window_reversex(x_, self.ker_size[level], H, W, batch_list)
            else:
                x_ = self.grids_inverse(x_, level=level)

            inp_dec_level1 = self.proj_feature_outs_max(torch.cat([inp_dec_level1, x_], dim=1))
            # print(inp_img.shape, inp_dec_level1.shape)
            if self.flip_transpose_max:
                inp_dec_level1_ = torch.flip(inp_dec_level1, dims=(-2, -1)).contiguous()
                inp_dec_level1_ = torch.transpose(inp_dec_level1_, dim0=-2, dim1=-1).contiguous()
            else:
                inp_dec_level1_ = inp_dec_level1
            outputs.append(self.up_decs_max(inp_dec_level1_) + inp_img)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img
        outputs.append(out_dec_level1)
        # print('xxxxxxxxxxxxxxxxxxx')
        return outputs
@ARCH_REGISTRY.register()
class uniDeblurSlide(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[6, 6, 12],
        baseblock=['localnaf', 'localnaf', 'localnaf'],
        ker_baseblock='EVSS',
        ffn_expansion_factor=3,
        bias=False,
        train_size=[256, 256],
        patch_size=[256, 256],
        ker_size=[[256, 256], [128, 128], [64, 64]],
        overlap_size=[[32, 32], [16, 16], [8, 8]],
        k_channels=[48, 96, 192],
        ker_start=1,
        window_size=8,
        global_fft=[False, False, False],
        patch_size_fft=[128, 128],
        kernel_learning=[False, False, False],
        dec_with_enc=False,
        geo_aug=False,
        flip_transpose_max=False
    ):
        super().__init__()
        self.kernel_learning = kernel_learning
        self.flip_transpose_max = flip_transpose_max
        self.geo_aug = geo_aug
        self.ker_start = ker_start
        self.patch_size = ker_size
        self.ker_size = ker_size
        self.out_channels = [dim, dim*2, dim*4]
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
        self.encoder = True
        self.dec_with_enc = dec_with_enc
        if self.kernel_learning[2]:
            if self.geo_aug:
                self.kernel_ins = nn.ModuleList()
                self.kernel_blks = nn.ModuleList()
                self.kernel_outs = nn.ModuleList()
                for _ in range(4):
                    self.kernel_ins.append(nn.Sequential(
                        nn.Conv2d(dim * 1 * 2, dim * 2 * 2, 1, bias=bias)
                    ))
                    # self.kernel_blks = nn.Sequential(
                    #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
                    # )
                    kernel_blks = nn.Sequential()
                    for i in range(4):
                        block = get_basicblock(ker_baseblock, channels=dim * 2 * 2, num_heads=1, sub_idx=i,
                                               train_size=[train_size[0] // 4, train_size[1] // 4],
                                               patch_size=[patch_size[0] // 4, patch_size[1] // 4])
                        kernel_blks.add_module(f"block{i}", block)
                    self.kernel_blks.append(kernel_blks)
                    self.kernel_outs.append(nn.Sequential(
                        LayerNorm(dim * 2 * 2),
                        nn.Conv2d(dim * 2 * 2, dim * 1 * 2, 1, bias=bias)
                    ))
            else:
                self.kernel_ins = nn.Sequential(
                    nn.Conv2d(dim * 4 * 2, k_channels[2], 1, bias=bias)
                )
                # self.kernel_blks = nn.Sequential(
                #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
                # )
                self.kernel_blks = nn.Sequential()
                for i in range(4):
                    block = get_basicblock(ker_baseblock, channels=k_channels[2], num_heads=1, sub_idx=i,
                                           train_size=[train_size[0]//4, train_size[1]//4],
                                           patch_size=[patch_size[0]//4, patch_size[1]//4])
                    self.kernel_blks.add_module(f"block{i}", block)

                self.kernel_outs = nn.Sequential(
                    LayerNorm(k_channels[2]),
                    nn.Conv2d(k_channels[2], dim * 4 * 2, 1, bias=bias)
                )

            self.proj_feature_outs = nn.Sequential(
                nn.Conv2d(dim * 4 * 2, dim * 4, 1, bias=bias)
            )
            if self.dec_with_enc:
                self.up_decs1 = nn.Sequential(
                    nn.Conv2d(dim * 4 * 2, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2)
                )
                self.up_decs2 = nn.Sequential(
                    nn.Conv2d(dim * 2 * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2)
                )
                self.up_decs3 = nn.Sequential(
                    nn.Conv2d(in_channels=dim * 2, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
            else:
                self.up_decs = nn.Sequential(
                    nn.Conv2d(dim * 4, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
        if self.kernel_learning[1]:
            self.kernel_ins_mid = nn.Sequential(
                nn.Conv2d(dim * 2 * 2, k_channels[1], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks_mid = nn.Sequential()
            for i in range(4):
                block = get_basicblock(ker_baseblock, channels=k_channels[1], num_heads=1, sub_idx=i,
                                       train_size=[train_size[0] // 2, train_size[1] // 2],
                                       patch_size=[patch_size[0] // 2, patch_size[1] // 2])
                self.kernel_blks_mid.add_module(f"block{i}", block)

            self.kernel_outs_mid = nn.Sequential(
                LayerNorm(k_channels[1]),
                nn.Conv2d(k_channels[1], dim * 2 * 2, 1, bias=bias)
            )
            self.up_decs_mid = nn.Sequential(
                nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=bias)
            )
            self.proj_feature_outs_mid = nn.Sequential(
                nn.Conv2d(dim * 2 * 2, dim * 2, 1, bias=bias)
            )
        if self.kernel_learning[0]:
            self.kernel_ins_max = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, k_channels[0], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks_max = nn.Sequential()
            for i in range(4):
                block = get_basicblock(ker_baseblock, channels=k_channels[0], num_heads=1, sub_idx=i,
                                       train_size=[train_size[0] // 2, train_size[1] // 2],
                                       patch_size=[patch_size[0] // 2, patch_size[1] // 2])
                self.kernel_blks_max.add_module(f"block{i}", block)

            self.kernel_outs_max = nn.Sequential(
                LayerNorm(k_channels[0]),
                nn.Conv2d(k_channels[0], dim * 1 * 2, 1, bias=bias)
            )
            self.up_decs_max = nn.Sequential(
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=bias)
            )
            self.proj_feature_outs_max = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, dim * 1, 1, bias=bias)
            )

        for p in self.kernel_ins.parameters(): p.requires_grad = False
        for p in self.kernel_blks.parameters(): p.requires_grad = False
        for p in self.kernel_outs.parameters(): p.requires_grad = False
        for p in self.up_decs.parameters(): p.requires_grad = False
        for p in self.proj_feature_outs.parameters(): p.requires_grad = False

        for p in self.kernel_ins_mid.parameters(): p.requires_grad = False
        for p in self.kernel_blks_mid.parameters(): p.requires_grad = False
        for p in self.kernel_outs_mid.parameters(): p.requires_grad = False
        for p in self.up_decs_mid.parameters(): p.requires_grad = False
        for p in self.proj_feature_outs_mid.parameters(): p.requires_grad = False

        for p in self.kernel_ins_max.parameters(): p.requires_grad = False
        for p in self.kernel_blks_max.parameters(): p.requires_grad = False
        for p in self.kernel_outs_max.parameters(): p.requires_grad = False
        for p in self.up_decs_max.parameters(): p.requires_grad = False
        for p in self.proj_feature_outs_max.parameters(): p.requires_grad = False
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.encoder_level1.add_module(f"block{i}", block)

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.encoder_level2.add_module(f"block{i}", block)

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.encoder_level3.add_module(f"block{i}", block)
        for p in self.patch_embed.parameters(): p.requires_grad = False
        for p in self.encoder_level1.parameters(): p.requires_grad = False
        for p in self.encoder_level2.parameters(): p.requires_grad = False
        for p in self.encoder_level3.parameters(): p.requires_grad = False
        for p in self.down1_2.parameters(): p.requires_grad = False
        for p in self.down2_3.parameters(): p.requires_grad = False
        self.decoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.decoder_level3.add_module(f"block{i}", block)

        self.up3_2 = Upsample(int(dim * 2 ** 2))

        self.decoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.decoder_level2.add_module(f"block{i}", block)

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.decoder_level1.add_module(f"block{i}", block)

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
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
    def forward(self, inp_img, gt=None):
        outputs = []
        with torch.no_grad():
            inp_enc_level1 = self.patch_embed(inp_img)
            out_enc_level1 = self.encoder_level1(inp_enc_level1)

            inp_enc_level2 = self.down1_2(out_enc_level1)

            out_enc_level2 = self.encoder_level2(inp_enc_level2)

            inp_enc_level3 = self.down2_3(out_enc_level2)

            out_enc_level3 = self.encoder_level3(inp_enc_level3)
            if self.kernel_learning[2]:
                level = 2
                b, c, H, W = out_enc_level3.shape
                if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                    x_, batch_list = window_partitionx(out_enc_level3, self.ker_size[level])
                else:
                    # print('11111111111111111')
                    # print('level: ', level)
                    x_ = self.grids(out_enc_level3, level=level)
                    # print(x_.shape)
                if self.geo_aug:
                    x_s = x_.chunk(4, dim=1)
                    x_os = list()
                    for j in range(4):
                        x_j = x_s[j]
                        if j % 2 == 1:
                            x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
                        if j % 2 == 0:
                            x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
                        x_fft = torch.fft.rfft2(x_j)
                        x_j = torch.cat([x_fft.real, x_fft.imag], dim=1)
                        x_j = self.kernel_ins[j](x_j)
                        x_j = self.kernel_blks[j](x_j)
                        x_j = self.kernel_outs[j](x_j)
                        x_real, x_imag = torch.chunk(x_j, 2, dim=1)
                        otf = torch.complex(x_real, x_imag)
                        # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
                        # print(x_fft.shape, otf.shape)
                        x_j = torch.fft.irfft2(otf * x_fft)
                        if j % 2 == 1:
                            x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
                        if j % 2 == 0:
                            x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
                        x_os.append(x_j)
                    x_ = torch.cat(x_os, dim=1)
                else:
                    x_fft = torch.fft.rfft2(x_)
                    x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
                    x_ = self.kernel_ins(x_)
                    x_ = self.kernel_blks(x_)
                    x_ = self.kernel_outs(x_)
                    x_real, x_imag = torch.chunk(x_, 2, dim=1)
                    otf = torch.complex(x_real, x_imag)
                    # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
                    # print(x_fft.shape, otf.shape)
                    x_ = torch.fft.irfft2(otf * x_fft)
                # print(x_.shape)
                if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                    x_ = window_reversex(x_, self.ker_size[level], H, W, batch_list, mask=True)
                else:
                    x_ = self.grids_inverse(x_, level=level)
                out_enc_level3 = self.proj_feature_outs(torch.cat([out_enc_level3, x_], dim=1))
                # if self.dec_with_enc:
                #     x0 = self.up_decs1(torch.cat([out_enc_level3, x_], dim=1))
                #     x0 = self.up_decs2(torch.cat([out_enc_level2, x0], dim=1))
                #     x0 = self.up_decs3(torch.cat([out_enc_level1, x0], dim=1))
                #     outputs.append(x0 + inp_img)
                # else:
                #     outputs.append(self.up_decs(out_enc_level3) + inp_img)

        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = inp_dec_level2 + out_enc_level2

        if self.kernel_learning[1]:
            level = 1
            b, c, H, W = inp_dec_level2.shape
            # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
            #     x_, batch_list = window_partitionx(inp_dec_level2, self.ker_size[level])
            # else:
                # print('11111111111111111')
                # print('level: ', level)
            x_ = self.grids(inp_dec_level2, level=level)
                # print(x_.shape)

            x_fft = torch.fft.rfft2(x_)
            x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_ = self.kernel_ins_mid(x_)
            x_ = self.kernel_blks_mid(x_)
            x_ = self.kernel_outs_mid(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_ = torch.fft.irfft2(otf * x_fft)

            # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
            #     x_ = window_reversex(x_, self.ker_size[level], H, W, batch_list, mask=True)
            # else:
            x_ = self.grids_inverse(x_, level=level)
            inp_dec_level2 = self.proj_feature_outs_mid(torch.cat([inp_dec_level2, x_], dim=1))

            # outputs.append(self.up_decs_mid(inp_dec_level2) + inp_img)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = inp_dec_level1 + out_enc_level1

        if self.kernel_learning[0]:
            level = 0
            b, c, H, W = inp_dec_level1.shape
            # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
            #     x_, batch_list = window_partitionx(inp_dec_level1, self.ker_size[level])
            # else:
                # print('11111111111111111')
                # print('level: ', level)
            x_ = self.grids(inp_dec_level1, level=level)
                # print(x_.shape)

            x_fft = torch.fft.rfft2(x_)
            x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_ = self.kernel_ins_max(x_)
            x_ = self.kernel_blks_max(x_)
            x_ = self.kernel_outs_max(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_ = torch.fft.irfft2(otf * x_fft)
            # print(x_.shape)
            # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
            #     x_ = window_reversex(x_, self.ker_size[level], H, W, batch_list, mask=True)
            # else:
            x_ = self.grids_inverse(x_, level=level)

            inp_dec_level1 = self.proj_feature_outs_max(torch.cat([inp_dec_level1, x_], dim=1))
            # # print(inp_img.shape, inp_dec_level1.shape)
            # if self.flip_transpose_max:
            #     inp_dec_level1_ = torch.flip(inp_dec_level1, dims=(-2, -1)).contiguous()
            #     inp_dec_level1_ = torch.transpose(inp_dec_level1_, dim0=-2, dim1=-1).contiguous()
            # else:
            #     inp_dec_level1_ = inp_dec_level1
            # outputs.append(self.up_decs_max(inp_dec_level1_) + inp_img)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img
        outputs.append(out_dec_level1)
        # print('xxxxxxxxxxxxxxxxxxx')
        return outputs
@ARCH_REGISTRY.register()
class uniDeblurL(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks_enc=[6, 6, 12],
        num_blocks_dec=[6, 6, 12],
        baseblock=['localnaf', 'localnaf', 'localnaf'],
        num_ker_blocks=[2, 3, 3, 4],
        ker_baseblock='EVSS',
        ffn_expansion_factor=3,
        bias=False,
        train_size=[256, 256],
        patch_size=[256, 256],
        ker_size=[[256, 256], [256, 256], [256, 256], [256, 256]],
        overlap_size=[[32, 32], [32, 32], [32, 32], [32, 32]],
        k_channels=[48, 48, 48, 48],
        ker_start=1,
        window_size=8,
        global_fft=[False, False, False],
        patch_size_fft=[128, 128],
        kernel_learning=[False, False, False, False],
        dec_with_enc=False,
        geo_aug=False,
        flip_transpose_max=False
    ):
        super().__init__()
        self.kernel_learning = kernel_learning
        self.flip_transpose_max = flip_transpose_max
        self.geo_aug = geo_aug
        self.ker_start = ker_start
        self.patch_size = ker_size
        self.ker_size = ker_size
        self.out_channels = [dim, dim, dim, dim]
        self.overlap_size = overlap_size
        self.up_scale = 1
        self.overlap_size_up = self.overlap_size  # (self.overlap_size[0] * self.up_scale, self.overlap_size[1] * self.up_scale)
        # self.kernel_size = [train_size, train_size]
        self.kernel_size_up = self.ker_size
        self.nr = [1, 1, 1, 1]
        self.nc = [1, 1, 1, 1]
        self.stride = [-1, -1, -1, -1]
        self.h = [-1, -1, -1, -1]
        self.w = [-1, -1, -1, -1]
        self.ek1, self.ek2 = [0, 0, 0, 0], [0, 0, 0, 0]
        self.num_patchs = [ 1, 1, 1, 1 ]
        self.original_size = [[], [], [], []]
        self.idxes = [None, None, None, None]
        self.fuse_matrix_w1 = [None, None, None, None]
        self.fuse_matrix_w2 = [None, None, None, None]
        self.fuse_matrix_h1 = [None, None, None, None]
        self.fuse_matrix_h2 = [None, None, None, None]

        self.fuse_matrix_eh1 = [None, None, None, None]
        self.fuse_matrix_eh2 = [None, None, None, None]

        self.fuse_matrix_ew1 = [None, None, None, None]
        self.fuse_matrix_ew2 = [None, None, None, None]
        self.encoder = True
        self.dec_with_enc = dec_with_enc
        if self.kernel_learning[3]:
            if self.geo_aug:
                self.kernel_ins = nn.ModuleList()
                self.kernel_blks = nn.ModuleList()
                self.kernel_outs = nn.ModuleList()
                for _ in range(4):
                    self.kernel_ins.append(nn.Sequential(
                        nn.Conv2d(dim * 1 * 2, dim * 2 * 2, 1, bias=bias)
                    ))
                    # self.kernel_blks = nn.Sequential(
                    #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
                    # )
                    kernel_blks = nn.Sequential()
                    for i in range(4):
                        block = get_basicblock(ker_baseblock, channels=dim * 2 * 2, num_heads=1, sub_idx=i,
                                               train_size=[train_size[0] // 4, train_size[1] // 4],
                                               patch_size=[patch_size[0] // 4, patch_size[1] // 4])
                        kernel_blks.add_module(f"block{i}", block)
                    self.kernel_blks.append(kernel_blks)
                    self.kernel_outs.append(nn.Sequential(
                        LayerNorm(dim * 2 * 2),
                        nn.Conv2d(dim * 2 * 2, dim * 1 * 2, 1, bias=bias)
                    ))
            else:
                # self.x_ins_small = nn.Sequential(
                #     nn.Conv2d(dim * 1 * 2, dim * 1, 1, bias=bias)
                # )
                self.kernel_ins_small = nn.Sequential(
                    nn.Conv2d(dim * 1 * 2, k_channels[3], 1, bias=bias)
                )
                # self.kernel_blks = nn.Sequential(
                #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
                # )
                self.kernel_blks_small = nn.Sequential()
                for i in range(num_ker_blocks[3]):
                    block = get_basicblock(ker_baseblock, channels=k_channels[3], num_heads=1, sub_idx=i,
                                           train_size=[train_size[0]//1, train_size[1]//1],
                                           patch_size=[patch_size[0]//1, patch_size[1]//1])
                    self.kernel_blks_small.add_module(f"block{i}", block)

                self.kernel_outs_small = nn.Sequential(
                    LayerNorm(k_channels[3]),
                    nn.Conv2d(k_channels[3], dim * 1 * 2, 1, bias=bias)
                )

            self.proj_feature_outs_small = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, dim * 1, 1, bias=bias)
            )

            self.up_decs_small = nn.Sequential(
                nn.Conv2d(dim * 4, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2),
                nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                # nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                #           bias=bias)
            )
            self.output_decs_small = nn.Sequential(
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=bias)
            )
        if self.kernel_learning[2]:
            self.x_ins_mid = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, dim * 1, 1, bias=bias)
            )
            self.kernel_ins_mid = nn.Sequential(
                nn.Conv2d(dim * 2 * 2, k_channels[2], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks_mid = nn.Sequential()
            for i in range(num_ker_blocks[2]):
                block = get_basicblock(ker_baseblock, channels=k_channels[2], num_heads=1, sub_idx=i,
                                       train_size=[train_size[0] // 1, train_size[1] // 1],
                                       patch_size=[patch_size[0] // 1, patch_size[1] // 1])
                self.kernel_blks_mid.add_module(f"block{i}", block)

            self.kernel_outs_mid = nn.Sequential(
                LayerNorm(k_channels[2]),
                nn.Conv2d(k_channels[2], dim * 1 * 2, 1, bias=bias)
            )
            self.up_decs_mid = nn.Sequential(
                nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                # nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                #           bias=bias)
            )
            self.output_decs_mid = nn.Sequential(
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=bias)
            )
            self.proj_feature_outs_mid = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, dim * 1, 1, bias=bias)
            )
        if self.kernel_learning[1]:
            self.x_ins_max = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, dim * 1, 1, bias=bias)
            )
            self.kernel_ins_max = nn.Sequential(
                nn.Conv2d(dim * 2 * 2, k_channels[0], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks_max = nn.Sequential()
            for i in range(num_ker_blocks[1]):
                block = get_basicblock(ker_baseblock, channels=k_channels[1], num_heads=1, sub_idx=i,
                                       train_size=[train_size[0] // 2, train_size[1] // 2],
                                       patch_size=[patch_size[0] // 2, patch_size[1] // 2])
                self.kernel_blks_max.add_module(f"block{i}", block)

            self.kernel_outs_max = nn.Sequential(
                LayerNorm(k_channels[1]),
                nn.Conv2d(k_channels[1], dim * 1 * 2, 1, bias=bias)
            )
            self.output_decs_max = nn.Sequential(
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=bias)
            )
            self.proj_feature_outs_max = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, dim * 1, 1, bias=bias)
            )
        if self.kernel_learning[0]:
            self.x_ins = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, dim * 1, 1, bias=bias)
            )
            self.kernel_ins = nn.Sequential(
                nn.Conv2d(dim * 2 * 2, k_channels[0], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks = nn.Sequential()
            for i in range(num_ker_blocks[0]):
                block = get_basicblock(ker_baseblock, channels=k_channels[0], num_heads=1, sub_idx=i,
                                       train_size=[train_size[0] // 2, train_size[1] // 2],
                                       patch_size=[patch_size[0] // 2, patch_size[1] // 2])
                self.kernel_blks.add_module(f"block{i}", block)

            self.kernel_outs = nn.Sequential(
                LayerNorm(k_channels[0]),
                nn.Conv2d(k_channels[0], dim * 1 * 2, 1, bias=bias)
            )
            self.output_decs = nn.Sequential(
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=bias)
            )
            self.proj_feature_outs = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, dim * 1, 1, bias=bias)
            )
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential()
        for i in range(num_blocks_enc[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.encoder_level1.add_module(f"block{i}", block)

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential()
        for i in range(num_blocks_enc[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.encoder_level2.add_module(f"block{i}", block)

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential()
        for i in range(num_blocks_enc[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.encoder_level3.add_module(f"block{i}", block)

        self.decoder_level3 = nn.Sequential()
        for i in range(num_blocks_dec[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.decoder_level3.add_module(f"block{i}", block)

        self.up3_2 = Upsample(int(dim * 2 ** 2))

        self.decoder_level2 = nn.Sequential()
        for i in range(num_blocks_dec[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.decoder_level2.add_module(f"block{i}", block)

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential()
        for i in range(num_blocks_dec[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.decoder_level1.add_module(f"block{i}", block)

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
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

    def forward(self, inp_img, gt=None):
        outputs = []
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)


        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = inp_dec_level2 + out_enc_level2


        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = inp_dec_level1 + out_enc_level1

        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img
        outputs.append(out_dec_level1)

        if self.kernel_learning[3]:
            level = 3
            x3 = self.up_decs_small(out_enc_level3)
            b, c, H, W = x3.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x3, batch_list = window_partitionx(x3, self.ker_size[level])
            else:
                x3 = self.grids(x3, level=level)

            x_fft = torch.fft.rfft2(x3)
            x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_ = self.kernel_ins_small(x_)
            x_ = self.kernel_blks_small(x_)
            x_ = self.kernel_outs_small(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_ = torch.fft.irfft2(otf * x_fft)
            # print(x_.shape)

            x3 = self.proj_feature_outs_small(torch.cat([x3, x_], dim=1))

            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                o3 = window_reversex(x3, self.ker_size[level], H, W, batch_list)
            else:
                o3 = self.grids_inverse(x3, level=level)
            outputs.append(self.output_decs_small(o3) + inp_img)
        if self.kernel_learning[2]:
            level = 2
            x2 = self.up_decs_mid(inp_dec_level2)
            b, c, H, W = x2.shape

            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x2, batch_list = window_partitionx(x2, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x2 = self.grids(x2, level=level)
                # print(x_.shape)

            x_fft = torch.fft.rfft2(self.x_ins_mid(torch.cat([x2, x3], dim=1)))
            x_ = torch.cat([x_fft.real, x_fft.imag, otf.real, otf.imag], dim=1)
            x_ = self.kernel_ins_mid(x_)
            x_ = self.kernel_blks_mid(x_)
            x_ = self.kernel_outs_mid(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_ = torch.fft.irfft2(otf * x_fft)
            x2 = self.proj_feature_outs_mid(torch.cat([x2, x_], dim=1))
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                o2 = window_reversex(x2, self.ker_size[level], H, W, batch_list)
            else:
                o2 = self.grids_inverse(x2, level=level)


            outputs.append(self.output_decs_mid(o2) + inp_img)
        if self.kernel_learning[1]:
            level = 1
            b, c, H, W = inp_dec_level1.shape
            x1 = inp_dec_level1
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x1, batch_list = window_partitionx(x1, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x1 = self.grids(x1, level=level)
                # print(x_.shape)

            x_fft = torch.fft.rfft2(self.x_ins_max(torch.cat([x1, x2], dim=1)))

            x_ = torch.cat([x_fft.real, x_fft.imag, otf.real, otf.imag], dim=1)
            x_ = self.kernel_ins_max(x_)
            x_ = self.kernel_blks_max(x_)
            x_ = self.kernel_outs_max(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_ = torch.fft.irfft2(otf * x_fft)
            # print(x_.shape)


            x1 = self.proj_feature_outs_max(torch.cat([x1, x_], dim=1))
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                o1 = window_reversex(x1, self.ker_size[level], H, W, batch_list)
            else:
                o1 = self.grids_inverse(x1, level=level)
            outputs.append(self.output_decs_max(o1) + inp_img)
        if self.kernel_learning[0]:
            level = 0
            b, c, H, W = inp_dec_level1.shape
            x0 = inp_dec_level1
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x0, batch_list = window_partitionx(x0, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x0 = self.grids(x0, level=level)
                # print(x_.shape)

            x_fft = torch.fft.rfft2(self.x_ins(torch.cat([x0, x1], dim=1)))
            x_ = torch.cat([x_fft.real, x_fft.imag, otf.real, otf.imag], dim=1)
            x_ = self.kernel_ins(x_)
            x_ = self.kernel_blks(x_)
            x_ = self.kernel_outs(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_ = torch.fft.irfft2(otf * x_fft)
            # print(x_.shape)

            x0 = self.proj_feature_outs(torch.cat([x0, x_], dim=1))
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                o0 = window_reversex(x0, self.ker_size[level], H, W, batch_list)
            else:
                o0 = self.grids_inverse(x0, level=level)
            outputs.append(self.output_decs(o0) + inp_img)
        # print('xxxxxxxxxxxxxxxxxxx')
        return outputs

@ARCH_REGISTRY.register()
class uniDeblurSlide1(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[6, 6, 12],
        baseblock=['localnaf', 'localnaf', 'localnaf'],
        ker_baseblock='EVSS',
        ffn_expansion_factor=3,
        bias=False,
        train_size=[256, 256],
        patch_size=[256, 256],
        ker_size=[[256, 256], [128, 128], [64, 64]],
        overlap_size=[[32, 32], [16, 16], [8, 8]],
        k_channels=[48, 96, 192],
        ker_start=1,
        window_size=8,
        global_fft=[False, False, False],
        patch_size_fft=[128, 128],
        kernel_learning=[False, False, False],
            dec_with_enc=False,
            geo_aug=False

    ):
        super().__init__()
        self.kernel_learning = kernel_learning
        self.geo_aug = geo_aug
        self.ker_start = ker_start
        self.patch_size = ker_size
        self.ker_size = ker_size
        self.out_channels = [dim, dim*2, dim*4]
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
        self.encoder = True
        self.dec_with_enc = dec_with_enc
        if self.kernel_learning[2]:
            if self.geo_aug:
                self.kernel_ins = nn.ModuleList()
                self.kernel_blks = nn.ModuleList()
                self.kernel_outs = nn.ModuleList()
                for _ in range(4):
                    self.kernel_ins.append(nn.Sequential(
                        nn.Conv2d(dim * 1 * 2, dim * 2 * 2, 1, bias=bias)
                    ))
                    # self.kernel_blks = nn.Sequential(
                    #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
                    # )
                    kernel_blks = nn.Sequential()
                    for i in range(4):
                        block = get_basicblock(ker_baseblock, channels=dim * 2 * 2, num_heads=1, sub_idx=i,
                                               train_size=[train_size[0] // 4, train_size[1] // 4],
                                               patch_size=[patch_size[0] // 4, patch_size[1] // 4])
                        kernel_blks.add_module(f"block{i}", block)
                    self.kernel_blks.append(kernel_blks)
                    self.kernel_outs.append(nn.Sequential(
                        LayerNorm(dim * 2 * 2),
                        nn.Conv2d(dim * 2 * 2, dim * 1 * 2, 1, bias=bias)
                    ))
            else:
                self.kernel_ins = nn.Sequential(
                    nn.Conv2d(dim * 4 * 2, k_channels[2], 1, bias=bias)
                )
                # self.kernel_blks = nn.Sequential(
                #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
                # )
                self.kernel_blks = nn.Sequential()
                for i in range(4):
                    block = get_basicblock(ker_baseblock, channels=k_channels[2], num_heads=1, sub_idx=i,
                                           train_size=[train_size[0]//4, train_size[1]//4],
                                           patch_size=[patch_size[0]//4, patch_size[1]//4])
                    self.kernel_blks.add_module(f"block{i}", block)

                self.kernel_outs = nn.Sequential(
                    LayerNorm(k_channels[2]),
                    nn.Conv2d(k_channels[2], dim * 4 * 2, 1, bias=bias)
                )

            self.proj_feature_outs = nn.Sequential(
                nn.Conv2d(dim * 4 * 2, dim * 4, 1, bias=bias)
            )
            if self.dec_with_enc:
                self.up_decs1 = nn.Sequential(
                    nn.Conv2d(dim * 4 * 2, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2)
                )
                self.up_decs2 = nn.Sequential(
                    nn.Conv2d(dim * 2 * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2)
                )
                self.up_decs3 = nn.Sequential(
                    nn.Conv2d(in_channels=dim * 2, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
            else:
                self.up_decs = nn.Sequential(
                    nn.Conv2d(dim * 4, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                    nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=bias)
                )
        if self.kernel_learning[1]:
            self.kernel_ins_mid = nn.Sequential(
                nn.Conv2d(dim * 2 * 2, k_channels[1], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks_mid = nn.Sequential()
            for i in range(4):
                block = get_basicblock(ker_baseblock, channels=k_channels[1], num_heads=1, sub_idx=i,
                                       train_size=[train_size[0] // 2, train_size[1] // 2],
                                       patch_size=[patch_size[0] // 2, patch_size[1] // 2])
                self.kernel_blks_mid.add_module(f"block{i}", block)

            self.kernel_outs_mid = nn.Sequential(
                LayerNorm(k_channels[1]),
                nn.Conv2d(k_channels[1], dim * 2 * 2, 1, bias=bias)
            )
            self.up_decs_mid = nn.Sequential(
                nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=bias)
            )
            self.proj_feature_outs_mid = nn.Sequential(
                nn.Conv2d(dim * 2 * 2, dim * 2, 1, bias=bias)
            )
        if self.kernel_learning[0]:
            self.kernel_ins_max = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, k_channels[0], 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks_max = nn.Sequential()
            for i in range(4):
                block = get_basicblock(ker_baseblock, channels=k_channels[0], num_heads=1, sub_idx=i,
                                       train_size=[train_size[0] // 2, train_size[1] // 2],
                                       patch_size=[patch_size[0] // 2, patch_size[1] // 2])
                self.kernel_blks_max.add_module(f"block{i}", block)

            self.kernel_outs_max = nn.Sequential(
                LayerNorm(k_channels[0]),
                nn.Conv2d(k_channels[0], dim * 1 * 2, 1, bias=bias)
            )
            self.up_decs_max = nn.Sequential(
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=bias)
            )
            self.proj_feature_outs_max = nn.Sequential(
                nn.Conv2d(dim * 1 * 2, dim * 1, 1, bias=bias)
            )
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.encoder_level1.add_module(f"block{i}", block)

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.encoder_level2.add_module(f"block{i}", block)

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.encoder_level3.add_module(f"block{i}", block)

        self.decoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.decoder_level3.add_module(f"block{i}", block)

        self.up3_2 = Upsample(int(dim * 2 ** 2))

        self.decoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.decoder_level2.add_module(f"block{i}", block)

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.decoder_level1.add_module(f"block{i}", block)

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
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
    def forward(self, inp_img, gt=None):
        outputs = []
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        if self.kernel_learning[2]:
            level = 2
            b, c, H, W = out_enc_level3.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_, batch_list = window_partitionx(out_enc_level3, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_ = self.grids(out_enc_level3, level=level)
                # print(x_.shape)
            out_enc_level3 = x_
            if self.geo_aug:
                x_s = x_.chunk(4, dim=1)
                x_os = list()
                for j in range(4):
                    x_j = x_s[j]
                    if j % 2 == 1:
                        x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
                    if j % 2 == 0:
                        x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
                    x_fft = torch.fft.rfft2(x_j)
                    x_j = torch.cat([x_fft.real, x_fft.imag], dim=1)
                    x_j = self.kernel_ins[j](x_j)
                    x_j = self.kernel_blks[j](x_j)
                    x_j = self.kernel_outs[j](x_j)
                    x_real, x_imag = torch.chunk(x_j, 2, dim=1)
                    otf = torch.complex(x_real, x_imag)
                    # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
                    # print(x_fft.shape, otf.shape)
                    x_j = torch.fft.irfft2(otf * x_fft)
                    if j % 2 == 1:
                        x_j = torch.flip(x_j, dims=(-2, -1)).contiguous()
                    if j % 2 == 0:
                        x_j = torch.transpose(x_j, dim0=-2, dim1=-1).contiguous()
                    x_os.append(x_j)
                x_ = torch.cat(x_os, dim=1)
            else:
                x_fft = torch.fft.rfft2(x_)
                x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
                x_ = self.kernel_ins(x_)
                x_ = self.kernel_blks(x_)
                x_ = self.kernel_outs(x_)
                x_real, x_imag = torch.chunk(x_, 2, dim=1)
                otf = torch.complex(x_real, x_imag)
                # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
                # print(x_fft.shape, otf.shape)
                x_ = torch.fft.irfft2(otf * x_fft)
            # print(x_.shape)
            # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
            #     x_ = window_reversex(x_, self.ker_size[level], H, W, batch_list)
            # else:
            #     x_ = self.grids_inverse(x_, level=level)
            out_enc_level3 = self.proj_feature_outs(torch.cat([out_enc_level3, x_], dim=1))
            # if self.dec_with_enc:
            #     x0 = self.up_decs1(torch.cat([out_enc_level3, x_], dim=1))
            #     x0 = self.up_decs2(torch.cat([out_enc_level2, x0], dim=1))
            #     x0 = self.up_decs3(torch.cat([out_enc_level1, x0], dim=1))
            #     outputs.append(x0 + inp_img)
            # else:
            #     outputs.append(self.up_decs(out_enc_level3) + inp_img)

        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)



        if self.kernel_learning[1]:
            level = 1
            b, c, H, W = out_enc_level2.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_, batch_list = window_partitionx(out_enc_level2, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_ = self.grids(out_enc_level2, level=level)
                # print(x_.shape)
            out_enc_level2 = x_
            inp_dec_level2 = inp_dec_level2 + out_enc_level2
            x_fft = torch.fft.rfft2(x_)
            x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_ = self.kernel_ins_mid(x_)
            x_ = self.kernel_blks_mid(x_)
            x_ = self.kernel_outs_mid(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_ = torch.fft.irfft2(otf * x_fft)

            # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
            #     x_ = window_reversex(x_, self.ker_size[level], H, W, batch_list)
            # else:
            #     x_ = self.grids_inverse(x_, level=level)
            inp_dec_level2 = self.proj_feature_outs_mid(torch.cat([inp_dec_level2, x_], dim=1))

            # outputs.append(self.up_decs_mid(inp_dec_level2) + inp_img)
        else:
            inp_dec_level2 = inp_dec_level2 + out_enc_level2
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        # inp_dec_level1 = inp_dec_level1 + out_enc_level1

        if self.kernel_learning[0]:
            level = 0
            b, c, H, W = out_enc_level1.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_, batch_list = window_partitionx(out_enc_level1, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_ = self.grids(out_enc_level1, level=level)
                # print(x_.shape)
            out_enc_level1 = x_
            inp_dec_level1 = inp_dec_level1 + out_enc_level1
            x_fft = torch.fft.rfft2(x_)
            x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_ = self.kernel_ins_max(x_)
            x_ = self.kernel_blks_max(x_)
            x_ = self.kernel_outs_max(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_ = torch.fft.irfft2(otf * x_fft)
            # print(x_.shape)
            # if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
            #     x_ = window_reversex(x_, self.ker_size[level], H, W, batch_list)
            # else:
            #     x_ = self.grids_inverse(x_, level=level)
            inp_dec_level1 = self.proj_feature_outs_max(torch.cat([inp_dec_level1, x_], dim=1))

            # outputs.append(self.up_decs_max(inp_dec_level1) + inp_img)
        else:
            inp_dec_level1 = inp_dec_level1 + out_enc_level1
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
            out_dec_level1 = window_reversex(out_dec_level1, self.ker_size[level], H, W, batch_list)
        else:
            out_dec_level1 = self.grids_inverse(out_dec_level1, level=level)
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        outputs.append(out_dec_level1)
        # print('xxxxxxxxxxxxxxxxxxx')
        return outputs
#
##
###
##########################################################################
##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192):
        super(PromptGenBlock, self).__init__()
        # self.prompt_param = nn.Parameter(
        #     torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size),
        #     requires_grad=True,
        # )
        self.linear_layer = nn.Conv2d(lin_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3x3 = nn.Conv2d(
            prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x):
        B, C, H, W = x.shape
        emb = x # .mean(dim=(-2, -1))
        prompt = F.softmax(self.linear_layer(emb), dim=1)
        prompt = self.conv3x3(prompt)

        return prompt




# ==========================================
# 1. Parameterized Motion Kernel (参数化核生成)
# ==========================================
class ParameterizedMotionKernel(nn.Module):
    def __init__(self, kernel_size=31, num_points=50, sigma=0.6):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_points = num_points
        self.sigma = sigma
        self.center = kernel_size // 2

        t_base = torch.linspace(-0.5, 0.5, steps=num_points)
        self.register_buffer('t_base', t_base)

        y_grid, x_grid = torch.meshgrid(
            torch.arange(kernel_size),
            torch.arange(kernel_size),
            indexing='ij'
        )
        self.register_buffer('x_grid', x_grid.float())
        self.register_buffer('y_grid', y_grid.float())

    def forward(self, length, angle, curvature):
        """
        支持批量逐像素参数输入
        Args:
            length, angle, curvature: (B, H, W) 形状的参数图
        Returns:
            kernel_norm: (B, H*W, kernel_size, kernel_size) 的像素级模糊核
        """
        B, _, H, W = length.shape
        N = B * H * W

        # 展平以便于并行计算: (N,)
        len_flat = length.view(-1)
        ang_flat = angle.view(-1)
        curv_flat = curvature.view(-1)

        # 1. 轨迹计算
        t = self.t_base.unsqueeze(0) * len_flat.unsqueeze(1)  # (N, num_points)
        x0 = t
        y0 = curv_flat.unsqueeze(1) * (t ** 2)

        # 2. 旋转变换
        theta = ang_flat * (math.pi / 180.0)
        cos_theta = torch.cos(theta).unsqueeze(1)
        sin_theta = torch.sin(theta).unsqueeze(1)

        x_rot = cos_theta * x0 - sin_theta * y0
        y_rot = sin_theta * x0 + cos_theta * y0

        x = x_rot + self.center
        y = y_rot + self.center

        # 3. 高斯渲染
        x = x.view(N, self.num_points, 1, 1)
        y = y.view(N, self.num_points, 1, 1)

        xg = self.x_grid.view(1, 1, self.kernel_size, self.kernel_size)
        yg = self.y_grid.view(1, 1, self.kernel_size, self.kernel_size)

        dist_sq = (xg - x) ** 2 + (yg - y) ** 2
        weights = torch.exp(-dist_sq / (2 * self.sigma ** 2))
        kernel = weights.sum(dim=1)  # (N, K, K)

        # 4. 归一化
        kernel_sum = kernel.sum(dim=(1, 2), keepdim=True) + 1e-8
        kernel_norm = kernel / kernel_sum

        return kernel_norm.view(B, H * W, self.kernel_size, self.kernel_size)


# ==========================================
# 2. Motion Prior Estimation Network (先验估计网络)
# ==========================================
class MotionMemoryBank(nn.Module):
    def __init__(self, num_items=100, feature_dim=64, decay_rate=0.999):
        super().__init__()
        self.num_items = num_items
        self.feature_dim = feature_dim
        self.decay_rate = decay_rate

        m_init = torch.randn(num_items, feature_dim)
        m_init = F.normalize(m_init, p=2, dim=1)
        self.register_buffer('memory_items', m_init)

    def forward(self, z):
        B, C, H, W = z.shape
        z_flat = z.view(B, C, -1).permute(0, 2, 1)

        z_norm = F.normalize(z_flat, p=2, dim=-1)
        m_norm = F.normalize(self.memory_items, p=2, dim=1)

        similarity = torch.matmul(z_norm, m_norm.t())
        weights = F.softmax(similarity, dim=-1)
        z_hat_flat = torch.matmul(weights, m_norm)

        z_hat = z_hat_flat.permute(0, 2, 1).view(B, C, H, W)

        if self.training:
            self._update_memory(z_flat.detach(), similarity.detach())

        return z_hat

    @torch.no_grad()
    def _update_memory(self, z_flat, similarity):
        max_indices = torch.argmax(similarity, dim=-1)
        z_flat_all = z_flat.contiguous().view(-1, self.feature_dim)
        max_indices_all = max_indices.view(-1)

        unique_indices = torch.unique(max_indices_all)
        for idx in unique_indices:
            mask = (max_indices_all == idx)
            z_j_avg = z_flat_all[mask].mean(dim=0)
            updated_m = self.decay_rate * self.memory_items[idx] + (1 - self.decay_rate) * z_j_avg
            self.memory_items[idx] = F.normalize(updated_m, p=2, dim=0)


class MotionPriorEstimationNetwork(nn.Module):
    def __init__(self, in_channels=3, feature_dim=64, num_memory_items=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)
        )
        self.memory_bank = MotionMemoryBank(num_items=num_memory_items, feature_dim=feature_dim)
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 2, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        z = self.encoder(x)
        z_hat = self.memory_bank(z)
        concat_features = torch.cat([z, z_hat], dim=1)
        params = self.decoder(concat_features)
        p_length, p_angle, p_curvature = params.chunk(3, dim=1)
        p_length = F.softplus(p_length) # + 1.
        p_angle = torch.sigmoid(p_angle) * 360.0
        p_curvature = torch.tanh(p_curvature) * 0.1

        return p_length, p_angle, p_curvature


# ==========================================
# 3. Spatially Varying Reblur Module (空间变化重模糊模块)
# ==========================================
class SpatiallyVaryingReblur(nn.Module):
    def __init__(self, kernel_size=19):
        """
        利用生成的像素级核对 GT 图像执行重模糊 (Reblur)
        """
        super().__init__()
        self.kernel_size = kernel_size


    def forward(self, gt_img, kernels):
        """
        Args:
            gt_img: Sharp 图像 (B, C, H, W)
            length, angle, curvature: 估计出的参数 (B, H, W)
        Returns:
            reblurred_img: 重新模糊后的图像 \hat{B} (B, C, H, W)
        """
        B, C, H, W = gt_img.shape
        K = self.kernel_size
        padding = 0 # K // 2
        pd = K // 2 * 2
        # 1. 生成所有像素的局部模糊核
        # kernels shape: (B, H*W, K, K)


        # 2. 提取 GT 图像的局部邻域块
        # 使用 unfold，将每个像素的 KxK 邻域提取出来
        # unfold 后 shape: (B, C * K * K, H * W)
        patches = F.unfold(gt_img, kernel_size=K, padding=padding)

        # reshape 为 (B, C, K*K, H*W) 以便按通道独立相乘
        patches = patches.view(B, C, K * K, -1)

        # 3. 准备卷积核进行逐像素乘积累加
        # kernels reshape -> (B, 1, K*K, H*W)
        # N = H * W
        # print(kernels.shape)
        kernels = kernels[:, :, K // 2:-(K // 2), K // 2:-(K // 2)].contiguous()
        # print(kernels.shape)
        kernels_flat = kernels.view(B, K * K, -1).unsqueeze(1)

        # 4. 执行空间变化卷积: \Omega(x,y) \otimes K(x,y)
        # 对应元素相乘后，沿着局部窗口 K*K 的维度求和
        out_unfold = (patches * kernels_flat).sum(dim=2)  # (B, C, H*W)

        # 5. 还原为图像形状
        reblurred_img = out_unfold.view(B, C, H-pd, W-pd)
        return reblurred_img
class KernelGenerator(nn.Module):
    def __init__(self, in_channels=64, feature_dim=64, num_memory_items=128, kernel_size=19):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)
        )
        self.memory_bank = MotionMemoryBank(num_items=num_memory_items, feature_dim=feature_dim)
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim * 2, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim * 2, kernel_size**2, kernel_size=1, padding=0)
        )

    def forward(self, x):
        z = self.encoder(x)
        z_hat = self.memory_bank(z)
        concat_features = torch.cat([z, z_hat], dim=1)
        params = self.decoder(concat_features)


        return torch.softmax(params, dim=1)
@ARCH_REGISTRY.register()
class uniDeblurX(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[6, 6, 12],
        baseblock=['localnaf', 'localnaf', 'localnaf'],
        ker_baseblock='EVSS',
        ffn_expansion_factor=3,
        bias=False,
        train_size=[256, 256],
        patch_size=[256, 256],
        ker_size=[[256, 256], [128, 128], [64, 64]],
        overlap_size=[[32, 32], [16, 16], [8, 8]],
        k_channels=[48, 96, 192],
        ker_start=1,
        window_size=8,
        global_fft=[False, False, False],
        patch_size_fft=[128, 128],
        kernel_learning=False,
        bin_edges=[20., 30., 40.],
        k=19, scale=4, max_val=1.0

    ):
        super().__init__()

        self.bin_edges = bin_edges
        self.num_classes = len(bin_edges) + 1
        self.k = k
        self.scale = scale
        self.max_val = max_val

        # self.prior_net = MotionPriorEstimationNetwork(dim * 4, dim * 4, dim * 4)
        self.reblur_module = SpatiallyVaryingReblur(kernel_size=k)

        self.criterion = L1Loss()
        self.noise_level3 = nn.Sequential()

        self.reduce_noise_level3 = prompt_attention(k**2, dim * 4, dim * 4)
        self.kernel_generator = KernelGenerator(dim * 4, feature_dim=dim * 4, num_memory_items=dim * 8, kernel_size=k) # ParameterizedMotionKernel(kernel_size=k)
        # self.reduce_noise_level3 = nn.Conv2d(
        #     self.num_classes + dim * 5, dim * 4, kernel_size=1, bias=bias
        # )
        # self.model_degradation_prior = PixelPSNRClassifier(bin_edges=bin_edges,dim=cdim,
        #          num_blocks=cnum_blocks,
        #          baseblock=cbaseblock,
        #          train_size=train_size,
        #          patch_size=patch_size,
        #          k=k, scale=scale, max_val=max_val)
        #
        # model_dict = torch.load(classifier_pth)
        # # print(model_dict.keys())
        # self.model_degradation_prior.load_state_dict(model_dict["params_ema"], strict=True)
        # for p in self.model_degradation_prior.parameters(): p.requires_grad = False

        self.kernel_learning = kernel_learning
        self.ker_start = ker_start
        self.patch_size = ker_size
        self.ker_size = ker_size
        self.out_channels = k_channels
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
        self.encoder = True

        if self.kernel_learning:
            self.kernel_ins = nn.Sequential(
                nn.Conv2d(dim * 4 * 2, dim * 4 * 2, 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks = nn.Sequential()
            for i in range(4):
                block = get_basicblock(ker_baseblock, channels=dim * 4 * 2, num_heads=1, sub_idx=i,
                                       train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
                self.kernel_blks.add_module(f"block{i}", block)

            self.kernel_outs = nn.Sequential(
                LayerNorm(dim * 4 * 2),
                nn.Conv2d(dim * 4 * 2, dim * 4 * 2, 1, bias=bias)
            )
            self.proj_feature_outs = nn.Sequential(
                nn.Conv2d(dim * 4 * 2, dim * 4, 1, bias=bias)
            )

            self.up_decs = nn.Sequential(
                nn.Conv2d(dim * 4, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2),
                nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=bias)
            )

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.encoder_level1.add_module(f"block{i}", block)

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.encoder_level2.add_module(f"block{i}", block)

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.encoder_level3.add_module(f"block{i}", block)
        # self.prompt3 = PromptGenBlock(
        #     prompt_dim=dim * 2, prompt_len=5, prompt_size=16, lin_dim=cdim*4
        # )

        self.decoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.decoder_level3.add_module(f"block{i}", block)

        self.up3_2 = Upsample(int(dim * 2 ** 2))

        self.decoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.decoder_level2.add_module(f"block{i}", block)

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.decoder_level1.add_module(f"block{i}", block)

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
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
    def forward(self, inp_img, gt=None):
        # with torch.no_grad():
        #     latent_degradation = self.model_degradation_prior(inp_img)['feature']
        outputs = []
        out_dict = {}
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)



        if self.kernel_learning:
            level = 2
            b, c, H, W = out_enc_level3.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_, batch_list = window_partitionx(out_enc_level3, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_ = self.grids(out_enc_level3, level=level)
                # print(x_.shape)
            x_fft = torch.fft.rfft2(x_)
            x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_ = self.kernel_ins(x_)
            x_ = self.kernel_blks(x_)
            x_ = self.kernel_outs(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_ = torch.fft.irfft2(otf * x_fft)
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_ = window_reversex(x_, self.ker_size[level], H, W, batch_list)
            else:
                x_ = self.grids_inverse(x_, level=level)
            out_enc_level3 = self.proj_feature_outs(torch.cat([out_enc_level3, x_], dim=1))
            outputs.append(self.up_decs(out_enc_level3) + inp_img)
        # latent = self.class_level3(self.classifier_in(out_enc_level3))
        # logits = self.classifier(latent)  # (B, num_classes, H/4, W/4)
        # out_dict['logits'] = logits
        # 1. 估计模糊核参数
        # p_len, p_ang, p_curv = self.prior_net(out_enc_level3)
        # print(f"Predicted params shape: Length {p_len.shape}, Angle {p_ang.shape}, Curv {p_curv.shape}")
        # kernels = self.kernel_generator(p_len, p_ang, p_curv)
        kernels = self.kernel_generator(out_enc_level3)
        # h, w = out_enc_level3.shape[-2:]
        # kernels = rearrange(kernels, 'b (h w) k1 k2 -> b (k1 k2) h w', h=h, w=w)
        # 2. 执行 Reblur 过程 (将参数应用于 GT 图像)

        if gt is not None and self.training:
            blurred_img = F.interpolate(inp_img, scale_factor=(1./self.scale, 1./self.scale))
            blurred_img = blurred_img[:, :, self.k//2:-(self.k//2), self.k//2:-(self.k//2)]
            gt_ = F.interpolate(gt, scale_factor=(1./self.scale, 1./self.scale))
            reblurred_img = self.reblur_module(gt_, kernels)

            # with torch.no_grad():
            #     psnr_map = psnr_pixelwise_with_shuffle(inp_img, gt, self.k, self.scale, self.max_val)
            #     labels = classify_psnr(psnr_map, self.bin_edges)  # (B, 1, Hd, Wd)
            #     labels = labels.squeeze(1)  # (B, Hd, Wd) 以供 CrossEntropyLoss
            loss = self.criterion(reblurred_img, blurred_img)
            out_dict['loss_model'] = loss # * 0.01
            # out_dict['labels'] = labels
            # 计算准确率（逐像素比较）
            # pred = logits.argmax(dim=1)  # (B, Hd, Wd)
            # correct = (pred == labels).float()
            # acc = correct.view(correct.size(0), -1).mean(dim=1)
            # out_dict['acc'] = acc * 100.
        # dec3_param = self.prompt3(latent_degradation)
        # latent = torch.cat([latent, logits], 1)

        latent = kernels
        latent = self.noise_level3(latent)
        # out_enc_level3 = torch.sigmoid(self.reduce_noise_level3(latent)) * out_enc_level3
        out_enc_level3 = self.reduce_noise_level3(out_enc_level3, latent)
        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = inp_dec_level2 + out_enc_level2

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = inp_dec_level1 + out_enc_level1
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img
        outputs.append(out_dec_level1)
        # print('xxxxxxxxxxxxxxxxxxx')
        out_dict['restored'] = outputs
        return out_dict
# @ARCH_REGISTRY.register()
# class uniDeblurX2(nn.Module):
#     def __init__(
#         self,
#         inp_channels=3,
#         out_channels=3,
#         dim=48,
#         num_blocks=[6, 6, 12],
#         baseblock=['localnaf', 'localnaf', 'localnaf'],
#         ker_baseblock='EVSS',
#         ffn_expansion_factor=3,
#         bias=False,
#         train_size=[256, 256],
#         patch_size=[256, 256],
#         ker_size=[[256, 256], [128, 128], [64, 64]],
#         overlap_size=[[32, 32], [16, 16], [8, 8]],
#         k_channels=[48, 96, 192],
#         ker_start=1,
#         window_size=8,
#         global_fft=[False, False, False],
#         patch_size_fft=[128, 128],
#         kernel_learning=False,
#         bin_edges=[20., 30., 40.],
#         k=19, scale=4, max_val=1.0
#
#     ):
#         super().__init__()
#
#         self.bin_edges = bin_edges
#         self.num_classes = len(bin_edges) + 1
#         self.k = k
#         self.scale = scale
#         self.max_val = max_val
#
#         self.class_level3 = nn.Sequential()
#         self.classifier_in = nn.Conv2d(dim * 4, dim * 1, kernel_size=1)
#         for i in range(4):
#             block = get_basicblock(baseblock[2], channels=dim * 1, num_heads=2, sub_idx=i,
#                                    train_size=[train_size[0] // 4, train_size[1] // 4],
#                                    patch_size=[patch_size[0] // 4, patch_size[1] // 4])
#             self.class_level3.add_module(f"block{i}", block)
#
#         self.classifier = nn.Sequential(nn.Conv2d(dim * 1, dim, kernel_size=1),
#                                         nn.ReLU(),
#                                         nn.Conv2d(dim * 1, self.num_classes, kernel_size=1))
#
#         self.criterion = nn.CrossEntropyLoss()
#         self.noise_level3 = nn.Sequential()
#         for i in range(4):
#
#             block = get_basicblock(baseblock[2], channels=self.num_classes + dim, num_heads=2, sub_idx=i,
#                                train_size=[train_size[0] // 4, train_size[1] // 4],
#                                patch_size=[patch_size[0] // 4, patch_size[1] // 4])
#             self.noise_level3.add_module(f"block{i}", block)
#         self.reduce_noise_level3 = prompt_attention(self.num_classes + dim, dim * 4, dim * 4)
#         # self.reduce_noise_level3 = nn.Conv2d(
#         #     self.num_classes + dim * 5, dim * 4, kernel_size=1, bias=bias
#         # )
#         # self.model_degradation_prior = PixelPSNRClassifier(bin_edges=bin_edges,dim=cdim,
#         #          num_blocks=cnum_blocks,
#         #          baseblock=cbaseblock,
#         #          train_size=train_size,
#         #          patch_size=patch_size,
#         #          k=k, scale=scale, max_val=max_val)
#         #
#         # model_dict = torch.load(classifier_pth)
#         # # print(model_dict.keys())
#         # self.model_degradation_prior.load_state_dict(model_dict["params_ema"], strict=True)
#         # for p in self.model_degradation_prior.parameters(): p.requires_grad = False
#
#         self.kernel_learning = kernel_learning
#         self.ker_start = ker_start
#         self.patch_size = ker_size
#         self.ker_size = ker_size
#         self.out_channels = k_channels
#         self.overlap_size = overlap_size
#         self.up_scale = 1
#         self.overlap_size_up = self.overlap_size  # (self.overlap_size[0] * self.up_scale, self.overlap_size[1] * self.up_scale)
#         # self.kernel_size = [train_size, train_size]
#         self.kernel_size_up = self.ker_size
#         self.nr = [1, 1, 1]
#         self.nc = [1, 1, 1]
#         self.stride = [-1, -1, -1]
#         self.h = [-1, -1, -1]
#         self.w = [-1, -1, -1]
#         self.ek1, self.ek2 = [0, 0, 0], [0, 0, 0]
#         self.num_patchs = [ 1, 1, 1 ]
#         self.original_size = [[], [], []]
#         self.idxes = [None, None, None]
#         self.fuse_matrix_w1 = [None, None, None]
#         self.fuse_matrix_w2 = [None, None, None]
#         self.fuse_matrix_h1 = [None, None, None]
#         self.fuse_matrix_h2 = [None, None, None]
#
#         self.fuse_matrix_eh1 = [None, None, None]
#         self.fuse_matrix_eh2 = [None, None, None]
#
#         self.fuse_matrix_ew1 = [None, None, None]
#         self.fuse_matrix_ew2 = [None, None, None]
#         self.encoder = True
#
#         if self.kernel_learning:
#             self.kernel_ins = nn.Sequential(
#                 nn.Conv2d(dim * 4 * 2, dim * 4 * 2, 1, bias=bias)
#             )
#             # self.kernel_blks = nn.Sequential(
#             #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
#             # )
#             self.kernel_blks = nn.Sequential()
#             for i in range(4):
#                 block = get_basicblock(ker_baseblock, channels=dim * 4 * 2, num_heads=1, sub_idx=i,
#                                        train_size=[train_size[0]//4, train_size[1]//4],
#                                        patch_size=[patch_size[0]//4, patch_size[1]//4])
#                 self.kernel_blks.add_module(f"block{i}", block)
#
#             self.kernel_outs = nn.Sequential(
#                 LayerNorm(dim * 4 * 2),
#                 nn.Conv2d(dim * 4 * 2, dim * 4 * 2, 1, bias=bias)
#             )
#             self.proj_feature_outs = nn.Sequential(
#                 nn.Conv2d(dim * 4 * 2, dim * 4, 1, bias=bias)
#             )
#
#             self.up_decs = nn.Sequential(
#                 nn.Conv2d(dim * 4, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2),
#                 nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
#                 nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
#                           bias=bias)
#             )
#
#         self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
#
#         self.encoder_level1 = nn.Sequential()
#         for i in range(num_blocks[0]):
#             block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
#                         train_size=[train_size[0], train_size[1]],
#                                        patch_size=[patch_size[0], patch_size[1]])
#             self.encoder_level1.add_module(f"block{i}", block)
#
#         self.down1_2 = Downsample(dim)
#         self.encoder_level2 = nn.Sequential()
#         for i in range(num_blocks[1]):
#             block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
#                         train_size=[train_size[0]//2, train_size[1]//2],
#                                        patch_size=[patch_size[0]//2, patch_size[1]//2])
#             self.encoder_level2.add_module(f"block{i}", block)
#
#         self.down2_3 = Downsample(int(dim * 2 ** 1))
#         self.encoder_level3 = nn.Sequential()
#         for i in range(num_blocks[2]):
#             block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
#                         train_size=[train_size[0]//4, train_size[1]//4],
#                                        patch_size=[patch_size[0]//4, patch_size[1]//4])
#             self.encoder_level3.add_module(f"block{i}", block)
#         # self.prompt3 = PromptGenBlock(
#         #     prompt_dim=dim * 2, prompt_len=5, prompt_size=16, lin_dim=cdim*4
#         # )
#
#         self.decoder_level3 = nn.Sequential()
#         for i in range(num_blocks[2]):
#             block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
#                         train_size=[train_size[0]//4, train_size[1]//4],
#                                        patch_size=[patch_size[0]//4, patch_size[1]//4])
#             self.decoder_level3.add_module(f"block{i}", block)
#
#         self.up3_2 = Upsample(int(dim * 2 ** 2))
#
#         self.decoder_level2 = nn.Sequential()
#         for i in range(num_blocks[1]):
#             block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
#                         train_size=[train_size[0]//2, train_size[1]//2],
#                                        patch_size=[patch_size[0]//2, patch_size[1]//2])
#             self.decoder_level2.add_module(f"block{i}", block)
#
#         self.up2_1 = Upsample(int(dim * 2 ** 1))
#
#         self.decoder_level1 = nn.Sequential()
#         for i in range(num_blocks[0]):
#             block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
#                         train_size=[train_size[0], train_size[1]],
#                                        patch_size=[patch_size[0], patch_size[1]])
#             self.decoder_level1.add_module(f"block{i}", block)
#
#         self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
#     def grids(self, x, level=0):
#         b, c, h, w = x.shape
#         n_level = 2 ** level
#
#         # assert b == 1
#         k1, k2 = self.ker_size[level][0] , self.ker_size[level][1]
#         # print(h, w, k1, k2)
#         k1 = min(h, k1)
#         k2 = min(w, k2)
#         # print(k1, k2)
#         # if self.dynic_overlap and n_level == 1:
#         #     num_row = h // self.ker_size[level][0] + 1
#         #     num_col = w // self.ker_size[level][1] + 1
#         #     self.overlap_size[level][0] = (num_row * k1 - h) // (num_row - 1)
#         #     self.overlap_size[level][1] = (num_col * k2 - w) // (num_col - 1)
#
#         overlap_size = [self.overlap_size[level][0], self.overlap_size[level][1]]  # (64, 64)
#
#         stride = (k1 - overlap_size[0], k2 - overlap_size[1])
#
#         num_row = (h - overlap_size[0] - 1) // stride[0] + 1
#         num_col = (w - overlap_size[1] - 1) // stride[1] + 1
#
#         # import math
#         step_j = k2 if num_col == 1 else stride[1]  # math.ceil((w - stride[1]) / (num_col - 1) - 1e-8)
#         step_i = k1 if num_row == 1 else stride[0]  # math.ceil((h - stride[0]) / (num_row - 1) - 1e-8)
#
#         parts = []
#         idxes = []
#         i = 0  # 0~h-1
#         last_i = False
#         # self.ek1, self.ek2 = None, None
#         while i < h and not last_i:
#             j = 0
#             if i + k1 >= h:
#                 # if not self.ek1:
#                 #     # print(step_i, i, k1, h)
#                 #     self.ek1 = i + k1 - h # - self.overlap_size[0]
#                 i = h - k1
#                 last_i = True
#
#             last_j = False
#             while j < w and not last_j:
#                 if j + k2 >= w:
#                     # if not self.ek2:
#                     #     self.ek2 = j + k2 - w # + self.overlap_size[1]
#                     j = w - k2
#                     last_j = True
#                 parts.append(x[:, :, i:i + k1, j:j + k2])
#                 idxes.append({'i': i * self.up_scale, 'j': j * self.up_scale})
#                 j = j + step_j
#             i = i + step_i
#
#         self.num_patchs[level] = len(parts)
#         # print(len(parts))
#         parts = torch.stack(parts, dim=1)
#         # print(parts.shape)
#         parts = parts.view(-1, c, k1, k2)
#         # print(parts.shape)
#         # parts = parts.view(b*self.num_patchs[level], -1, k1, k2)
#         # if level == 0:
#
#         self.original_size[level] = (b, self.out_channels[level], h * self.up_scale , w * self.up_scale)
#         self.stride[level] = (stride[0] * self.up_scale, stride[1] * self.up_scale)
#         self.nr[level] = num_row
#         self.nc[level] = num_col
#         self.idxes[level] = idxes
#         return parts
#
#     def get_overlap_matrix(self, h, w, level=0):
#         # if self.grid:
#         # if self.fuse_matrix_h1 is None:
#         # print( self.ek1,  self.ek2)
#         self.ek1[level] = self.nr[level] * self.stride[level][0] + self.overlap_size[level][0] * 2 - h
#         self.ek2[level] = self.nc[level] * self.stride[level][1] + self.overlap_size[level][1] * 2 - w
#         # print(self.ek1, self.ek2, h, w, self.nr, self.nc, self.stride[level], self.overlap_size[level], h, w)
#         if self.h[level] != h or self.w[level] != w:
#
#             self.fuse_matrix_w1[level] = torch.linspace(1., 0., self.overlap_size[level][1]).view(1, 1, self.overlap_size[level][1]).unsqueeze(0)
#             self.fuse_matrix_w2[level] = torch.linspace(0., 1., self.overlap_size[level][1]).view(1, 1, self.overlap_size[level][1]).unsqueeze(0)
#             self.fuse_matrix_h1[level] = torch.linspace(1., 0., self.overlap_size[level][0]).view(1, self.overlap_size[level][0], 1).unsqueeze(0)
#             self.fuse_matrix_h2[level] = torch.linspace(0., 1., self.overlap_size[level][0]).view(1, self.overlap_size[level][0], 1).unsqueeze(0)
#         if self.h[level] != h:
#             self.fuse_matrix_eh1[level] = torch.linspace(1., 0., self.ek1[level]).view(1, self.ek1[level], 1).unsqueeze(0)
#             self.fuse_matrix_eh2[level] = torch.linspace(0., 1., self.ek1[level]).view(1, self.ek1[level], 1).unsqueeze(0)
#         if self.w[level] != w:
#             self.fuse_matrix_ew1[level] = torch.linspace(1., 0., self.ek2[level]).view(1, 1, self.ek2[level]).unsqueeze(0)
#             self.fuse_matrix_ew2[level] = torch.linspace(0., 1., self.ek2[level]).view(1, 1, self.ek2[level]).unsqueeze(0)
#         self.h[level] = h
#         self.w[level] = w
#
#     def grids_inverse(self, outs, level=0, out_device=None, pix_max=1., size=None):
#         if out_device is None:
#             out_device = outs.device
#         # print(out_device)
#         # print(outs.shape)
#         type_out = torch.uint8 if pix_max == 255. else torch.float32
#         # type_out = torch.int16
#         # print(self.original_size)
#         if size is None:
#             preds = torch.zeros(self.original_size[level], device=out_device, dtype=type_out)  # .to(out_device)
#             b, c, h, w = self.original_size[level]
#         else:
#             preds = torch.zeros(size, device=out_device, dtype=type_out)  # .to(out_device)
#             b, c, h, w = size
#         # print(self.original_size)
#         # outs = torch.clamp(outs, 0, 1) # * pix_max
#         # outs = outs.type(type_out)
#         # count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
#         k1, k2 = self.kernel_size_up[level]
#         k1 = min(h, k1)
#         k2 = min(w, k2)
#         # print(outs.shape, self.num_patchs[level])
#         # print(b, c, h, w)
#         # print(outs.shape)
#         outs = outs.clone().view(b, self.num_patchs[level], -1, k1, k2)
#         # if not self.h or not self.w:
#         self.get_overlap_matrix(h, w, level)
#         # rounding_mode = 'floor'
#         # print(self.idxes)
#         for cnt, each_idx in enumerate(self.idxes[level]):
#             i = each_idx['i']
#             j = each_idx['j']
#             if self.overlap_size_up[level][0] > 0:
#                 if i != 0 and i + k1 != h:
#                     # print(1)
#                     outs[:, cnt, :, :self.overlap_size_up[level][0], :] = torch.mul(outs[:, cnt, :, :self.overlap_size_up[level][0], :],
#                                                                           self.fuse_matrix_h2[level].to(outs.device))
#                     # print(outs[cnt, :,  i + k1 - self.overlap_size[0]:i + k1, :].shape,
#                     #       self.fuse_matrix_h1.shape)
#                 if i + k1 * 2 - self.ek1[level] < h:
#                     # print(2)
#                     # print(self.fuse_matrix_h1.shape, outs[:, cnt, :, -self.overlap_size_up[level][0]:, :].shape)
#                     # print(torch.mul(outs[:, cnt, :, -self.overlap_size_up[level][0]:, :],
#                     #                                                        self.fuse_matrix_h1.to(outs.device)).shape)
#                     outs[:, cnt, :, -self.overlap_size_up[level][0]:, :] = torch.mul(outs[:, cnt, :, -self.overlap_size_up[level][0]:, :],
#                                                                            self.fuse_matrix_h1[level].to(outs.device))
#             # print(self.fuse_matrix_eh1.dtype)
#             if self.ek1[level] > 0:
#                 if i + k1 == h and i !=0:
#                     # print(3)
#                     outs[:, cnt, :, :self.ek1[level], :] = torch.mul(outs[:, cnt, :, :self.ek1[level], :], self.fuse_matrix_eh2[level].to(outs.device))
#                 if i + k1 * 2 - self.ek1[level] == h:
#                     # print(4)
#                     outs[:, cnt, :, -self.ek1[level]:, :] = torch.mul(outs[:, cnt, :, -self.ek1[level]:, :],
#                                                             self.fuse_matrix_eh1[level].to(outs.device))
#
#             if self.overlap_size_up[level][1] > 0:
#                 if j != 0 and j + k2 != w:
#                     # print(1)
#                     outs[:, cnt, :, :, :self.overlap_size_up[level][1]] = torch.mul(outs[:, cnt, :, :, :self.overlap_size_up[level][1]],
#                                                                           self.fuse_matrix_w2[level].to(outs.device))
#                 if j + k2 * 2 - self.ek2[level] < w:
#                     # print(2)
#                     # print(j, j + k2 - self.overlap_size[1], j + k2, self.fuse_matrix_w1.shape)
#                     # print(self.fuse_matrix_w1.shape)
#                     outs[:, cnt, :, :, -self.overlap_size_up[level][1]:] = torch.mul(outs[:, cnt, :, :, -self.overlap_size_up[level][1]:],
#                                                                            self.fuse_matrix_w1[level].to(outs.device))
#             if self.ek2[level] > 0:
#                 if j + k2 == w and j != 0:
#                     # print(3)
#                     # print('j + k2 == w: ', self.ek2, outs[cnt, :, :, :self.ek2].shape, self.fuse_matrix_ew1.shape)
#                     outs[:, cnt, :, :, :self.ek2[level]] = torch.mul(outs[:, cnt, :, :, :self.ek2[level]], self.fuse_matrix_ew2[level].to(outs.device))
#                 if j + k2 * 2 - self.ek2[level] == w:
#                     # print(4)
#                     # print('j + k2*2 - self.ek2 == w: ')
#                     # print(self.ek2[level], self.fuse_matrix_ew1[level].shape,outs[:, cnt, :, :, -self.ek2[level]:].shape )
#                     outs[:, cnt, :, :, -self.ek2[level]:] = torch.mul(outs[:, cnt, :, :, -self.ek2[level]:],
#                                                             self.fuse_matrix_ew1[level].to(outs.device))
#             # print(i, j)
#             # print(preds[0, :, i:i + k1, j:j + k2].shape)
#             # pred_win = outs[cnt, :, :, :].clone() * pix_max
#             # pred_win = pred_win.type(type_out)
#             # print(pred_win.dtype, preds.dtype, type_out, outs.dtype)
#             # print(outs[cnt, :, :, :].max())
#             # print(preds[:, :, i:i + k1, j:j + k2].shape, j, outs.shape)
#             # print(preds.shape, outs.shape)
#             preds[:, :, i:i + k1, j:j + k2] += outs[:, cnt, :, :, :].type(type_out).to(out_device)
#             # preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :].to(out_device) * pix_max
#             # count_mt[0, 0, i:i + k1, j:j + k2] += 1.
#
#         del outs
#         torch.cuda.empty_cache()
#         return preds  # / count_mt
#     def forward(self, inp_img, gt=None):
#         # with torch.no_grad():
#         #     latent_degradation = self.model_degradation_prior(inp_img)['feature']
#         outputs = []
#         out_dict = {}
#         inp_enc_level1 = self.patch_embed(inp_img)
#         out_enc_level1 = self.encoder_level1(inp_enc_level1)
#
#         inp_enc_level2 = self.down1_2(out_enc_level1)
#
#         out_enc_level2 = self.encoder_level2(inp_enc_level2)
#
#         inp_enc_level3 = self.down2_3(out_enc_level2)
#
#         out_enc_level3 = self.encoder_level3(inp_enc_level3)
#
#
#
#         if self.kernel_learning:
#             level = 2
#             b, c, H, W = out_enc_level3.shape
#             if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
#                 x_, batch_list = window_partitionx(out_enc_level3, self.ker_size[level])
#             else:
#                 # print('11111111111111111')
#                 # print('level: ', level)
#                 x_ = self.grids(out_enc_level3, level=level)
#                 # print(x_.shape)
#             x_fft = torch.fft.rfft2(x_)
#             x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
#             x_ = self.kernel_ins(x_)
#             x_ = self.kernel_blks(x_)
#             x_ = self.kernel_outs(x_)
#             x_real, x_imag = torch.chunk(x_, 2, dim=1)
#             otf = torch.complex(x_real, x_imag)
#             # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
#             # print(x_fft.shape, otf.shape)
#             x_ = torch.fft.irfft2(otf * x_fft)
#             if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
#                 x_ = window_reversex(x_, self.ker_size[level], H, W, batch_list)
#             else:
#                 x_ = self.grids_inverse(x_, level=level)
#             out_enc_level3 = self.proj_feature_outs(torch.cat([out_enc_level3, x_], dim=1))
#             outputs.append(self.up_decs(out_enc_level3) + inp_img)
#         latent = self.class_level3(self.classifier_in(out_enc_level3))
#         logits = self.classifier(latent)  # (B, num_classes, H/4, W/4)
#         out_dict['logits'] = logits
#
#         if gt is not None:
#             with torch.no_grad():
#                 psnr_map = psnr_pixelwise_with_shuffle(inp_img, gt, self.k, self.scale, self.max_val)
#                 labels = classify_psnr(psnr_map, self.bin_edges)  # (B, 1, Hd, Wd)
#                 labels = labels.squeeze(1)  # (B, Hd, Wd) 以供 CrossEntropyLoss
#             loss = self.criterion(logits, labels)
#             out_dict['loss_model'] = loss * 0.01
#             out_dict['labels'] = labels
#             # 计算准确率（逐像素比较）
#             # pred = logits.argmax(dim=1)  # (B, Hd, Wd)
#             # correct = (pred == labels).float()
#             # acc = correct.view(correct.size(0), -1).mean(dim=1)
#             # out_dict['acc'] = acc * 100.
#         # dec3_param = self.prompt3(latent_degradation)
#         latent = torch.cat([latent, logits], 1)
#         latent = self.noise_level3(latent)
#         # out_enc_level3 = torch.sigmoid(self.reduce_noise_level3(latent)) * out_enc_level3
#         out_enc_level3 = self.reduce_noise_level3(out_enc_level3, latent)
#         out_dec_level3 = self.decoder_level3(out_enc_level3)
#
#         inp_dec_level2 = self.up3_2(out_dec_level3)
#
#         inp_dec_level2 = inp_dec_level2 + out_enc_level2
#
#         out_dec_level2 = self.decoder_level2(inp_dec_level2)
#
#         inp_dec_level1 = self.up2_1(out_dec_level2)
#
#         inp_dec_level1 = inp_dec_level1 + out_enc_level1
#         out_dec_level1 = self.decoder_level1(inp_dec_level1)
#
#         out_dec_level1 = self.output(out_dec_level1) + inp_img
#         outputs.append(out_dec_level1)
#         # print('xxxxxxxxxxxxxxxxxxx')
#         out_dict['restored'] = outputs
#         return out_dict
@ARCH_REGISTRY.register()
class uniDeblurX1(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[6, 6, 12],
        baseblock=['localnaf', 'localnaf', 'localnaf'],
        ker_baseblock='EVSS',
        ffn_expansion_factor=3,
        bias=False,
        train_size=[256, 256],
        patch_size=[256, 256],
        ker_size=[[256, 256], [128, 128], [64, 64]],
        overlap_size=[[32, 32], [16, 16], [8, 8]],
        k_channels=[48, 96, 192],
        ker_start=1,
        window_size=8,
        global_fft=[False, False, False],
        patch_size_fft=[128, 128],
        kernel_learning=False,
            cdim=32,
        bin_edges=[20., 30., 40.], cnum_blocks=[4, 4, 8],
                 cbaseblock=['localnaf', 'localnaf', 'localnaf'],
                 k=19, scale=4, max_val=1.0, classifier_pth=''

    ):
        super().__init__()



        self.model_degradation_prior = PixelPSNRClassifier(bin_edges=bin_edges,dim=cdim,
                 num_blocks=cnum_blocks,
                 baseblock=cbaseblock,
                 train_size=train_size,
                 patch_size=patch_size,
                 k=k, scale=scale, max_val=max_val)

        model_dict = torch.load(classifier_pth)
        # print(model_dict.keys())
        self.model_degradation_prior.load_state_dict(model_dict["params_ema"], strict=True)
        for p in self.model_degradation_prior.parameters(): p.requires_grad = False

        self.kernel_learning = kernel_learning
        self.ker_start = ker_start
        self.patch_size = ker_size
        self.ker_size = ker_size
        self.out_channels = k_channels
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
        self.encoder = True

        if self.kernel_learning:
            self.kernel_ins = nn.Sequential(
                nn.Conv2d(dim * 4 * 2, dim * 4 * 2, 1, bias=bias)
            )
            # self.kernel_blks = nn.Sequential(
            #     *[SS2D(d_model=dim * 4 * 2, patch_size=None) for _ in range(4)]
            # )
            self.kernel_blks = nn.Sequential()
            for i in range(4):
                block = get_basicblock(ker_baseblock, channels=dim * 4 * 2, num_heads=1, sub_idx=i,
                                       train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
                self.kernel_blks.add_module(f"block{i}", block)

            self.kernel_outs = nn.Sequential(
                LayerNorm(dim * 4 * 2),
                nn.Conv2d(dim * 4 * 2, dim * 4 * 2, 1, bias=bias)
            )
            self.proj_feature_outs = nn.Sequential(
                nn.Conv2d(dim * 4 * 2, dim * 4, 1, bias=bias)
            )

            self.up_decs = nn.Sequential(
                nn.Conv2d(dim * 4, dim * 4 * 2, 1, bias=bias), nn.PixelShuffle(2),
                nn.Conv2d(dim * 2, dim * 2 * 2, 1, bias=bias), nn.PixelShuffle(2),
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=bias)
            )

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.encoder_level1.add_module(f"block{i}", block)

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.encoder_level2.add_module(f"block{i}", block)

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.encoder_level3.add_module(f"block{i}", block)
        self.prompt3 = PromptGenBlock(
            prompt_dim=dim * 2, prompt_len=5, prompt_size=16, lin_dim=cdim*4
        )
        self.noise_level3 = nn.Sequential()
        for i in range(1):

            block = get_basicblock(baseblock[2], channels=dim * 6, num_heads=2, sub_idx=i,
                               train_size=[train_size[0] // 4, train_size[1] // 4],
                               patch_size=[patch_size[0] // 4, patch_size[1] // 4])
            self.noise_level3.add_module(f"block{i}", block)
        self.reduce_noise_level3 = nn.Conv2d(
            dim * 6, dim * 4, kernel_size=1, bias=bias
        )
        self.decoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                        train_size=[train_size[0]//4, train_size[1]//4],
                                       patch_size=[patch_size[0]//4, patch_size[1]//4])
            self.decoder_level3.add_module(f"block{i}", block)

        self.up3_2 = Upsample(int(dim * 2 ** 2))

        self.decoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                        train_size=[train_size[0]//2, train_size[1]//2],
                                       patch_size=[patch_size[0]//2, patch_size[1]//2])
            self.decoder_level2.add_module(f"block{i}", block)

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                        train_size=[train_size[0], train_size[1]],
                                       patch_size=[patch_size[0], patch_size[1]])
            self.decoder_level1.add_module(f"block{i}", block)

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
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
    def forward(self, inp_img):
        with torch.no_grad():
            latent_degradation = self.model_degradation_prior(inp_img)['feature']
        outputs = []
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)


        dec3_param = self.prompt3(latent_degradation)
        latent = torch.cat([out_enc_level3, dec3_param], 1)
        latent = self.noise_level3(latent)
        out_enc_level3 = self.reduce_noise_level3(latent)

        if self.kernel_learning:
            level = 2
            b, c, H, W = out_enc_level3.shape
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_, batch_list = window_partitionx(out_enc_level3, self.ker_size[level])
            else:
                # print('11111111111111111')
                # print('level: ', level)
                x_ = self.grids(out_enc_level3, level=level)
                # print(x_.shape)
            x_fft = torch.fft.rfft2(x_)
            x_ = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_ = self.kernel_ins(x_)
            x_ = self.kernel_blks(x_)
            x_ = self.kernel_outs(x_)
            x_real, x_imag = torch.chunk(x_, 2, dim=1)
            otf = torch.complex(x_real, x_imag)
            # x = torch.fft.irfft2(otf * torch.fft.rfft2(x_blur))
            # print(x_fft.shape, otf.shape)
            x_ = torch.fft.irfft2(otf * x_fft)
            if self.training or (self.overlap_size[level][0] == 0 and self.overlap_size[level][1] == 0):
                x_ = window_reversex(x_, self.ker_size[level], H, W, batch_list)
            else:
                x_ = self.grids_inverse(x_, level=level)
            x = self.proj_feature_outs(torch.cat([out_enc_level3, x_], dim=1))
            outputs.append(self.up_decs(x) + inp_img)


        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = inp_dec_level2 + out_enc_level2

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = inp_dec_level1 + out_enc_level1
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img
        outputs.append(out_dec_level1)
        # print('xxxxxxxxxxxxxxxxxxx')
        return outputs
# ===================== 1. 像素级 PSNR 计算 =====================
def psnr_pixelwise_with_shuffle(blur, gt, k, scale=4, max_val=1.0):
    """返回 (B, 1, H/scale, W/scale) 的 PSNR 图"""
    if blur.dim() == 3:
        blur = blur.unsqueeze(0)
        gt = gt.unsqueeze(0)
    B, C, H, W = blur.shape

    blur_down = F.pixel_unshuffle(blur, scale)
    gt_down   = F.pixel_unshuffle(gt, scale)
    _, C_down, Hd, Wd = blur_down.shape

    diff_sq = (blur_down - gt_down) ** 2
    weight = torch.ones(1, C_down, k, k, device=diff_sq.device, dtype=diff_sq.dtype)
    diff_sq = F.pad(diff_sq, pad=(k//2, k//2, k//2, k//2), mode='reflect')
    mse = F.conv2d(diff_sq, weight, stride=1) / (C_down * k * k)
    eps = 1e-10
    psnr_map = 10 * torch.log10((max_val ** 2) / (mse + eps))
    # print(psnr_map.shape, psnr_map[:, :, H//8, W//8])
    return psnr_map

# ===================== 2. 基于边界列表的分类标签生成 =====================
def classify_psnr(psnr_map, bin_edges):
    """
    根据 bin_edges（升序列表）将 PSNR 值映射为类别：
        - 值 < bin_edges[0]                  → 类别 0
        - bin_edges[i] ≤ 值 < bin_edges[i+1] → 类别 i+1
        - 值 ≥ bin_edges[-1]                 → 类别 len(bin_edges)
    返回类别索引图 (B, 1, H, W)
    """
    # torch.bucketize 默认 right=False，正好满足左闭右开区间
    # 返回值范围 0 ~ len(bin_edges)
    labels = torch.bucketize(psnr_map, torch.tensor(bin_edges, device=psnr_map.device), right=False)
    return labels  # (B, 1, H, W)

# ===================== 3. 分类网络（含损失计算） =====================
@ARCH_REGISTRY.register()
class PixelPSNRClassifier(nn.Module):
    def __init__(self, inp_channels=3, bin_edges=None, dim=16,
                 num_blocks=[4, 4, 8],
                 baseblock=['localnaf', 'localnaf', 'localnaf'],
                 train_size=[256, 256],
                 patch_size=[256, 256],
                 k=19, scale=4, max_val=1.0):
        """
        参数：
            bin_edges : 升序列表，例如 [20., 25., 30., 35., 40.]
                        类别数自动设为 len(bin_edges) + 1
            in_channels : 输入图像通道数
            base_channels : 特征图基础通道数
            k, scale, max_val : PSNR 计算参数
        """
        super().__init__()
        if bin_edges is None:
            raise ValueError("必须提供 bin_edges 列表")
        self.bin_edges = bin_edges
        self.num_classes = len(bin_edges) + 1
        self.k = k
        self.scale = scale
        self.max_val = max_val

        # 网络结构（两次下采样，输出尺寸与 PSNR 图一致）
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = get_basicblock(baseblock[0], channels=dim, num_heads=1, sub_idx=i,
                                   train_size=[train_size[0], train_size[1]],
                                   patch_size=[patch_size[0], patch_size[1]])
            self.encoder_level1.add_module(f"block{i}", block)

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = get_basicblock(baseblock[1], channels=dim * 2, num_heads=2, sub_idx=i,
                                   train_size=[train_size[0] // 2, train_size[1] // 2],
                                   patch_size=[patch_size[0] // 2, patch_size[1] // 2])
            self.encoder_level2.add_module(f"block{i}", block)

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = get_basicblock(baseblock[2], channels=dim * 4, num_heads=4, sub_idx=i,
                                   train_size=[train_size[0] // 4, train_size[1] // 4],
                                   patch_size=[patch_size[0] // 4, patch_size[1] // 4])
            self.encoder_level3.add_module(f"block{i}", block)
        self.classifier = nn.Conv2d(dim*4, self.num_classes, kernel_size=1)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, blur, gt=None):
        """
        训练时传入 blur, gt，返回 (loss, logits, labels)
        推理时只传入 blur，返回 logits
        """

        out_dict = {}
        # ----- 特征提取 -----
        inp_enc_level1 = self.patch_embed(blur)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        out_dict['feature'] = out_enc_level3
        logits = self.classifier(out_enc_level3)          # (B, num_classes, H/4, W/4)
        out_dict['logits'] = logits

        if gt is not None:

            with torch.no_grad():
                psnr_map = psnr_pixelwise_with_shuffle(blur, gt, self.k, self.scale, self.max_val)
                labels = classify_psnr(psnr_map, self.bin_edges)  # (B, 1, Hd, Wd)
                labels = labels.squeeze(1)    # (B, Hd, Wd) 以供 CrossEntropyLoss
            loss = self.criterion(logits, labels)
            out_dict['loss_model'] = loss
            out_dict['labels'] = labels
            # 计算准确率（逐像素比较）
            pred = logits.argmax(dim=1)  # (B, Hd, Wd)
            correct = (pred == labels).float()
            acc = correct.view(correct.size(0), -1).mean(dim=1)
            out_dict['acc'] = acc * 100.
            return out_dict
        else:
            return out_dict
