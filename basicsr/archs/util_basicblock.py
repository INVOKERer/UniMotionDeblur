import kornia
import torch
from torch.nn import init as init
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
import numbers
from .util_win import *
from .BioIR_arch import AttBlock as BioBlock
from einops.layers.torch import Rearrange, Reduce
def get_basicblock(baseblock, channels, num_heads, train_size, patch_size=None, kernel_size=19, dp_rate=0., sub_idx=0):
    spatial_size = train_size # [train_size[0]//(2**level), train_size[1]//(2**level)]
    # print(spatial_size)
    # print(train_size)
    # print(dp_rate)
    # print(layers)
    i = sub_idx
    if baseblock == 'naf':
        modules = NAFBlock(channels)
    elif baseblock == 'naf_patch8':
        modules = NAFBlock(channels, patch_size=[8, 8])
    elif baseblock == 'naf_patch':
        modules = NAFBlock(channels, patch_size=patch_size)
    elif baseblock == 'localnaf':
        modules = NAFBlock(channels, local_avgpool=True, train_size=train_size)
    elif baseblock == 'localnaf_FourierComplex':
        modules = FourierNAFBlock(channels, local_avgpool=True, train_size=train_size, patch_size=None, DW_Fourier=1, window_size=None, fft_mode='complex')
    elif baseblock == 'naf_FourierComplex':
        modules = FourierNAFBlock(channels, local_avgpool=False, train_size=train_size, patch_size=None, DW_Fourier=1,
                                  window_size=None, fft_mode='complex')

    elif baseblock == 'localnaf_FourierChannel':
        modules = FourierNAFBlock(channels, local_avgpool=True, train_size=train_size, patch_size=None, DW_Fourier=1, window_size=None, fft_mode='channel')
    elif baseblock == 'localnaf_FourierBatch':
        modules = FourierNAFBlock(channels, local_avgpool=True, train_size=train_size, patch_size=None, DW_Fourier=1, window_size=None, fft_mode='batch')
    elif baseblock == 'localnaf_FourierBatch_patch':
        modules = FourierNAFBlock(channels, local_avgpool=True, train_size=train_size, patch_size=patch_size, DW_Fourier=1, window_size=patch_size, fft_mode='batch')
    elif baseblock == 'localnaf_FourierSplit':
        modules = FourierNAFBlock(channels, local_avgpool=True, train_size=train_size, patch_size=None, DW_Fourier=1, window_size=None, fft_mode='channel_split')
    elif baseblock == 'naf_FourierBatch':
        modules = FourierNAFBlock(channels, local_avgpool=False, train_size=train_size, patch_size=None, DW_Fourier=1, window_size=None, fft_mode='batch')
    elif baseblock == 'EVSBlock':
        # if level > 0:
        #     for i in range(level):
        #         sub_idx = sub_idx + layers[i]
        modules = EVSBlock(channels, ffn_expansion_factor=3, idx=i, att=True, patch_size=patch_size)
    elif baseblock == 'EVSBlock_noflip':
        modules = EVSBlock(channels, ffn_expansion_factor=3, idx=-1, att=True, patch_size=patch_size)
    elif baseblock == 'NAFEVSBlock':
        modules = NAFEVSBlock(channels, train_size=train_size, idx=i)
    elif baseblock == 'NAFEVSBlock_noflip':
        modules = NAFEVSBlock(channels, train_size=train_size, idx=-1)
    elif baseblock == 'LocalNAFEVSBlock':
        modules = NAFEVSBlock(channels, train_size=train_size, local_avgpool=True, idx=i)
    elif baseblock == 'LocalNAFEVSBlock_noflip':
        modules = NAFEVSBlock(channels, train_size=train_size, local_avgpool=True, idx=-1)
    elif baseblock == 'NAFEVSBlock_patch':
        modules = NAFEVSBlock(channels, train_size=train_size, window_size=patch_size, idx=i)
    elif baseblock == 'LocalNAFEVSBlock_patch':
        modules = NAFEVSBlock(channels, train_size=train_size, window_size=patch_size, local_avgpool=True, idx=i)
    elif baseblock == 'LocalNAFEVSBlock_patch_noflip':
        modules = NAFEVSBlock(channels, train_size=train_size, local_avgpool=True, window_size=patch_size, idx=-1)
    elif baseblock == 'EVSS_LN1':
        modules = EVSS_LN1(channels, patch_size=None)
    elif baseblock == 'EVSS':
        modules = EVSS(channels, patch_size=None)
    elif baseblock == 'EVSS_noskip':
        modules = EVSS_noskip(channels, patch_size=None)

    elif baseblock == 'SS2D':
        modules = SS2D(channels, patch_size=None)
    elif baseblock == 'SS2D_Freq':
        modules = SS2D_Freq(channels, patch_size=None)
    elif baseblock == 'ResSS2D_Freq':
        modules = ResSS2D_Freq(channels, patch_size=None)
    elif baseblock == 'ResSS2D':
        modules = ResSS2D(channels, patch_size=None)
    elif baseblock == 'ResSS2D4ES':
        modules = ResSS2D4ES(channels, patch_size=None)
    elif baseblock == 'ResMultiSS2D_Freq':
        modules = ResMultiSS2D_Freq(channels, patch_size=None)
    elif baseblock == 'BioBlock':
        modules = BioBlock(channels, ffn_expansion_factor=3)
    elif baseblock == 'BioBlock2.66':
        modules = BioBlock(channels)
    elif baseblock == 'LNResBlock':
        modules = LNResBlock(channels)
    elif baseblock == 'LNConvBlock':
        modules = LNConvBlock(channels)
    elif baseblock == 'ConvBlock':
        modules = ConvBlock(channels)
    elif baseblock == 'ConvBlock1x1':
        modules = ConvBlock1x1(channels)
    elif baseblock == 'ReLU':
        modules = nn.ReLU()
    elif baseblock == 'GELU':
        modules = nn.GELU()
    elif baseblock == 'ResMLPBlock':
        modules = ResMLPBlock(channels, [train_size[0], train_size[1]//2+1])
    elif baseblock == 'ResMLPMixBlock':
        modules = ResMLPMixBlock(channels, [train_size[0], train_size[1]//2+1], patch_size=[16, 16])
    elif baseblock == 'ResGateBlock':
        modules = ResGateBlock(channels, False)
    elif baseblock == 'ResGateSCABlock':
        modules = ResGateBlock(channels, True)
    elif baseblock == 'ResDepthConvBlock':
        modules = ResDepthConvBlock(channels, False)
    elif baseblock == 'ResDepthConvSCABlock':
        modules = ResDepthConvBlock(channels, True)
    elif baseblock == 'ResBlock':
        modules = ResBlock(channels)
    elif baseblock == 'LNResBlock1x1':
        modules = LNResBlock1x1(channels)
    elif baseblock == 'ResFFTReLUBlock':
        modules = ResFFTReLUBlock(channels)
    elif baseblock == 'LayerNorm2d':
        modules = LayerNorm2d(channels)
    elif baseblock == 'LayerNorm':
        modules = LayerNorm(channels, 'WithBias')
    elif baseblock == 'PreNorm2d':
        modules = PreNorm2d(channels)
    else:
        modules = nn.Identity()
    return modules

def get_modules(baseblock, level, channels, layers, num_heads, train_size, patch_size=None, kernel_size=19, dp_rate=0., sub_idx=0):
    spatial_size = train_size # [train_size[0]//(2**level), train_size[1]//(2**level)]
    # print(spatial_size)
    # print(train_size)
    # print(dp_rate)
    # print(layers)
    if baseblock == 'naf':
        modules = [NAFBlock(channels[level]) for i in range(layers[level])]
    elif baseblock == 'naf_patch8':
        modules = [NAFBlock(channels[level], patch_size=[8, 8]) for i in range(layers[level])]
    elif baseblock == 'naf_patch':
        modules = [NAFBlock(channels[level], patch_size=[8, 8]) for i in range(layers[level])]
    elif baseblock == 'BioBlock':
        modules = [BioBlock(channels[level]) for i in range(layers[level])]
    elif baseblock == 'localnaf':
        modules = [NAFBlock(channels[level], local_avgpool=True, train_size=train_size) for i in range(layers[level])]
    elif baseblock == 'localnaf_FourierComplex':
        modules = [FourierNAFBlock(channels[level], local_avgpool=True, train_size=train_size, patch_size=None, DW_Fourier=1, window_size=None, fft_mode='complex') for i in range(layers[level])]
    elif baseblock == 'localnaf_FourierChannel':
        modules = [FourierNAFBlock(channels[level], local_avgpool=True, train_size=train_size, patch_size=None, DW_Fourier=1, window_size=None, fft_mode='channel') for i in range(layers[level])]
    elif baseblock == 'localnaf_FourierBatch':
        modules = [FourierNAFBlock(channels[level], local_avgpool=True, train_size=train_size, patch_size=None, DW_Fourier=1, window_size=None, fft_mode='batch') for i in range(layers[level])]
    elif baseblock == 'localnaf_FourierBatch_patch':
        modules = [FourierNAFBlock(channels[level], local_avgpool=True, train_size=train_size, patch_size=patch_size, DW_Fourier=1, window_size=patch_size, fft_mode='batch') for i in range(layers[level])]
    elif baseblock == 'localnaf_FourierSplit':
        modules = [
            FourierNAFBlock(channels[level], local_avgpool=True, train_size=train_size, patch_size=None, DW_Fourier=1, window_size=None, fft_mode='channel_split') for
            i in range(layers[level])]
    elif baseblock == 'naf_FourierBatch':
        modules = [FourierNAFBlock(channels[level], local_avgpool=False, train_size=train_size, patch_size=None, DW_Fourier=1, window_size=None, fft_mode='batch') for i in range(layers[level])]

    elif baseblock == 'EVSBlock':
        # if level > 0:
        #     for i in range(level):
        #         sub_idx = sub_idx + layers[i]
        modules = [EVSBlock(channels[level], ffn_expansion_factor=3, idx=i, att=True, patch_size=patch_size) for i in range(layers[level])]
    elif baseblock == 'EVSBlock_noflip':
        modules = [EVSBlock(channels[level], ffn_expansion_factor=3, idx=-1, att=True, patch_size=patch_size) for i in range(layers[level])]
    elif baseblock == 'NAFEVSBlock':
        modules = [NAFEVSBlock(channels[level], train_size=train_size, idx=i) for i in range(layers[level])]

    elif baseblock == 'NAFEVSBlock_noflip':
        modules = [NAFEVSBlock(channels[level], train_size=train_size, idx=-1) for i in range(layers[level])]
    elif baseblock == 'LocalNAFEVSBlock':
        modules = [NAFEVSBlock(channels[level], train_size=train_size, local_avgpool=True, idx=i) for i in range(layers[level])]

    elif baseblock == 'LocalNAFEVSBlock_noflip':
        modules = [NAFEVSBlock(channels[level], train_size=train_size, local_avgpool=True, idx=-1) for i in range(layers[level])]
    elif baseblock == 'NAFEVSBlock_patch':
        modules = [NAFEVSBlock(channels[level], train_size=train_size, window_size=patch_size, idx=i) for i in
                   range(layers[level])]
    elif baseblock == 'LocalNAFEVSBlock_patch':
        modules = [NAFEVSBlock(channels[level], train_size=train_size, window_size=patch_size, local_avgpool=True, idx=i) for i in range(layers[level])]
    elif baseblock == 'LocalNAFEVSBlock_patch_noflip':
        modules = [NAFEVSBlock(channels[level], train_size=train_size, local_avgpool=True, window_size=patch_size, idx=-1) for i in range(layers[level])]
    elif baseblock == 'EVSS':
        modules = [EVSS(channels[level]) for i in range(layers[level])]
    elif baseblock == 'EVSS_noskip':
        modules = [EVSS_noskip(channels[level]) for i in range(layers[level])]
    elif baseblock == 'LNResBlock':
        modules = [LNResBlock(channels[level]) for i in range(layers[level])]
    elif baseblock == 'LNResBlock1x1':
        modules = [LNResBlock1x1(channels[level]) for i in range(layers[level])]
    elif baseblock == 'ResBlock':
        modules = [ResBlock(channels[level]) for i in range(layers[level])]
    else:
        modules = [nn.Identity()]
    return modules


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

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
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
class PreNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(PreNorm2d, self).__init__()

        self.pre_block = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True))
        self.eps = eps

    def forward(self, x):
        return x / (torch.abs(self.pre_block(x)) + self.eps)
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-6) * self.weight


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
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-6) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
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


class LocalInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=False, train_size=None):
        super().__init__()
        assert not track_running_stats
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.avgpool = AvgPool2d(train_size=train_size)
        self.eps = eps

    def forward(self, input):
        mean_x = self.avgpool(input)  # E(x)
        mean_xx = self.avgpool(torch.mul(input, input))  # E(x^2)
        mean_x2 = torch.mul(mean_x, mean_x)  # (E(x))^2
        var_x = mean_xx - mean_x2  # Var(x) = E(x^2) - (E(x))^2
        mean = mean_x
        var = var_x
        input = (input - mean) / (torch.sqrt(var + self.eps))
        if self.affine:
            input = input * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return input
class SimpleGate(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=self.dim)
        return x1 * x2
class LNResBlock(nn.Module):
    def __init__(self, c, drop_out_rate=0.):
        super().__init__()
        # dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)

        self.act = nn.ReLU()
        # SimpleGate

        self.norm1 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.register_parameter('beta', nn.Parameter(torch.zeros((1, c, 1, 1)),
                                                     requires_grad=True))
        # self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)

        x = self.dropout1(x)


        return inp + x * self.beta
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        # dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)

        self.act = nn.GELU()


    def forward(self, inp):
        x = inp

        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)


        return inp + x
class fft_bench_complex_mlp(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, window_size=None, bias=False, fft=True):
        super(fft_bench_complex_mlp, self).__init__()
        self.act_fft = act_method()
        self.window_size = window_size
        # dim = out_channel
        self.fft = fft
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_weight1_real = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(hid_dim, dim))
        self.complex_weight2_imag = nn.Parameter(torch.Tensor(hid_dim, dim))
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(16))
        if bias:
            self.b1_real = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
            self.b1_imag = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
            self.b2_real = nn.Parameter(torch.zeros((1, 1, 1, dim)), requires_grad=True)
            self.b2_imag = nn.Parameter(torch.zeros((1, 1, 1, dim)), requires_grad=True)
        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size is not None and (H != self.window_size[0] or W != self.window_size[1]):
            x, batch_list = window_partitionx(x, self.window_size)
        if self.fft:
            x = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
        weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
        if self.bias:
            b1 = torch.complex(self.b1_real, self.b1_imag)
            b2 = torch.complex(self.b2_real, self.b2_imag)
        # y = rearrange(x, 'b c h w -> b h w c')
        y = x.permute(0, 2, 3, 1)
        y = y @ weight1
        if self.bias:
            y = y + b1
        y = torch.cat([y.real, y.imag], dim=dim)

        y = self.act_fft(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = y @ weight2
        if self.bias:
            y = y + b2
        y = y.permute(0, 3, 1, 2)
        # y = rearrange(y, 'b h w c -> b c h w')
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        if self.window_size is not None and (H != self.window_size[0] or W != self.window_size[1]):
            if self.fft:
                y = torch.fft.irfft2(y, s=self.window_size, norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            if self.fft:
                y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y

class ResFFTReLUBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        # dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.main_fft = fft_bench_complex_mlp(c)
        self.act = nn.GELU()


    def forward(self, inp):
        x = inp
        x_fft = self.main_fft(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)


        return inp + x + x_fft
class LNConvBlock(nn.Module):
    def __init__(self, c, drop_out_rate=0.):
        super().__init__()
        # dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)

        self.act = nn.ReLU()
        # SimpleGate

        self.norm1 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.register_parameter('beta', nn.Parameter(torch.zeros((1, c, 1, 1)),
                                                     requires_grad=True))
        # self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)

        x = self.dropout1(x)


        return x * self.beta
class ConvBlock(nn.Module):
    def __init__(self, c, drop_out_rate=0.):
        super().__init__()
        # dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)

        self.act = nn.ReLU()
        # SimpleGate

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()


    def forward(self, inp):

        x = self.conv1(inp)
        x = self.act(x)
        x = self.conv2(x)

        x = self.dropout1(x)


        return x

class ResMLPBlock(nn.Module):
    def __init__(self, dim, train_size=[64, 64]):
        super().__init__()

        # dw_channel = c * DW_Expand
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1,
                      bias=False),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, stride=1,
                      groups=dim, bias=False),
                nn.GELU(),
                nn.Linear(train_size[1], train_size[1]),
                nn.GELU(),
                Rearrange('b c h w -> b c w h'),
                nn.Linear(train_size[0], train_size[0]),
                Rearrange('b c w h -> b c h w'),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1,
                      bias=False)
                )


    def forward(self, inp):
        x = self.mlp(inp) + inp

        return x
class ResMLPMixBlock(nn.Module):
    def __init__(self, dim, train_size, patch_size, scattn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim*2, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1, stride=1,
                               groups=dim * 2, bias=False)
        self.patch_size = patch_size
        # dw_channel = c * DW_Expand
        self.scattn = scattn
        self.mlp = nn.Sequential(
                nn.Linear(train_size[1], train_size[1]),
                nn.GELU(),
                Rearrange('b c h w -> b c w h'),
                nn.Linear(train_size[0], train_size[0]),
                Rearrange('b c w h -> b c h w'),
                )
        n = patch_size[0] * patch_size[1]
        self.mix = nn.Sequential(
            Rearrange('b c (h1 p1) (w1 p2) -> b c h1 w1 (p1 p2)', p1=patch_size[0], p2=patch_size[1]),
            nn.Linear(n, n),
            nn.GELU(),
            Rearrange('b c h1 w1 (p1 p2)-> b c (h1 p1) (w1 p2)', p1=patch_size[0], p2=patch_size[1]),
            Rearrange('b c (h1 p1) (w1 p2)-> b c p1 p2 (h1 w1)', h1=patch_size[0], w1=patch_size[1]),
            nn.Linear(n, n),
            Rearrange('b c p1 p2 (h1 w1) -> b c (h1 p1) (w1 p2)', h1=patch_size[0], w1=patch_size[1])
        )
        if self.scattn:
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1,
                          groups=1, bias=True),
            )

        self.conv3 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=False)
    def forward(self, inp):

        x1, x2 = self.conv2(self.conv1(inp)).chunk(2, dim=1)
        b, c, h, w = x2.shape
        h_n = (self.patch_size[0] - h % self.patch_size[0]) % self.patch_size[0]
        w_n = (self.patch_size[1] - w % self.patch_size[1]) % self.patch_size[1]

        x2= torch.nn.functional.pad(x2, (0, w_n, 0, h_n), mode='reflect')
        x = self.mlp(x1) * self.mix(x2)[:, :, :h, :w]
        if self.scattn:
            x = x * self.sca(x)
        x = self.conv3(x)
        return x + inp

class ResGateBlock(nn.Module):
    def __init__(self, dim, scattn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim*2, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1, stride=1,
                               groups=dim * 2, bias=False)
        # self.patch_size = patch_size
        # dw_channel = c * DW_Expand
        self.scattn = scattn
        self.act = nn.GELU()

        if self.scattn:
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1,
                          groups=1, bias=True),
            )

        self.conv3 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=False)
    def forward(self, inp):

        x1, x2 = self.conv2(self.conv1(inp)).chunk(2, dim=1)

        x = x1 * self.act(x2)
        if self.scattn:
            x = x * self.sca(x)
        x = self.conv3(x)
        return x + inp

class ResDepthConvBlock(nn.Module):
    def __init__(self, dim, scattn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, stride=1,
                               groups=dim, bias=False)
        # self.patch_size = patch_size
        # dw_channel = c * DW_Expand
        self.scattn = scattn
        self.act = nn.GELU()

        if self.scattn:
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1,
                          groups=1, bias=True),
            )

        self.conv3 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=False)
    def forward(self, inp):

        x = self.conv2(self.conv1(inp))

        x = x * self.act(x)
        if self.scattn:
            x = x * self.sca(x)
        x = self.conv3(x)
        return x + inp
class ConvBlock1x1(nn.Module):
    def __init__(self, c, drop_out_rate=0.):
        super().__init__()
        # dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1,
                               bias=True)

        self.act = nn.ReLU()
        # SimpleGate

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()


    def forward(self, x):

        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)

        x = self.dropout1(x)


        return x
class LNResBlock1x1(nn.Module):
    def __init__(self, c, drop_out_rate=0.):
        super().__init__()
        # dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1,
                               bias=True)

        self.act = nn.ReLU()
        # SimpleGate

        self.norm1 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.register_parameter('beta', nn.Parameter(torch.zeros((1, c, 1, 1)),
                                                     requires_grad=True))
        # self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)

        x = self.dropout1(x)


        return inp + x * self.beta
class ResBlock(nn.Module):
    def __init__(self, ch):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            # nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, input):
        res = self.body(input)
        output = res + input
        return output
class NAFBlock(nn.Module):
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
class ComplexLinear(nn.Module):
    def __init__(
        self,
        dim,
        dim_out
    ):
        super().__init__()
        linear = nn.Linear(dim, dim_out, dtype = torch.cfloat)
        # self.weight = nn.Parameter(torch.view_as_real(linear.weight))
        # self.bias = nn.Parameter(torch.view_as_real(linear.bias))
        self.register_parameter('weight', nn.Parameter(torch.view_as_real(linear.weight), requires_grad=True))
        self.register_parameter('bias', nn.Parameter(torch.view_as_real(linear.bias), requires_grad=True))

    def forward(self, x):
        weight = torch.view_as_complex(self.weight)
        bias = torch.view_as_complex(self.bias)
        return F.linear(x, weight, bias)
class fft_branch(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, window_size=None, bias=False, fft=True, fft_mode='complex'):
        super(fft_branch, self).__init__()
        self.act_fft = act_method()
        self.window_size = window_size
        # dim = out_channel
        self.fft_mode = fft_mode
        self.fft = fft
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        if fft_mode == 'complex':
            self.complex_weight1_linear = ComplexLinear(dim, hid_dim)
            self.complex_weight2_linear = ComplexLinear(hid_dim, dim)
        elif fft_mode == 'batch':
            self.complex_weight1_linear = nn.Conv2d(dim, hid_dim, kernel_size=1, bias=bias)
            self.complex_weight2_linear = nn.Conv2d(hid_dim, dim, kernel_size=1, bias=bias)
        else:
            groups = 2 if 'split' in fft_mode else 1
            self.complex_weight1_linear = nn.Conv2d(dim*2, hid_dim*2, kernel_size=1, bias=bias, groups=groups)
            self.complex_weight2_linear = nn.Conv2d(hid_dim*2, dim*2, kernel_size=1, bias=bias, groups=groups)
        if fft_mode == 'batch':
            self.dim = 0
        else:
            self.dim = 1
        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size is not None and (H != self.window_size[0] or W != self.window_size[1]):
            x, batch_list = window_partitionx(x, self.window_size)
        if self.fft:
            x = torch.fft.rfft2(x, norm=self.norm)

        # y = rearrange(x, 'b c h w -> b h w c')
        if self.fft_mode == 'complex':
            x = x.permute(0, 2, 3, 1)
        y = self.complex_weight1_linear(x)

        y = torch.cat([y.real, y.imag], dim=self.dim)

        y = self.act_fft(y)
        y_real, y_imag = torch.chunk(y, 2, dim=self.dim)
        y = torch.complex(y_real, y_imag)
        y = self.complex_weight2_linear(y)
        if self.fft_mode == 'complex':
            y = y.permute(0, 3, 1, 2)
        # y = rearrange(y, 'b h w c -> b c h w')
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        if self.window_size is not None and (H != self.window_size[0] or W != self.window_size[1]):
            if self.fft:
                y = torch.fft.irfft2(y, s=self.window_size, norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            if self.fft:
                y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class FourierNAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., local_avgpool=False, train_size=[256, 256],
                 patch_size=None, DW_Fourier=1, window_size=None, fft_mode='channel'):
        super().__init__()
        dw_channel = c * DW_Expand
        self.fft_mode = fft_mode
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
        self.fft_block1 = fft_branch(c, DW_Fourier, window_size=window_size, bias=True,
                                                   act_method=nn.GELU, fft_mode=fft_mode)  # , act_method=nn.GELU
        self.fft_block2 = fft_branch(c, DW_Fourier, window_size=window_size, bias=True, act_method=nn.GELU, fft_mode=fft_mode)
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.register_parameter('beta', nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True))
        self.register_parameter('gamma', nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True))
        # self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)

        b, c, H, W = x.shape
        if self.patch_size is not None and (H != self.patch_size[0] or W != self.patch_size[1]):
            x, batch_list = window_partitionx(x, self.patch_size)
        x = x * self.sca(x)
        if self.patch_size is not None and (H != self.patch_size[0] or W != self.patch_size[1]):
            x = window_reversex(x, self.patch_size, H, W, batch_list)
        x = self.conv3(x)
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x)))  # * self.gamma
        # else:
        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x


class EDFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(EDFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.register_parameter('fft', nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)),
                                                    requires_grad=True))
        # self.fft = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))

    def forward(self, x):

        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2

        x = self.project_out(x)
        # if self.global_fft:
        #     x += x_global
        b, c, h, w = x.shape
        h_n = (8 - h % 8) % 8
        w_n = (8 - w % 8) % 8

        x = torch.nn.functional.pad(x, (0, w_n, 0, h_n), mode='reflect')
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)

        x = x[:, :, :h, :w]

        return x


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=8,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            patch_size=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.GELU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),

        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.x_conv = nn.Conv1d(in_channels=(self.dt_rank + self.d_state * 2),
                                out_channels=(self.dt_rank + self.d_state * 2), kernel_size=7, padding=3,
                                groups=(self.dt_rank + self.d_state * 2))

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 1
        x_hwwh = x.view(B, 1, -1, L)
        xs = x_hwwh

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        x_dbl = self.x_conv(x_dbl.squeeze(1)).unsqueeze(1)

        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        # print(As.shape, Bs.shape, Cs.shape, Ds.shape, dts.shape)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, **kwargs):
        x = rearrange(x, 'b c h w -> b h w c')
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        if self.patch_size is not None and (H > self.patch_size[0] or W > self.patch_size[1]):
            x, batch_list = window_partitionx(x, self.patch_size)
            # print(x.shape, batch_list)
        b, c, h, w = x.shape

        y1 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(b, h, w, -1)

        if self.patch_size is not None and (H > self.patch_size[0] or W > self.patch_size[1]):
            y = rearrange(y, 'b h w c -> b c h w')
            y = window_reversex(y, self.patch_size, H, W, batch_list, mask=True)
            y = rearrange(y, 'b c h w -> b h w c')
        y = self.out_norm(y)
        y = y * F.gelu(z)
        out = self.out_proj(y)
        out = rearrange(out, 'b h w c -> b c h w')
        return out
class SS2D_Freq(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=8,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            patch_size=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.conv2d_2 = nn.Conv2d(
            in_channels=self.d_inner*2,
            out_channels=self.d_inner,
            groups=1,
            bias=conv_bias,
            kernel_size=1,
            padding=0,
            **factory_kwargs,
        )
        self.act = nn.GELU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),

        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.x_conv = nn.Conv1d(in_channels=(self.dt_rank + self.d_state * 2),
                                out_channels=(self.dt_rank + self.d_state * 2), kernel_size=7, padding=3,
                                groups=(self.dt_rank + self.d_state * 2))

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 1
        x_hwwh = x.view(B, 1, -1, L)
        xs = x_hwwh

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        x_dbl = self.x_conv(x_dbl.squeeze(1)).unsqueeze(1)

        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        # print(As.shape, Bs.shape, Cs.shape, Ds.shape, dts.shape)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, **kwargs):
        x = rearrange(x, 'b c h w -> b h w c')
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x)
        x1 = self.act(x)
        x2 = -self.act(-x)
        x = self.conv2d_2(torch.cat([x1,x2], dim=1))
        if self.patch_size is not None and (H > self.patch_size[0] or W > self.patch_size[1]):
            x, batch_list = window_partitionx(x, self.patch_size)
            # print(x.shape, batch_list)
        b, c, h, w = x.shape

        y1 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(b, h, w, -1)

        if self.patch_size is not None and (H > self.patch_size[0] or W > self.patch_size[1]):
            y = rearrange(y, 'b h w c -> b c h w')
            y = window_reversex(y, self.patch_size, H, W, batch_list, mask=True)
            y = rearrange(y, 'b c h w -> b h w c')
        y = self.out_norm(y)
        y = y * F.gelu(z)
        out = self.out_proj(y)
        out = rearrange(out, 'b h w c -> b c h w')
        return out

class ResSS2D_Freq(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=8,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            patch_size=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.conv2d_2 = nn.Conv2d(
            in_channels=self.d_inner*2,
            out_channels=self.d_inner,
            groups=1,
            bias=conv_bias,
            kernel_size=1,
            padding=0,
            **factory_kwargs,
        )
        self.act = nn.GELU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),

        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.x_conv = nn.Conv1d(in_channels=(self.dt_rank + self.d_state * 2),
                                out_channels=(self.dt_rank + self.d_state * 2), kernel_size=7, padding=3,
                                groups=(self.dt_rank + self.d_state * 2))

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 1
        x_hwwh = x.view(B, 1, -1, L)
        xs = x_hwwh

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        x_dbl = self.x_conv(x_dbl.squeeze(1)).unsqueeze(1)

        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        # print(As.shape, Bs.shape, Cs.shape, Ds.shape, dts.shape)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, **kwargs):
        inp = x
        x = rearrange(x, 'b c h w -> b h w c')
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x)
        x1 = self.act(x)
        x2 = -self.act(-x)
        x = self.conv2d_2(torch.cat([x1,x2], dim=1))
        if self.patch_size is not None and (H > self.patch_size[0] or W > self.patch_size[1]):
            x, batch_list = window_partitionx(x, self.patch_size)
            # print(x.shape, batch_list)
        b, c, h, w = x.shape

        y1 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(b, h, w, -1)

        if self.patch_size is not None and (H > self.patch_size[0] or W > self.patch_size[1]):
            y = rearrange(y, 'b h w c -> b c h w')
            y = window_reversex(y, self.patch_size, H, W, batch_list, mask=True)
            y = rearrange(y, 'b c h w -> b h w c')
        y = self.out_norm(y)
        y = y * F.gelu(z)
        out = self.out_proj(y)
        out = rearrange(out, 'b h w c -> b c h w')
        return out + inp
class ResSS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=8,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            patch_size=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.GELU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),

        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.x_conv = nn.Conv1d(in_channels=(self.dt_rank + self.d_state * 2),
                                out_channels=(self.dt_rank + self.d_state * 2), kernel_size=7, padding=3,
                                groups=(self.dt_rank + self.d_state * 2))

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 1
        x_hwwh = x.view(B, 1, -1, L)
        xs = x_hwwh

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        x_dbl = self.x_conv(x_dbl.squeeze(1)).unsqueeze(1)

        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        # print(As.shape, Bs.shape, Cs.shape, Ds.shape, dts.shape)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, **kwargs):
        inp = x
        x = rearrange(x, 'b c h w -> b h w c')
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        if self.patch_size is not None and (H > self.patch_size[0] or W > self.patch_size[1]):
            x, batch_list = window_partitionx(x, self.patch_size)
            # print(x.shape, batch_list)
        b, c, h, w = x.shape

        y1 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(b, h, w, -1)

        if self.patch_size is not None and (H > self.patch_size[0] or W > self.patch_size[1]):
            y = rearrange(y, 'b h w c -> b c h w')
            y = window_reversex(y, self.patch_size, H, W, batch_list, mask=True)
            y = rearrange(y, 'b c h w -> b h w c')
        y = self.out_norm(y)
        y = y * F.gelu(z)
        out = self.out_proj(y)
        out = rearrange(out, 'b h w c -> b c h w')
        return out + inp
class ResMultiSS2D_Freq(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=8,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            patch_size=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.conv2d_2 = nn.Conv2d(
            in_channels=self.d_inner*2,
            out_channels=self.d_inner,
            groups=1,
            bias=conv_bias,
            kernel_size=1,
            padding=0,
            **factory_kwargs,
        )
        self.act = nn.GELU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),

        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.x_conv = nn.Conv1d(in_channels=(self.dt_rank + self.d_state * 2),
                                out_channels=(self.dt_rank + self.d_state * 2), kernel_size=7, padding=3,
                                groups=(self.dt_rank + self.d_state * 2))

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 1
        x_hwwh = x.view(B, 1, -1, L)
        xs = x_hwwh

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        x_dbl = self.x_conv(x_dbl.squeeze(1)).unsqueeze(1)

        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        # print(As.shape, Bs.shape, Cs.shape, Ds.shape, dts.shape)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, **kwargs):
        inp = x
        x = rearrange(x, 'b c h w -> b h w c')
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x)
        x1 = self.act(x)
        x2 = -self.act(-x)
        x = self.conv2d_2(torch.cat([x1,x2], dim=1))
        if self.patch_size is not None and (H > self.patch_size[0] or W > self.patch_size[1]):
            x, batch_list = window_partitionx(x, self.patch_size)
            # print(x.shape, batch_list)
        b, c, h, w = x.shape

        y1 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(b, h, w, -1)

        if self.patch_size is not None and (H > self.patch_size[0] or W > self.patch_size[1]):
            y = rearrange(y, 'b h w c -> b c h w')
            y = window_reversex(y, self.patch_size, H, W, batch_list, mask=True)
            y = rearrange(y, 'b c h w -> b h w c')
        y = self.out_norm(y)
        y = y * F.gelu(z)
        out = self.out_proj(y)
        out = rearrange(out, 'b h w c -> b c h w')
        return out * inp
class EVSS(nn.Module):
    def __init__(self, dim, patch_size=None):
        super(EVSS, self).__init__()

        # self.att = att
        # if self.att:
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = SS2D(dim, patch_size=patch_size)

        # self.norm2 = LayerNorm2d(dim)
        # self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # if self.att:
        x = x + self.attn(self.norm1(x))

        # x = x + self.ffn(self.norm2(x))

        return x
class EVSS_LN1(nn.Module):
    def __init__(self, dim, patch_size=None):
        super(EVSS_LN1, self).__init__()

        # self.att = att
        # if self.att:
        self.norm1 = LayerNorm2d(dim)
        self.attn = SS2D(dim, patch_size=patch_size)

        # self.norm2 = LayerNorm2d(dim)
        # self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # if self.att:
        x = x + self.attn(self.norm1(x))

        # x = x + self.ffn(self.norm2(x))

        return x
class EVSS_Freq(nn.Module):
    def __init__(self, dim, patch_size=None):
        super(EVSS_Freq, self).__init__()

        # self.att = att
        # if self.att:
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = SS2D_Freq(dim, patch_size=patch_size)

        # self.norm2 = LayerNorm2d(dim)
        # self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # if self.att:
        x = x + self.attn(self.norm1(x))

        # x = x + self.ffn(self.norm2(x))

        return x
class EVSS_noskip(nn.Module):
    def __init__(self, dim, patch_size=None):
        super(EVSS_noskip, self).__init__()

        # self.att = att
        # if self.att:
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = SS2D(dim, patch_size=patch_size)

        # self.norm2 = LayerNorm2d(dim)
        # self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # if self.att:
        x = self.attn(self.norm1(x))

        # x = x + self.ffn(self.norm2(x))

        return x
class EVS2S(nn.Module):
    def __init__(self, dim, patch_size=None):
        super(EVS2S, self).__init__()

        # self.att = att
        # if self.att:
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = SS2D(dim, patch_size=patch_size)
        self.attn2 = SS2D(dim, patch_size=patch_size)
        # self.norm2 = LayerNorm2d(dim)
        # self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # if self.att:
        x = self.attn(self.norm1(x))

        # x = x + self.ffn(self.norm2(x))

        return self.attn2(x)
class EVS4S(nn.Module):
    def __init__(self, dim, patch_size=None):
        super(EVS4S, self).__init__()

        # self.att = att
        # if self.att:
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn1 = SS2D(dim, patch_size=patch_size)
        self.attn2 = SS2D(dim, patch_size=patch_size)
        self.attn3 = SS2D(dim, patch_size=patch_size)
        self.attn4 = SS2D(dim, patch_size=patch_size)
        # self.norm2 = LayerNorm2d(dim)
        # self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):

        x = self.norm1(x)
        # x = torch.flip(x, dims=(-2, -1)).contiguous()
        x = self.attn1(x)
        # x = torch.transpose(x, dim0=-2, dim1=-1).contiguous()
        x = self.attn2(x)
        # x = torch.flip(x, dims=(-2, -1)).contiguous()
        x = self.attn3(x)
        # x = torch.transpose(x, dim0=-2, dim1=-1).contiguous()
        x = self.attn4(x)
        # x = x + self.ffn(self.norm2(x))

        return x

class ResSS2D4ES(nn.Module):
    def __init__(self, dim, patch_size=None):
        super(ResSS2D4ES, self).__init__()

        # self.att = att
        # if self.att:
        # self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn1 = SS2D(dim, patch_size=patch_size)
        self.attn2 = SS2D(dim, patch_size=patch_size)
        self.attn3 = SS2D(dim, patch_size=patch_size)
        self.attn4 = SS2D(dim, patch_size=patch_size)

        self.d1 = nn.Parameter(torch.cat([torch.ones((1, dim//2, 1, 1)) * -1, torch.ones((1, dim//2, 1, 1))], dim=1),
                               requires_grad=False)
        self.d2 = nn.Parameter(torch.cat([torch.ones((1, dim // 2, 1, 1)), torch.ones((1, dim // 2, 1, 1)) * -1], dim=1),
                               requires_grad=False)

        # self.norm2 = LayerNorm2d(dim)
        # self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):

        # x = self.norm1(x)
        # x = torch.flip(x, dims=(-2, -1)).contiguous()
        x = self.d1 * x
        x = self.attn1(x) + x
        # x = torch.transpose(x, dim0=-2, dim1=-1).contiguous()
        x = self.d2 * x
        x = self.attn2(x)+ x
        # x = torch.flip(x, dims=(-2, -1)).contiguous()
        x = self.d1 * x
        x = self.attn3(x)+ x
        # x = torch.transpose(x, dim0=-2, dim1=-1).contiguous()
        x = self.d2 * x
        x = self.attn4(x)+ x
        # x = x + self.ffn(self.norm2(x))

        return x
class EVSBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=3, bias=False, att=True, idx=3, patch_size=[256, 256]):
        super(EVSBlock, self).__init__()

        self.att = att
        self.idx = idx
        if self.att:
            self.norm1 = LayerNorm(dim, 'WithBias')
            self.attn = SS2D(d_model=dim, patch=patch_size)

        self.norm2 = LayerNorm(dim, 'WithBias')
        self.ffn = EDFFN(dim, ffn_expansion_factor, bias)

        self.kernel_size = patch_size


    def forward(self, x):
        if self.att:
            if self.idx >= 0:
                if self.idx % 2 == 1:
                    x = torch.flip(x, dims=(-2, -1)).contiguous()
                if self.idx % 2 == 0:
                    x = torch.transpose(x, dim0=-2, dim1=-1).contiguous()

            x = x + self.attn(self.norm1(x))

        x = x + self.ffn(self.norm2(x))

        return x


class NAFEVSBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., local_avgpool=False, train_size=[256, 256],
                 patch_size=None, window_size=None, idx=-1):
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

        self.idx = idx
        self.inner_rank = 32
        self.norm_mamba = LayerNorm2d(dw_channel // 2)
        self.attn = SS2D(dw_channel // 2, patch_size=window_size)

        self.register_parameter('lamb_l', nn.Parameter(torch.ones((1, c, 1, 1)),
                                                       requires_grad=True))
        self.register_parameter('lamb_g', nn.Parameter(torch.zeros((1, c, 1, 1)),
                                                       requires_grad=True))
        self.register_parameter('alpha', nn.Parameter(torch.ones((1, c, 1, 1)),
                                                      requires_grad=True))
        self.register_parameter('gamma', nn.Parameter(torch.zeros((1, c, 1, 1)),
                                                      requires_grad=True))
        self.register_parameter('beta', nn.Parameter(torch.zeros((1, c, 1, 1)),
                                                     requires_grad=True))

    def forward(self, inp):
        if self.idx != -1:

            if self.idx % 2 == 1:
                inp = torch.flip(inp, dims=(-2, -1)).contiguous()
            if self.idx % 2 == 0:
                inp = torch.transpose(inp, dim0=-2, dim1=-1).contiguous()
        x = inp

        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)

        b, c, H, W = x.shape
        x_local = x
        if self.patch_size is not None and (H != self.patch_size[0] or W != self.patch_size[1]):
            x_local, batch_list = window_partitionx(x_local, self.patch_size)
        x_local = x_local * self.sca(x_local)
        if self.patch_size is not None and (H != self.patch_size[0] or W != self.patch_size[1]):
            x_local = window_reversex(x_local, self.patch_size, H, W, batch_list)

        x_local = self.conv3(x_local)
        x_local = self.conv3(x_local)
        x_global = self.norm_mamba(x)
        x_global = self.attn(x_global) * (self.alpha + 1.)


        x = x_local * self.lamb_l + x_global * self.lamb_g  # .view(b, h, w, c).permute(0, 3, 1, 2)
        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma