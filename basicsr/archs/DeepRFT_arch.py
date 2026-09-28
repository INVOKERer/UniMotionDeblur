
# from basicsr.models.archs.arch_util import *
import torch
from torch.nn import init as init
import math
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from .util_win import *
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, relu_method=nn.ReLU, transpose=False,
                 channel_shuffle_g=0, norm_method=nn.BatchNorm2d, groups=1):
        super(BasicConv, self).__init__()
        self.channel_shuffle_g = channel_shuffle_g
        self.norm = norm
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        if norm:
            layers.append(norm_method(out_channel))

        if relu:
            if relu_method == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif relu_method == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))
            else:
                layers.append(relu_method())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
class conv_bench(nn.Module):
    def __init__(self, n_feat, kernel_size=3, act_method=nn.ReLU, bias=False):
        super(conv_bench, self).__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=bias)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=bias)
        self.act = act_method()

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))
class ResBlock(nn.Module):
    def __init__(self, n_feat, kernel_size=3, norm='backward'): # 'backward'
        super(ResBlock, self).__init__()
        self.main = conv_bench(n_feat, kernel_size=kernel_size)
        # self.main_fft = fft_bench_complex_mlp(n_feat, norm)

    def forward(self, x):
        res = self.main(x) # + self.main_fft(x)
        # res += x
        return res + x
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

class fft_branch(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, window_size=None, bias=False, fft=True, fft_mode='channel'):
        super(fft_branch, self).__init__()
        self.act_fft = act_method()
        self.window_size = window_size
        # dim = out_channel
        self.fft_mode = fft_mode
        self.fft = fft
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        if fft_mode == 'batch':
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

        y = torch.cat([x.real, x.imag], dim=self.dim)
        y = self.complex_weight1_linear(y)



        y = self.act_fft(y)


        y = self.complex_weight2_linear(y)
        y_real, y_imag = torch.chunk(y, 2, dim=self.dim)
        y = torch.complex(y_real, y_imag)
        # if self.fft_mode == 'complex':
        #     y = y.permute(0, 3, 1, 2)
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
class ResFourier(nn.Module):
    def __init__(self, n_feat, kernel_size=3, norm='backward', num_heads=1): # 'backward'
        super(ResFourier, self).__init__()
        self.main = conv_bench(n_feat, kernel_size=kernel_size)
        self.main_fft = fft_branch(n_feat, norm=norm)

    def forward(self, x):

        res = self.main(x) + self.main_fft(x)
        # res += x
        return res + x
class ResFourier_complex(nn.Module):
    def __init__(self, n_feat, kernel_size=3, norm='backward', num_heads=1): # 'backward'
        super(ResFourier_complex, self).__init__()
        self.main = conv_bench(n_feat, kernel_size=kernel_size)
        self.main_fft = fft_bench_complex_mlp(n_feat, norm=norm)

    def forward(self, x):

        res = self.main(x) + self.main_fft(x)
        # res += x
        return res + x
class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8, ResBlock=ResBlock):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DBlock(nn.Module):
    def __init__(self, channel, num_res=8, ResBlock=ResBlock):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class AFF(nn.Module):
    def __init__(self, in_channel, out_channel, BasicConv=BasicConv):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)

class SCM(nn.Module):
    def __init__(self, out_plane, BasicConv=BasicConv, inchannel=3):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(inchannel, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-inchannel, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)

class FAM(nn.Module):
    def __init__(self, channel, BasicConv=BasicConv):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out
def get_P(phase, fnorm='backward'):
    return torch.fft.irfft2(torch.exp(1j*phase), norm=fnorm)
def get_iP(Phase, fnorm='backward'):
    rf_P = torch.fft.rfft2(Phase, norm=fnorm)
    x_ir_angle = torch.log(rf_P) / 1j
    return x_ir_angle.real, x_ir_angle.imag # torch.abs(en1x_ir_angle) #
@ARCH_REGISTRY.register()
class DeepRFT(nn.Module):
    def __init__(self, width=32, num_res=20, window_size=None, inference=False):
        super(DeepRFT, self).__init__()
        self.window_size = window_size
        self.inference = inference
        ResBlockx = ResFourier
        # ResBlockx = ResFourier_complex_gelu
        # ResBlockx = ResFourier_complex
        base_channel = width

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res, ResBlock=ResBlockx),
            EBlock(base_channel*2, num_res, ResBlock=ResBlockx),
            EBlock(base_channel*4, num_res, ResBlock=ResBlockx),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res, ResBlock=ResBlockx),
            DBlock(base_channel * 2, num_res, ResBlock=ResBlockx),
            DBlock(base_channel, num_res, ResBlock=ResBlockx)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1, BasicConv=BasicConv),
            AFF(base_channel * 7, base_channel*2, BasicConv=BasicConv)
        ])

        self.FAM1 = FAM(base_channel * 4, BasicConv=BasicConv)
        self.SCM1 = SCM(base_channel * 4, BasicConv=BasicConv)
        self.FAM2 = FAM(base_channel * 2, BasicConv=BasicConv)
        self.SCM2 = SCM(base_channel * 2, BasicConv=BasicConv)

    def forward(self, x, gt=None):
        _, _, H, W = x.shape
        # if self.window_size is not None and (H != self.window_size or W != self.window_size):
        #     x, batch_list = window_partitionx(x, self.window_size)
        # print(x.shape)
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        z = self.Decoder[0](z)
        if not self.inference:
            z_ = self.ConvsOut[0](z)
            out_3 = z_+x_4
            # if self.window_size is not None and (H != self.window_size or W != self.window_size):
            #     out_3 = window_reversex(out_3, self.window_size//4, H//4, W//4, batch_list)
            outputs.append(out_3)
        z = self.feat_extract[3](z)
        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)

        if not self.inference:
            z_ = self.ConvsOut[1](z)
            out_2 = z_ + x_2
            # if self.window_size is not None and (H != self.window_size or W != self.window_size):
            #     out_2 = window_reversex(out_2, self.window_size // 2, H // 2, W // 2, batch_list)
            outputs.append(out_2)
        z = self.feat_extract[4](z)
        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        if not self.inference:
            # outputs.append(z+x)
            out = z + x
            # if self.window_size is not None and (H != self.window_size or W != self.window_size):
            #     out = window_reversex(out, self.window_size, H, W, batch_list)
            outputs.append(out)
            return outputs # [::-1]
        else:
            out = z + x
            # if self.window_size is not None and (H != self.window_size or W != self.window_size):
            #     out = window_reversex(out, self.window_size, H, W, batch_list)
            return out





if __name__=='__main__':
    import torch
    TLPLN = DeepRFT() # .cuda()
    x = torch.randn(1,3,256,256)
    x = x # .cuda()
    y = TLPLN(x)
    print(x.shape, y)

