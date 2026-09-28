
# from basicsr.models.archs.arch_util import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

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
class MIMOUNet(nn.Module):
    def __init__(self, width=32, num_res=20, inference=False, window_size=None):
        super(MIMOUNet, self).__init__()
        # self.window_size = window_size
        self.inference = inference
        ResBlockx = ResBlock
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
    TLPLN = MIMOUNet() # .cuda()
    x = torch.randn(1,3,256,256)
    x = x # .cuda()
    y = TLPLN(x)
    print(x.shape, y)

