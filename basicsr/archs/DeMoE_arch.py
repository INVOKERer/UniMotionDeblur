import torch
import torch.nn as nn
import torch.nn.functional as F
from .util_basicblock import NAFBlock
from basicsr.utils.registry import ARCH_REGISTRY
# try:
#     from .arch_util import CustomSequential
#     from .arch_model import EfficientClassificationHead,NAFBlock
#     from .moeblocks import  MoEBlock
# except:
#     from arch_util import CustomSequential
#     from arch_model import EfficientClassificationHead, NAFBlock
#     from moeblocks import  MoEBlock



class CustomSequential(nn.Module):
    '''
    Similar to nn.Sequential, but it lets us introduce a second argument in the forward method
    so adaptors can be considered in the inference.
    '''
    def __init__(self, *args):
        super(CustomSequential, self).__init__()
        self.modules_list = nn.ModuleList(args)

    def forward(self, x):
        for module in self.modules_list:
            x = module(x)
        return x
class EfficientClassificationHead(nn.Module):

    def __init__(self, in_channels, num_classes=5):
        super().__init__()
        self.conv_bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),  # Channel reduction
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2))

        self.attention = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid())

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.conv_bottleneck(x)
        attention_mask = self.attention(x)
        x = x * attention_mask  # Spatial attention
        return self.classifier(x)
class MoEBlock(nn.Module):
    def __init__(self, c, n=5, used=3):
        super().__init__()
        self.used = int(used)
        self.num_experts = n
        self.experts = nn.ModuleList([NAFBlock(c=c) for _ in range(n)])

    # Sparse implementation for large n
    def forward(self, feat, weights):
        B, _, _, _ = feat.shape
        k = self.used
        # Get top-k weights and indices
        topk_weights, topk_indices = torch.topk(weights, k, dim=1)  # (B, k)
        expert_counts = torch.bincount(topk_indices.flatten(), minlength=self.num_experts)
        # Apply l1 normalization to keep the sum to 1 and maintain aspect relation between weights
        topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)  # (B, k)
        mask = torch.zeros(B, self.num_experts, dtype=torch.float32, device=feat.device)
        mask.scatter_(1, topk_indices, 1.0)  # Set 1.0 for used experts

        # Initialize output tensor
        outputs = torch.zeros_like(feat)

        # Process only used experts
        for expert_idx in range(self.num_experts):
            batch_mask = mask[:, expert_idx].bool()  # Convert to boolean mask
            if batch_mask.any():
                # Get the weights for this expert
                expert_weights = topk_weights[batch_mask, (topk_indices[batch_mask] == expert_idx).nonzero()[:, 1]]
                expert_out = self.experts[expert_idx](feat[batch_mask])
                outputs[batch_mask] += expert_out * expert_weights.view(-1, 1, 1, 1)

        return outputs, expert_counts, weights
TASKS = {'defocus': [1.0, 0, 0, 0, 0],
         'global_motion': [0, 1.0, 0, 0, 0],
         'local_motion': [0, 0, 1.0, 0, 0],
         'synth_global_motion': [0, 0, 0, 1.0, 0],
         'low_light': [0, 0, 0, 0, 1.0]}
@ARCH_REGISTRY.register()
class DeMoE(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], num_exp=5, k_used=3, window_size=8):
        super().__init__()

        self.num_experts = num_exp
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.experts = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                CustomSequential(
                    *[NAFBlock(chan)  if i==0 else NAFBlock(chan) for i in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            CustomSequential(
                *[NAFBlock(chan)  if i==0 else NAFBlock(chan) for i in range(middle_blk_num)]
            )
        self.experts.append(MoEBlock(c=chan, n=num_exp, used=k_used))

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                CustomSequential(
                    *[NAFBlock(chan)  if i==0 else NAFBlock(chan) for i in range(num)]
                )
            )
            self.experts.append(MoEBlock(c=chan, n=num_exp, used=k_used))


        self.mlp_branch = EfficientClassificationHead(in_channels=width*2**len(enc_blk_nums), num_classes=num_exp)

        

        self.padder_size = 2 ** len(self.encoders)
    
    def forward(self, inp, task = 'auto'):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []
        bins = []
        weights = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        class_weights_0 = self.mlp_branch(x)
        class_weights = F.softmax(class_weights_0)
        # if the task is selected manually
        if task != 'auto':
            class_weights = torch.tensor(TASKS[task], device=x.device).unsqueeze(0).expand(B, -1)
        x = self.middle_blks(x)
        x, expert_bins, weight = self.experts[0].forward(x, class_weights)
        bins.append(expert_bins)
        weights.append(weight)
        for decoder, up, enc_skip, expert in zip(self.decoders, self.ups, encs[::-1], self.experts[1::1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            x, expert_bins, weight= expert.forward(x, class_weights)
            bins.append(expert_bins)
            weights.append(weight)
        x = self.ending(x)
        x = x + inp
    
        return {'restored': x[:, :, :H, :W],
                'bin_counts': torch.stack(bins, dim=0),
                'pred_labels': class_weights,
                'weights': weights}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), value = 0)
        return x

if __name__=='__main__':

    from ptflops import get_model_complexity_info

    net = DeMoE(img_channel=3, width=32,
                 middle_blk_num=2, enc_blk_nums=[2,2,2,2], dec_blk_nums=[2,2,2,2],k_used=1)
    print('State dict: ',len(net.state_dict().keys()))
    macs, params = get_model_complexity_info(net, input_res=(3, 256, 256), print_per_layer_stat=False, verbose=False)
    print(macs, params)

    
