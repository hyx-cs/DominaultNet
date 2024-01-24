import torch
import torch.nn as nn
import torch.nn.functional as F
from model.common import *
class invPixelShuffle(nn.Module):
    def __init__(self, ratio=2):
        super(invPixelShuffle, self).__init__()
        self.ratio = ratio
    def forward(self, tensor):
        ratio = self.ratio
        b = tensor.size(0)
        ch = tensor.size(1)
        y = tensor.size(2)
        x = tensor.size(3)
        assert x % ratio == 0 and y % ratio == 0, 'x, y, ratio : {}, {}, {}'.format(x, y, ratio)
        return tensor.view(b, ch, y // ratio, ratio, x // ratio, ratio).permute(0, 1, 3, 5, 2, 4).contiguous().view(b,-1,y // ratio,x // ratio)

class upsample(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.Conv33 = nn.Conv2d(in_channels=n_feats, out_channels=n_feats*2, kernel_size=3, stride=1, padding=1)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)
    def forward(self, x):
        x = self.Conv33(x)
        x = self.pixelshuffle(x)
        return x
class down_sampler_module(nn.Module):
    def __init__(self, in_channels, bn=False, act=False, bias=True):
        super(down_sampler_module, self).__init__()

        self.up = invPixelShuffle(2)
        self.conv = nn.Conv2d(in_channels=in_channels*4, out_channels=in_channels, kernel_size=3,
        stride=1, padding=1) #conv(n_feat*4, n_feat, 3, bias)
    def forward(self, x):
        x = self.up(x)
        # print(x.shape)
        x = self.conv(x)
        return x

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()

        img_channels = 3
        n_feats = 64

        self.Linear = nn.Conv2d(n_feats, img_channels, kernel_size=3, stride=1, padding=1)
        self.head = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.Pre = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1)

        self.upsample_1 = upsample(n_feats=128)
        self.upsample_2 = upsample(n_feats=256)
        self.upsample_3 = upsample(n_feats=512)
        self.down_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.down_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.down_3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)

        self.PCSR_1_64 = PCSR1(nf=64)
        self.PCSR_1_128 = PCSR1(nf=128)
        self.PCSR_1_256 = PCSR1(nf=256)
        self.PCSR_1_512 = PCSR1(nf=512)

        self.PCSR2 = PCSR2(n_feats=64)

        self.Con64_64 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1)
        self.Con64_128 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1)
        self.Con64_256 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1)

        self.fusion_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fusion_2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.fusion_3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.CA = cbam.ChannelGate(gate_channels=64)

    def forward(self, x, parsing=None):

        feature_map = []
        parsing_32 = parsing
        parsing_64 = nn.functional.interpolate(parsing, scale_factor=2, mode='nearest')
        parsing_128 = nn.functional.interpolate(parsing_64, scale_factor=2, mode='nearest')

        #   Stage1  重建出空间分辨率一致的图像
        x = self.head(x)
        x_up = nn.functional.interpolate(x, scale_factor=2)
        x_1 = self.PCSR2(self.PCSR2(x_up, parsing_32), parsing_32)
        feature_map.append(x_1)
        x_1_up = nn.functional.interpolate(x_1, scale_factor=2)
        x_2 = self.PCSR2(self.PCSR2(x_1_up, parsing_64), parsing_64)
        feature_map.append(x_2)
        x_2_up = nn.functional.interpolate(x_2, scale_factor=2)
        x_3 = self.PCSR2(self.PCSR2(x_2_up, parsing_128), parsing_128)
        feature_map.append(x_3)
        # stage_1 = self.Linear(x_3)


        #   Stage2  进行通道数深入，提高图像的细节和泛化能力
        y_1 = self.down_1(x_3)
        y_1 = self.PCSR_1_128(y_1)
        feature_map.append(y_1)
        y_2 = self.down_2(y_1)
        y_2 = self.PCSR_1_256(y_2)
        feature_map.append(y_2)
        y_3 = self.down_3(y_2)
        y_3 = self.PCSR_1_512(y_3)
        feature_map.append(y_3)
        z_1 = self.upsample_3(y_3)
        z_1 = self.PCSR_1_256(self.fusion_1(torch.cat((z_1, y_2), dim=1)))
        feature_map.append(z_1)
        z_2 = self.upsample_2(z_1)
        z_2 = self.PCSR_1_128(self.fusion_2(torch.cat((z_2, y_1), dim=1)))
        feature_map.append(z_2)
        z_3 = self.upsample_1(z_2)
        z_3 = self.PCSR_1_64(self.fusion_3(torch.cat((z_3, x_3), dim=1)))
        feature_map.append(z_3)
        # stage_2 = self.Linear(out)


        # Stage3 进行高质量图像重建
        out = self.PCSR2(z_3, parsing_128)
        feature_map.append(out)
        out = self.Linear(out)
        return out, feature_map


if __name__ == '__main__':
    net = net()
    print(net)
