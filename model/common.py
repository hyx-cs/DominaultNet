import torch.nn as nn
import math
import torch
from torch.nn import Upsample

from model import cbam
import torchvision.models as models
import torch.nn.functional as F


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4 or 3)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

#AadIN
def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)
def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=(kernel_size // 2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False
            
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)

class Upsampler_module(nn.Module):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        super(Upsampler_module, self).__init__()
        self.up = nn.Upsample(scale_factor=scale, mode='bicubic')
        self.conv = nn.Conv2d(in_channels=n_feats, out_channels=n_feats*4, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        x = self.up(x)
        return x
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

        return tensor.view(b, ch, y // ratio, ratio, x // ratio, ratio).permute(0, 1, 3, 5, 2, 4).contiguous().view(b,
                                                                                                                    -1,
                                                                                                                    y // ratio,
                                                                                                                   x // ratio)
class invUpsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(invPixelShuffle(2))
                m.append(conv(n_feat * 4, n_feat, 3, bias))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(invPixelShuffle(3))
            m.append(conv(n_feat * 9, n_feat, 3, bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(invUpsampler, self).__init__(*m)
class invUpsampler_module(nn.Module):
    def __init__(self, n_feat):
        super(invUpsampler_module, self).__init__()

        self.up = invPixelShuffle(2)
        self.conv = nn.Conv2d(in_channels=n_feat*4, out_channels=n_feat, kernel_size=3,
        stride=1, padding=1) #conv(n_feat*4, n_feat, 3, bias)

    def forward(self, x):
        x = self.up(x)
        # print(x.shape)
        x = self.conv(x)
        return x

class fringe(nn.Module):
    def __init__(self):
        super().__init__()
        self.Avgpooling = nn.AvgPool2d(kernel_size=2)
        self.Upsample = nn.Upsample(scale_factor=2)
    def forward(self, x):
        res = x
        x = self.Upsample(self.Avgpooling(x))
        out = res - x
        return out

class PAConv(nn.Module):
    def __init__(self, nf, k_size=3):
        super(PAConv, self).__init__()
        self.k2 = nn.Conv2d(nf, nf, 1)  # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
    def forward(self, x):
        y = self.k2(x)
        y = self.sigmoid(y)
        out = torch.mul(self.k3(x), y)
        out = self.k4(out)
        return out

class PCSR1(nn.Module):
    """SCPA is modified from SCNet (Jiang-Jiang Liu et al. Improving Convolutional Networks with Self-Calibrated Convolutions. In CVPR, 2020)
        Github: https://github.com/MCG-NKU/SCNet
    """
    def __init__(self, nf, reduction=2, stride=1, dilation=1):
        super(PCSR1, self).__init__()
        group_width = nf
        self.conv1_a_1 = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.conv1_a_2 = nn.Conv2d(3, group_width, kernel_size=1, bias=False)
        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.k1 = nn.Sequential(
            nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation, bias=False)
        )
        self.PAConv = PAConv(group_width)
        self.conv3 = nn.Conv2d(group_width * reduction, nf, kernel_size=1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        residual = x
        out_a = self.conv1_a_1(x)
        out_b = self.conv1_b(x)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)
        out_a = self.k1(out_a)
        out_b = self.PAConv(out_b)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)
        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out += residual
        return out


class PCSR2(nn.Module):
    def __init__(self, n_feats, bias=True, act=nn.ReLU(True), res_scale=1, gama=2, lamb=4):
        super(PCSR2, self).__init__()
        # First branch
        self.conv_2 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),)
        self.attention_layer1 = cbam.CSAR_SpatialGate(n_feats, gama=gama)
        self.attention_layer2 = cbam.ChannelGate(n_feats, reduction_ratio=lamb, pool_types=['avg', 'max', 'var'])
        self.res_scale = res_scale
        # Second branch
        self.conv_feature = nn.Sequential(*[nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=1, padding=1),act])
        self.conv_parsing = nn.Sequential(*[nn.Conv2d(in_channels=3, out_channels=n_feats, kernel_size=3, stride=1, padding=1),act])
        self.conv_fusion = nn.Conv2d(in_channels=n_feats * 2, out_channels=n_feats, kernel_size=3, stride=1,padding=1)
        self.attention_fusion = nn.Conv2d(in_channels=n_feats * 2, out_channels=n_feats, kernel_size=1, stride=1)

    def forward(self, x, p):
        # First branch
        res = self.conv_2(x)
        res1 = self.attention_layer1(res)
        res2 = self.attention_layer2(res)
        res = torch.cat((res1, res2), 1)
        res = self.conv_fusion(res)
        # Second branch
        fea = self.conv_feature(x)
        par = self.conv_parsing(p)
        fea = torch.cat((fea, par), 1)
        fea = self.conv_fusion(fea)
        fea_fusion = torch.cat((fea, res), 1)
        res = self.attention_fusion(fea_fusion)
        res += x
        return res

