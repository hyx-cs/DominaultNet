import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=(kernel_size // 2), bias=bias)
class ParsingNet(nn.Module):
    def __init__(self, conv=default_conv):
        super(ParsingNet, self).__init__()

        n_resblocks = 8
        n_feats = 64
        kernel_size = 3
        act = nn.ReLU(True)
        m_head = [conv(in_channels=3, out_channels=n_feats, kernel_size=kernel_size)]
        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=1
            ) for _ in range(n_resblocks)
        ]

        m_feature = [

            conv(n_feats, 3, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.feature = nn.Sequential(*m_feature)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        feature = self.feature(res)
        return feature
