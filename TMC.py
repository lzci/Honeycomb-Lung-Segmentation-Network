import torch.nn.functional as F
import torch.nn as nn
import torch
# MSA
class Mixed_Scal_FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=True):
        super(Mixed_Scal_FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                   groups=hidden_features * 2, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2,
                                   groups=hidden_features * 2, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1,
                                     groups=hidden_features, bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2,
                                     groups=hidden_features, bias=bias)

        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1_3, x2_3 = self.relu3(self.dwconv3x3(x)).chunk(2, dim=1)
        x1_5, x2_5 = self.relu5(self.dwconv5x5(x)).chunk(2, dim=1)

        x1 = x1_3 * self.sigmoid(x1_5)
        x2 = x2_3 * self.sigmoid(x2_5)

        x = torch.cat([x1, x2], dim=1)

        x = self.project_out(x)

        return x

class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        in_x = x.clone()
        x = self.norm(x)
        x = x + self.act(self.pos(x))

        x = self.fc1(x)
        x = self.act(x)

        x = self.fc2(x)

        return x + in_x


class TMergeC(nn.Module):
    def __init__(self, channels, sam_kernel=5, s=2, ratio=16):
        super(TMergeC, self).__init__()
        self.conv = nn.Conv2d(channels * 2, channels + 4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(channels * 2)
        self.qconv = nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2, bias=True, groups=channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.gcn1 = Mixed_Scal_FeedForward(dim=channels)
        self.mlp1 = MLP(channels)
        self.act = nn.GELU()

    def forward(self, t, c):
        # print(t.shape,c.shape)
        B, C, H, W = c.shape
        # tasc = F.interpolate(t, size=c.size()[2:], mode='bilinear')
        # tasc =t
        # print(tasc.shape)
        t1 = t
        # c1 = F.interpolate(c, size=t.size()[2:], mode='bilinear')
        c1 = c
        t1_a = self.gcn1(t1)
        c1_a = self.gcn1(c1)
        fuse = torch.cat((t1_a, c1_a), 1)
        fuse = self.act(self.conv(self.bn(fuse)))

        q, gates = torch.split(fuse, (C, 4), 1)
        q = self.act(self.qconv(q))
        t1_b = t1_a * (gates[:, 0, :, :].unsqueeze(1)) * q
        c1_b = c1_a * (gates[:, 1, :, :].unsqueeze(1)) * q
        # t1_c = t1_b + t1

        # t1_c = F.interpolate(t1_b, size=tasc.size()[2:], mode='bilinear') + tasc
        t1_c = t1_b + t

        # c1_c = F.interpolate(c1_b, size=c.size()[2:], mode='bilinear') + c
        c1_c = c1_b + c
        # c_c = c1_b + c1
        out = self.conv1(self.mlp1(t1_c + c1_c))

        return out