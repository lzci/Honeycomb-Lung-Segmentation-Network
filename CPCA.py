from torch import nn
import torch
import numpy as np

import torch.nn.functional
from timm.models.layers import DropPath, to_3tuple, trunc_normal_
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from thop import profile

class ChannelAttention(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))  # 1 64 1 1
        # print(x1.shape)
        # print('x:', x.shape)
        x1 = self.fc1(x1)  # 1 16 1 1
        # print(x1.shape)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)  # 1 16 1 1
        # print(x1.shape)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))  # 1 64 1 1
        # print(x2.shape)
        # print('x:', x.shape)
        x2 = self.fc1(x2)  # 1 16 1 1
        # print(x2.shape)
        x2 = F.relu(x2, inplace=True)  # 1 16 1 1
        x2 = self.fc2(x2)  # 1 16 1 1
        x2 = torch.sigmoid(x2)  # 1 16 1 1
        x = x1 + x2  # 1 16 1 1
        # print(x.shape)
        x = x.view(-1, self.input_channels, 1, 1)  # 1 64 1 1
        # print(x.shape)
        return x

class RepBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 channelAttention_reduce=4):
        super().__init__()

        self.C = in_channels
        self.O = out_channels

        assert in_channels == out_channels
        self.ca = ChannelAttention(input_channels=in_channels, internal_neurons=in_channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels)
        self.dconv11_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels)
        self.dconv1_21 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels)
        self.dconv21_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        #   Global Perceptron
        inputs = self.conv(inputs)  # 1 64 256 256
        # print(inputs.shape)
        inputs = self.act(inputs)  # 1 64 256 256
        # print(inputs.shape)

        channel_att_vec = self.ca(inputs)  # 1 64 1 1
        # print(channel_att_vec.shape)
        inputs = channel_att_vec * inputs  # 1 64 256 256

        x_init = self.dconv5_5(inputs)  # 1 64 256 256
        # print(x_init.shape)
        x_1 = self.dconv1_7(x_init)  # 1 64 256 256
        x_1 = self.dconv7_1(x_1)  # 1 64 256 256
        x_2 = self.dconv1_11(x_init)  # 1 64 256 256
        x_2 = self.dconv11_1(x_2)  # 1 64 256 256
        x_3 = self.dconv1_21(x_init)  # 1 64 256 256
        x_3 = self.dconv21_1(x_3)  # 1 64 256 256
        x = x_1 + x_2 + x_3 + x_init  # 1 64 256 256
        # print(x.shape)
        spatial_att = self.conv(x)  # 1 64 256 256
        # print(spatial_att.shape)
        out = spatial_att * inputs  # 1 64 256 256
        # print(out.shape)
        out = self.conv(out)  # 1 64 256 256
        return out

class FFNBlock2(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_channels or in_channels
        hidden_features = hidden_channels or in_channels
        self.conv1 = nn.Conv2d(in_channels, hidden_features, kernel_size=(1, 1), padding=0)
        self.conv2 = nn.Conv2d(hidden_channels, out_features, kernel_size=(1, 1), padding=0)
        self.dconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3, 3), padding=(1, 1),
                               groups=hidden_features)
        self.act = act_layer()

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)  # 1 64 256 256
        print(x.shape)
        x = self.dconv(x)
        x = self.act(x)
        x = self.conv2(x)
        return x

class CPCA(nn.Module):
    def __init__(self, dim, drop_path=0., ffn_expand=4, channelAttention_reduce=4):
        super().__init__()
        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.bn = nn.BatchNorm2d(dim)
        # self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.ffn_block = FFNBlock2(dim, dim * ffn_expand)
        self.repmlp_block = RepBlock(in_channels=dim, out_channels=dim, channelAttention_reduce=channelAttention_reduce)

    def forward(self, x):
        input = x.clone()  # 1 64 256 256
        # print(input.shape)

        x = self.bn(x)  # 1 64 256 256
        # print(x.shape)
        x = self.repmlp_block(x)  # 1 64 256 256
        # print(x.shape)
        x = input + self.drop_path(x)  # 1 64 256 256
        # x2 = self.bn(x)  # 1 64 256 256
        # x2 = self.ffn_block(x2)
        # x = x + self.drop_path(x2)

        return x

# import models3.md1.Config as Config
# torch.cuda.set_device(1)
# config_vit = Config.get_CTranS_config()
# model = Block(64).cuda()
# # # input_tensor = torch.randn(1, 3, 256, 256).cuda()
# # # logits = model(input_tensor)
# # # print(logits.shape)
# from ptflops import get_model_complexity_info
# flops, params = get_model_complexity_info(model, input_res=(64, 256, 256), as_strings=True,
#                                               print_per_layer_stat=False)
# print('      - Flops:  ' + flops)
# print('      - Params: ' + params)