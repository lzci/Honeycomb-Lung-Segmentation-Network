import torch.nn as nn
import torch
from models4.md10.p2t import *
from models4.md10.CPCA import CPCA
from models4.md10.TMC import TMergeC


# [-1,512,64,64] [-1,512,64,64]  torch.cat ->  [-1,1024,64,64]
# [-1,512,64,64] [-1,512,64,64]      +     ->  [-1,512,64,64]

class UnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn2(out)

        return out


# class Spartial_Attention(nn.Module):
#     def __init__(self, kernel_size=5):
#         super(Spartial_Attention, self).__init__()
#         assert kernel_size % 2 == 1, "kernel_size = {}".format(kernel_size)
#         padding = (kernel_size - 1) // 2
#         self.__layer = nn.Sequential(
#             # nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding),
#             nn.Conv2d(2, 2, kernel_size=3, groups=2, padding=1),
#             nn.Conv2d(2, 1, kernel_size=3, padding=2, dilation=2),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         avg_mask = torch.mean(x, dim=1, keepdim=True)
#         max_mask, _ = torch.max(x, dim=1, keepdim=True)
#         mask = torch.cat([avg_mask, max_mask], dim=1)
#         mask = self.__layer(mask)
#         return mask


# class Channel_Attention(nn.Module):
#     def __init__(self, channel1, channel2, r=16):
#         super(Channel_Attention, self).__init__()
#
#         self.__avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.__max_pool = nn.AdaptiveMaxPool2d((1, 1))
#
#         self.__fc = nn.Sequential(
#             nn.Conv2d(channel1, channel1 // r, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(channel1 // r, channel1, 1, bias=False),
#         )
#         self.__sigmoid = nn.Sigmoid()
#         self.cv1 = nn.Conv2d(channel1, channel2, 1)
#
#     def forward(self, x):
#         # print(x.shape)
#         y1 = self.__avg_pool(x)
#         # print(y1.shape)
#         y1 = self.__fc(y1)
#         # print(y1.shape)
#         y2 = self.__max_pool(x)
#         # print(y2.shape)
#         y2 = self.__fc(y2)
#         # print(y2.shape)
#
#         y = self.__sigmoid(y1 + y2)
#         return self.cv1(x * y)


# class TMergeC(nn.Module):
#     def __init__(self,in_channels, sam_kernel=5, s=2, ratio=16):
#         super(TMergeC, self).__init__()
#         self.upp2t = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
#         self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding=0)
#     def forward(self, t, c):
#         t_out = self.upp2t(t)
#         out = t_out + c
#         return out




class md10(nn.Module):
    def __init__(self, output_channels=2, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [64, 128, 256, 512, 1024]

        self.backbone = p2t_base()
        self.tmc1 = TMergeC(nb_filter[0])
        self.tmc2 = TMergeC(nb_filter[1])
        self.tmc3 = TMergeC(nb_filter[2])
        self.tmc4 = TMergeC(nb_filter[3])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.poola = nn.MaxPool2d(kernel_size=4, stride=4)
        self.up_1 = nn.ConvTranspose2d(nb_filter[3], nb_filter[3], kernel_size=1, stride=1)
        self.up_bn1 = nn.BatchNorm2d(nb_filter[3])
        self.up_2 = nn.ConvTranspose2d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)
        self.up_bn2 = nn.BatchNorm2d(nb_filter[2])
        self.up_3 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up_bn3 = nn.BatchNorm2d(nb_filter[1])
        self.up_4 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.up_bn4 = nn.BatchNorm2d(nb_filter[0])

        self.conv0_0 = UnetBlock(input_channels, nb_filter[0])
        self.conv1_0 = UnetBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = UnetBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = UnetBlock(nb_filter[2], nb_filter[3])
        # self.conv4_0 = UnetBlock(nb_filter[3], nb_filter[4])

        self.conv3_1 = UnetBlock(nb_filter[4], nb_filter[3])
        self.conv2_2 = UnetBlock(nb_filter[3], nb_filter[2])
        self.conv1_3 = UnetBlock(nb_filter[2], nb_filter[1])
        self.conv0_4 = UnetBlock(nb_filter[1], nb_filter[0])
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.final = nn.Conv2d(nb_filter[0], output_channels, kernel_size=3, padding=1)

    def forward(self, input):
        p2t = self.backbone(input)
        p2t1 = p2t[0]  # 1 64 64 64
        p2t2 = p2t[1]  # 1 128 32 32
        p2t3 = p2t[2]  # 1 256 16 16
        p2t4 = p2t[3]  # 1 512 8 8
        # print(p2t1.shape,p2t2.shape,p2t3.shape,p2t4.shape)
        x0_0 = self.poola(self.conv0_0(input))
        x0_0 = self.tmc1(p2t1, x0_0)  #  # 1, 64, 64, 64
        # print(x0_0.shape)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x1_0 = self.tmc2(p2t2, x1_0)  # 1, 128, 32, 32
        # print(x1_0.shape)
        x2_0 = self.conv2_0(self.pool(x1_0))
        x2_0 = self.tmc3(p2t3, x2_0)  # 1, 256, 16, 16
        # print(x2_0.shape)
        x3_0 = self.conv3_0(self.pool(x2_0))
        x3_0 = self.tmc4(p2t4, x3_0)  # 1, 512, 8, 8
        # print(x3_0.shape)


        x3_1 = self.conv3_1(torch.cat([x3_0, self.up_bn1(self.up_1(x3_0))], 1))  # 1, 512, 8, 8
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up_bn2(self.up_2(x3_1))], 1))  # 1, 256, 16, 16
        # print(x2_2.shape)
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up_bn3(self.up_3(x2_2))], 1))  # 1, 128, 32, 32
        # print(x1_3.shape)
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up_bn4(self.up_4(x1_3))], 1))  # 1, 64, 64, 64
        # print(x0_4.shape)

        # x0_4 = self.cpca(x0_4)
        # print(x0_4.shape)
        x0_4 = self.up(x0_4)
        output = self.final(x0_4)
        # print(output.shape)
        output = nn.Sigmoid()(output)

        return output


# torch.cuda.set_device(1)
#
# model = md10().cuda()
# from ptflops import get_model_complexity_info
#
# flops, params = get_model_complexity_info(model, input_res=(3, 256, 256), as_strings=True,
#                                           print_per_layer_stat=False)
# print('      - Flops:  ' + flops)
# print('      - Params: ' + params)
# - Flops: 15.62 GMac
# - Params: 58.08 M
