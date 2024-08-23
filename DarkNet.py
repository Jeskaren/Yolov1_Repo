# -*- coding:utf-8 -*-
# 一万年太久，只争朝夕
# Jeskaren
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义 Darknet 网络模型
class darknet(nn.Module):
    def __init__(self):
        super(darknet, self).__init__()

        self.leaky_relu = nn.LeakyReLU(inplace=True, negative_slope=0.1)

        self.conv1 = self._conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = self._conv_block(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = self._conv_block(192, 128, kernel_size=1, stride=1, padding=0)
        self.conv4 = self._conv_block(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = self._conv_block(256, 256, kernel_size=1, stride=1, padding=0)
        self.conv6 = self._conv_block(256, 512, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = self._conv_block(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv8 = self._conv_block(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv9 = self._conv_block(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv10 = self._conv_block(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv11 = self._conv_block(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv12 = self._conv_block(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv13 = self._conv_block(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv14 = self._conv_block(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv15 = self._conv_block(512, 512, kernel_size=1, stride=1, padding=0)
        self.conv16 = self._conv_block(512, 1024, kernel_size=3, stride=1, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv17 = self._conv_block(1024, 512, kernel_size=1, stride=1, padding=0)
        self.conv18 = self._conv_block(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv19 = self._conv_block(1024, 512, kernel_size=1, stride=1, padding=0)
        self.conv20 = self._conv_block(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv21 = self._conv_block(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.conv22 = self._conv_block(1024, 1024, kernel_size=3, stride=2, padding=1)

        self.conv23 = self._conv_block(1024, 1024, kernel_size=1, stride=1, padding=0)
        self.conv24 = self._conv_block(1024, 1024, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(7*7*1024, 4096)
        self.fc2 = nn.Linear(4096, 7*7*30)



    # 定义卷积层块的结构
    def _conv_block(self, in_channels, out_channels, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            self.leaky_relu
        )

    # 前向传播过程
    def forward(self, x):

        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool3(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.maxpool4(x)
        x = self.conv17(x)
        x = self.conv18(x)
        x = self.conv19(x)
        x = self.conv20(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.conv23(x)
        x = self.conv24(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = x.view(x.size(0), 7, 7, 30)

        return x

if __name__ == '__main__':
    model = darknet()
    print(model)