#Neural net architecture

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #Convolution Block 1 - C1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.10)
        ) # output_size = 32

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.10)
        ) # output_size = 32

        #Adding AntMan with dilated kernel
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=1, dilation=2, stride=2, bias=False),
        ) # output_size = 17

        #Convolution Block 2 - C2
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.10)
        ) # output_size = 17

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.10)
        ) # output_size = 15

        #Adding AntMan with Dilated Kernel
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=1, dilation=2, stride=2, bias=False),
        ) # output_size = 9

        #Convolution Block 3 - C3
        #Convolution layer with Depthwise separable covolution
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, groups=32, bias=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.10)
        ) # output_size = 9

        #Convolution layer with Depthwise separable covolution
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, groups=64, bias=False),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(1,1),bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.10)
        ) # output_size = 9

        #Adding AntMan with dilated kernel
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), padding=1, stride=2, dilation=2, bias=False),
        ) # output_size = 6

        #Convolution Block 4 - C4
        #Convolution layer with Depthwise separable covolution
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, groups=32, bias=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.10)
        ) # output_size = 6

        #Convolution layer with Depthwise separable covolution
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, groups=64, bias=False),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,1), bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.10)
        ) # output_size = 6

        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.10)
        ) # output_size = 4

        #Output Layer - O
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # output_size = 1

        self.conv13 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 1

    def forward(self, x):
        #C1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        #C2
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        #C3
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        #C4
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        #O1
        x = self.gap(x)
        x = self.conv13(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)