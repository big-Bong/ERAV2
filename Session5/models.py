#File for defining our neural network module

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    #This defines the structure of the neural network.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) #1st conv layer: 1 input channel, 32 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) #2nd conv layer: 32 input channels, 64 output channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3) #3rd conv layer: 64 input channels, 128 output channesl
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3) #4th conv layer: 128 input channels, 256 output channels
        self.fc1 = nn.Linear(4096, 50) #1st fully connected layers: 256*4*4 input channels, 50 output channels
        self.fc2 = nn.Linear(50, 10) #2nd fully connected layers: 50 input channels, 10 output channels

    def forward(self, x):
        x = F.relu(self.conv1(x), 2) # 28>26 | 1>3 | 1>1
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) #26>24>12 | 3>5>6 | 1>1>2
        x = F.relu(self.conv3(x), 2) # 12>10 | 6>10 | 2>2
        x = F.relu(F.max_pool2d(self.conv4(x), 2)) # 10>8>4 | 10>14>16 | 2>2>4
        x = x.view(-1, 4096) # 4*4*256 = 4096
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

