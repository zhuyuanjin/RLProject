import torch
from torch import nn
import torch.nn.functional as F

class PongNet(nn.Module):
    def __init__(self):
        super(PongNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=6, stride=4)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.fc = nn.Linear(144, 24)
        self.head = nn.Linear(24,6)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        x = self.head(x)
        return x

    def init_weights(self, std):
        for key in self.state_dict().keys():
            self.state_dict()[key] = torch.nn.init.normal(self.state_dict()[key], 0, std)


