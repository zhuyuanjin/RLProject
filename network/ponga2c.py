from torch import nn
from torch.nn import functional as F

class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 6)
        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x)



class Value(nn.Module):

    def __init__(self):
        super(Value,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 6)
        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)
        self.head = nn.Linear(6,1)

    def forward(self):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.head(x)
        return value



