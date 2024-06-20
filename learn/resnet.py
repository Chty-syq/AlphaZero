import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()
        self.n_actions = args.n_actions

        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv_p = nn.Conv2d(128, 4, kernel_size=1)
        self.conv_v = nn.Conv2d(128, 2, kernel_size=1)
        self.fc_p1 = nn.Linear(4 * self.n_actions, self.n_actions)
        self.fc_v1 = nn.Linear(2 * self.n_actions, 64)
        self.fc_v2 = nn.Linear(64, 1)

    def get_feature(self, state):
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

    def forward(self, state):
        feature = self.get_feature(state)

        probs = F.relu(self.conv_p(feature))
        probs = probs.view(-1, 4 * self.n_actions)
        probs = F.log_softmax(self.fc_p1(probs), dim=1)

        value = F.relu(self.conv_v(feature))
        value = value.view(-1, 2 * self.n_actions)
        value = F.relu(self.fc_v1(value))
        value = F.tanh(self.fc_v2(value))

        return probs, value
