import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

"""
Collection of used and tested NN architectures
"""
class Net(nn.Module):
    """
    *NOT IN USE*
    Initial NN architecture tested
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=5, padding=30)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, dilation=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, dilation=3)
        self.bn3 = nn.BatchNorm2d(64)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, dilation=2)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=5)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5)
        self.softmax = nn.Softmax()
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.bn1(self.pool(self.relu(self.conv1(x))))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.bn3(self.relu(self.conv3(x)))
        x = self.upsample(x)
        x = self.bn6(self.relu(self.conv6(x)))
        y1 = self.conv4(x)
        y1 = self.softmax(y1)
        y2 = self.conv5(x)
        y2 = F.sigmoid(y2)
        return y1, y2


class TrainDAN(nn.Module):
    """
    *NOT IN USE*
    Domain Adaptation Network:
    https://arxiv.org/pdf/1709.00179.pdf
    and
    https://arxiv.org/pdf/1409.7495.pdf
    """

    def __init__(self):
        super().__init__()
        self.general_features = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=5, padding=30),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, dilation=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, dilation=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, dilation=3),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, dilation=3),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, dilation=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, dilation=2),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, dilation=2),
            nn.BatchNorm2d(16)
        )
        self.segm = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=2, kernel_size=5), nn.Softmax())
        self.gauss = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=1, kernel_size=5), nn.Sigmoid())
        self.dan = nn.Sequential(GradReverse(), nn.Linear(147456, 320), nn.BatchNorm1d(320), nn.Linear(320, 1),
                                 nn.Sigmoid())

    def forward(self, x):
        features = self.general_features(x)
        gaus = self.gauss(features)
        seg = self.segm(features)
        dan = self.dan(features)
        return features, seg, gaus, dan


class TestDAN(TrainDAN):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        features = self.general_features(x)
        gaus = self.gauss(features)
        seg = self.segm(features)
        return gaus, seg


class NetExperiment(nn.Module):
    """
    NN architecture used in this project. Based on paper:
    https://arxiv.org/pdf/1709.00179.pdf
    """

    def __init__(self):
        super(NetExperiment, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=5, padding=30)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, dilation=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, dilation=2)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, dilation=3)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, dilation=3)
        self.bn5 = nn.BatchNorm2d(32)

        self.conv6 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, dilation=2)
        self.bn6 = nn.BatchNorm2d(16)

        self.conv7 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, dilation=2)
        self.bn7 = nn.BatchNorm2d(16)

        self.conv8 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=5)

        self.conv9 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=5)

        self.softmax = nn.Softmax(dim=1)

        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.bn3(self.relu(self.conv3(x)))
        x = self.bn4(self.relu(self.conv4(x)))
        x = self.bn5(self.relu(self.conv5(x)))
        x = self.bn6(self.relu(self.conv6(x)))
        x = self.bn7(self.relu(self.conv7(x)))

        y1 = self.conv8(x)
        y1 = self.softmax(y1)
        y2 = self.conv9(x)
        y2 = self.sigmoid(y2)
        return y1, y2


class GradReverse(Function):
    def __init__(self):
        super(Function, self).__init__()

    def forward(self, x, **kwargs):
        return x.view_as(x)

    def backward(self, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)



class NetTracking(nn.Module):
    """
    *NOT IN USE*
    """
    def __init__(self):
        super(NetTracking, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=5, padding=28)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, dilation=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, dilation=3)
        self.bn3 = nn.BatchNorm2d(64)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=5)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5)
        self.softmax = nn.Softmax()
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.bn1(self.pool(self.relu(self.conv1(x))))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.bn3(self.relu(self.conv3(x)))
        x = self.upsample(x)
        y1 = self.conv4(x)
        y1 = self.softmax(y1)
        y2 = self.conv5(x)
        y2 = self.sigmoid(y2)
        return y1, y2