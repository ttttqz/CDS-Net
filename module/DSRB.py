import torch.nn as nn
import torch.nn.functional as F
from module.AGCA import AGCA


class DSRB(nn.Module):
    def __init__(self, channels, kernel_dim=1, dilation=1):
        super(DSRB, self).__init__()
        self.pad = kernel_dim + dilation
        self.border_input = kernel_dim + 2 * dilation + 1
        self.sigmod = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.agca = AGCA(in_channel=channels, ratio=4)

        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)
        self.conv2 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)
        self.conv3 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)
        self.conv4 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)

    def forward(self, x):
        tmp = x
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        x1 = self.conv1(x[:, :, :-self.border_input, :-self.border_input])
        x2 = self.conv2(x[:, :, self.border_input:, :-self.border_input])
        x3 = self.conv3(x[:, :, :-self.border_input, self.border_input:])
        x4 = self.conv4(x[:, :, self.border_input:, self.border_input:])
        res = 2 * self.sigmod(tmp - (x1 + x2 + x3 + x4) / 4) - 1
        x = self.relu(tmp * res)
        x = self.agca(x)
        return x
