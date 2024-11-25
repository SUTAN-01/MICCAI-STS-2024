# channel_attention.py

import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // 16, bias=False)
        self.fc2 = nn.Linear(in_channels // 16, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.fc1(self.avg_pool(x).view(x.size(0), -1)))
        max_out = self.fc2(self.fc1(self.max_pool(x).view(x.size(0), -1)))
        out = avg_out + max_out
        return x * self.sigmoid(out.view(x.size(0), -1, 1, 1))
