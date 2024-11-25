import torch
import torch.nn as nn
import torch.nn.functional as F


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2

        # Define the layers
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.W = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)

    def forward(self, x):
        batch_size, C, H, W = x.size()

        # Generate theta, phi, and g
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        # Compute attention map
        theta_x = theta_x.permute(0, 2, 1)
        attention = F.softmax(torch.bmm(theta_x, phi_x), dim=-1)
        out = torch.bmm(g_x, attention)
        out = out.view(batch_size, self.inter_channels, H, W)
        out = self.W(out)
        return x + out
