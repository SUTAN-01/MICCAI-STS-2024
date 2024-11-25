import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # 定义编码器部分
        self.encoder1 = self._conv_block(in_channels, 64)
        self.encoder2 = self._conv_block(64, 128)
        self.encoder3 = self._conv_block(128, 256)
        self.encoder4 = self._conv_block(256, 512)

        # 中间部分
        self.middle = self._conv_block(512, 1024)

        # 定义解码器部分
        self.upconv4 = self._upconv_block(1024, 512)
        self.upconv3 = self._upconv_block(512, 256)
        self.upconv2 = self._upconv_block(256, 128)
        self.upconv1 = self._upconv_block(128, 64)

        # 定义输出层
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def _upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码器部分
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        e4 = self.encoder4(F.max_pool2d(e3, 2))

        # 中间部分
        m = self.middle(F.max_pool2d(e4, 2))

        # 解码器部分
        d4 = self.upconv4(m)
        d4 = torch.cat([d4, e4], dim=1)  # 拼接跳跃连接
        d4 = self._conv_block(d4.size(1), 512)(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)  # 拼接跳跃连接
        d3 = self._conv_block(d3.size(1), 256)(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)  # 拼接跳跃连接
        d2 = self._conv_block(d2.size(1), 128)(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)  # 拼接跳跃连接
        d1 = self._conv_block(d1.size(1), 64)(d1)

        # 输出层
        out = self.out_conv(d1)
        return out
