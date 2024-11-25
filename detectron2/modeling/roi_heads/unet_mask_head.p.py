from detectron2.modeling import ROI_MASK_HEAD_REGISTRY
import torch
import torch.nn as nn

class UNetMaskHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetMaskHead, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        self.middle = self.conv_block(512, 1024)

        self.upconv4 = self.upconv_block(1024, 512)
        self.upconv3 = self.upconv_block(512, 256)
        self.upconv2 = self.upconv_block(256, 128)
        self.upconv1 = self.upconv_block(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(nn.functional.max_pool2d(enc1, 2))
        enc3 = self.encoder3(nn.functional.max_pool2d(enc2, 2))
        enc4 = self.encoder4(nn.functional.max_pool2d(enc3, 2))

        mid = self.middle(nn.functional.max_pool2d(enc4, 2))

        up4 = self.upconv4(mid)
        up4 = torch.cat([up4, enc4], dim=1)
        up3 = self.upconv3(up4)
        up3 = torch.cat([up3, enc3], dim=1)
        up2 = self.upconv2(up3)
        up2 = torch.cat([up2, enc2], dim=1)
        up1 = self.upconv1(up2)
        up1 = torch.cat([up1, enc1], dim=1)

        return self.out_conv(up1)

@ROI_MASK_HEAD_REGISTRY.register()
class UNetMaskHeadWrapper(nn.Module):
    def __init__(self, cfg, input_shape):
        super(UNetMaskHeadWrapper, self).__init__()
        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.mask_head = UNetMaskHead(in_channels=input_shape.channels, out_channels=num_classes)

    def forward(self, features):
        return self.mask_head(features['mask'])
