import torchvision
import torch.nn as nn


class DeconvBlock(nn.Module):
    
    def __init__(self, in_features, out_features, deconv_kernel, deconv_stride=2, deconv_pad=1):
        super().__init__()
        self.offset_conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=2 * 3 * 3,
            kernel_size=3,
            padding=1
        )
        self.deformconv = torchvision.ops.DeformConv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=3,
            padding=1
        )
        self.batchnorm1 = nn.BatchNorm2d(out_features)
        self.deconv = nn.ConvTranspose2d(
            in_channels=out_features,
            out_channels=out_features,
            kernel_size=deconv_kernel,
            stride=deconv_stride,
            padding=deconv_pad,
            bias=False
        )
        self.batchnorm2 = nn.BatchNorm2d(out_features)
        self.silu = nn.SiLU()
        
    def forward(self, x):
        offset = self.offset_conv(x)
        x = self.deformconv(x, offset)
        x = self.batchnorm1(x)
        x = self.silu(x)
        x = self.deconv(x)
        x = self.batchnorm2(x)
        x = self.silu(x)
        return x
        
class DeconvLayers(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.deconv_channels = cfg.MODEL.CENTERNET.DECONV_CHANNEL
        self.deconv_kernels = cfg.MODEL.CENTERNET.DECONV_KERNEL
        self.deconv1 = DeconvBlock(
            in_features=self.deconv_channels[0],
            out_features=self.deconv_channels[1],
            deconv_kernel=self.deconv_kernels[0]
        )
        self.deconv2 = DeconvBlock(
            in_features=self.deconv_channels[1],
            out_features=self.deconv_channels[2],
            deconv_kernel=self.deconv_kernels[1]
        )
        self.deconv3 = DeconvBlock(
            in_features=self.deconv_channels[2],
            out_features=self.deconv_channels[3],
            deconv_kernel=self.deconv_kernels[2]
        )
        
    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x