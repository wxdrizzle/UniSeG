import torch
from torch import nn


class ConvDomainDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        if in_channels <= 128:
            chan = 128 if in_channels >= 64 else 64
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, chan, kernel_size=3, stride=2, padding=1), # Downsample by 2
                nn.BatchNorm2d(chan),
                nn.ReLU(),
                nn.Conv2d(chan, 128, kernel_size=3, stride=2, padding=1), # Downsample by 2
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # Downsample by 2
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), # Output shape: (N, 256, 1, 1)
                nn.Flatten(), # Shape: (N, 256)
                nn.Linear(256, 1),
                nn.Sigmoid())
        else:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), # Output shape: (N, 256, 1, 1)
                nn.Flatten(), # Shape: (N, 256)
                nn.Linear(256, 1),
                nn.Sigmoid())

    def forward(self, x):
        return self.model(x)


# class SegmentationDiscriminator(nn.Module):
#     def __init__(self, num_classes):
#         super(SegmentationDiscriminator, self).__init__()
#         self.model = nn.Sequential(
#             # Input shape: (N, num_classes, H, W)
#             nn.Conv2d(num_classes, 64, kernel_size=4, stride=2, padding=1), # Downsample by 2
#             # nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # Downsample by 2
#             # nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # Downsample by 2
#             # nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # Downsample by 2
#             # nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.AdaptiveAvgPool2d(1), # Output shape: (N, 512, 1, 1)
#             nn.Flatten(), # Shape: (N, 512)
#             nn.Linear(512, 1),
#             nn.Sigmoid() # Output probability between 0 and 1
#         )
#
#     def forward(self, x):
#         return self.model(x)


class SegmentationDiscriminator(nn.Module):
    def __init__(self, num_classes, ndf=64):
        super().__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        # 上采样至与输入相同大小
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        x = self.up_sample(x)
        #x = self.sigmoid(x)
        return x
