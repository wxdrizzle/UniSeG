import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.dynunet_block import UnetBasicBlock


class Reg(nn.Module):
    def __init__(self, spatial_dims, in_channels, n_blocks):
        super().__init__()

        blocks = []
        for i in range(n_blocks):
            blocks.append(
                UnetBasicBlock(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=in_channels,
                               kernel_size=3, stride=1, norm_name=('batch', {
                                   'num_features': in_channels
                               })))
        Conv = getattr(nn, f'Conv{spatial_dims}d')
        blocks.append(Conv(in_channels, 2 * spatial_dims, kernel_size=3, padding='same'))
        self.blocks = nn.Sequential(*blocks)
        nn.init.constant_(self.blocks[-1].weight, 0)
        nn.init.constant_(self.blocks[-1].bias, 0)

    def forward(self, x):
        return self.blocks(x)
