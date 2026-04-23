import torch
import torch.nn as nn
from monai.networks.blocks import dynunet_block
from collections.abc import Sequence
import numpy as np
import torch

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.utils import get_act_layer, get_norm_layer


class UnetBasicBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {
            "inplace": True,
            "negative_slope": 0.01
        }),
        dropout: tuple | str | float | None = None,
    ):
        super().__init__()
        self.conv1 = dynunet_block.get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.conv2 = dynunet_block.get_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm_name = norm_name
        if norm_name[0] == 'AdaptiveInstanceNorm':
            self.norm1 = AdaINLayer(spatial_dims, out_channels, norm_name[1]['style_dim'])
            self.norm2 = AdaINLayer(spatial_dims, out_channels, norm_name[1]['style_dim'])
        else:
            self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
            self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp, style_code=None):
        out = self.conv1(inp)
        if style_code is not None:
            assert self.norm_name[0] == 'AdaptiveInstanceNorm'
            out = self.norm1(out, style_code)
        else:
            out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        if style_code is not None:
            assert self.norm_name[0] == 'AdaptiveInstanceNorm'
            out = self.norm2(out, style_code)
        else:
            out = self.norm2(out)
        out = self.lrelu(out)
        return out


class AdaINLayer(nn.Module):
    def __init__(self, spatial_dims, n_channels, style_dim):
        super().__init__()
        if spatial_dims == 2:
            self.instance_norm = nn.InstanceNorm2d(n_channels, affine=False)
        elif spatial_dims == 3:
            self.instance_norm = nn.InstanceNorm3d(n_channels, affine=False)
        else:
            raise ValueError("spatial_dims can only be 2 or 3")
        self.style_scale_transform = nn.Linear(style_dim, n_channels)
        self.style_shift_transform = nn.Linear(style_dim, n_channels)

    def forward(self, x, style_code):
        # x: (N, C, ...) or (N, C, ...)
        assert x.dim() == 4 or x.dim() == 5
        x_norm = self.instance_norm(x)

        if isinstance(style_code, torch.Tensor):
            gamma = self.style_scale_transform(style_code) # (N, C)
            beta = self.style_shift_transform(style_code) # (N, C)
            gamma = gamma[:, :, *[None] * (x.dim() - 2)] # (N, C, ...)
            beta = beta[:, :, *[None] * (x.dim() - 2)] # (N, C, ...)
        else:
            assert isinstance(style_code, dict)
            mask = style_code['mask'] # (N, K, ...)
            code = style_code['code'] # (N, K, style_dim)
            gamma = self.style_scale_transform(code) # (N, K, C)
            beta = self.style_shift_transform(code) # (N, K, C)

            mask = mask[:, :, None] # (N, K, 1, ...)
            gamma = gamma[:, :, :, *[None] * (mask.dim() - 3)] # (N, K, C, ...)
            beta = beta[:, :, :, *[None] * (mask.dim() - 3)] # (N, K, C, ...)
            gamma = (mask * gamma).sum(dim=1) # (N, C, ...)
            beta = (mask * beta).sum(dim=1) # (N, C, ...)

        # Modulate normalized activations
        out = gamma*x_norm + beta
        return out


class UpBlock(nn.Module):
    def __init__(self, cfg, role, spatial_dims: int, in_channels: int, out_channels: int):
        super().__init__()
        self.cfg = cfg
        self.role = role
        if spatial_dims == 2:
            cls_conv = nn.Conv2d
            interp_mode = 'bilinear'
        else:
            cls_conv = nn.Conv3d
            interp_mode = 'trilinear'

        assert cfg.net[role].structure.name == 'unet_decoder'
        kernel_size = cfg.net[role].structure.unet_decoder.kernel_size

        self.up = nn.Sequential(
            cls_conv(in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=False, padding='same'),
            nn.Upsample(scale_factor=2, mode=interp_mode),
        )

        if cfg.net.style_code.enable and role == 'decoder_rec':
            if cfg.net.style_code.mode == 'AdaIN':
                norm_name = ('AdaptiveInstanceNorm', {'style_dim': cfg.net.style_code.adain.style_dim})
            else:
                raise NotImplementedError
        else:
            norm_name = ('batch', {'num_features': out_channels})

        self.conv_block = UnetBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
            act_name=("leakyrelu", {
                "inplace": True,
                "negative_slope": 0.01
            }),
            dropout=None,
        )

    def forward(self, x, skip, style_code=None):
        # x: [B, Ci, H, W]
        # skip: [B, Co, 2H, 2W]
        # style_code: [B, style_dim]
        # return: [B, Co, 2H, 2W]
        assert x.dim() in [4, 5]
        x = self.up(x)
        x = x + skip
        if isinstance(style_code, dict):
            mask = style_code['mask'] # (N, K, ...)
            if mask.shape[2] != x.shape[2]:
                if x.dim() == 4:
                    mode = 'bilinear'
                elif x.dim() == 5:
                    mode = 'trilinear'
                mask = torch.nn.functional.interpolate(mask, size=x.shape[2:], mode=mode)
            style_code['mask'] = mask

        x = self.conv_block(x, style_code)
        return x


class MyDecoder(nn.Module):
    def __init__(self, cfg, role, spatial_dims, channels, out_channels):
        super().__init__()
        self.role = role
        self.level2upblock = nn.ModuleDict()
        for level, channel in enumerate(channels[:-1]):
            self.level2upblock[str(level)] = UpBlock(cfg, role, spatial_dims, channels[level + 1], channel)

        self.out_conv = getattr(nn, f'Conv{spatial_dims}d')(channels[0], out_channels, kernel_size=1)

    def forward(self, level2skip, style_code=None):
        levels = [int(x) for x in level2skip.keys()]
        levels.sort(reverse=True)
        levels = [str(l) for l in levels]

        x = level2skip[levels[0]]
        for level in levels[1:]:
            skip = level2skip[level]
            x = self.level2upblock[level](x, skip, style_code)
        x = self.out_conv(x)
        return x
