import torch
import torch.nn as nn


class StyleEncoder(nn.Module):
    def __init__(self, spatial_dims, in_channels, style_dim, n_classes=None):
        super().__init__()
        self.n_classes = n_classes
        Conv = getattr(nn, f'Conv{spatial_dims}d')
        Pool = getattr(nn, f'AdaptiveAvgPool{spatial_dims}d')
        if n_classes is not None:
            out_chans = style_dim * n_classes
        else:
            out_chans = style_dim

        self.model = nn.Sequential(
            Conv(in_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.01, inplace=True),
            Pool(1), # Output size: (N, 64, 1, 1)
            nn.Flatten(), # Shape: (N, 64)
            nn.Linear(64, out_chans) # Output style code of dimension 'style_dim'
        )

    def forward(self, x):
        s = self.model(x)
        if self.n_classes is not None:
            s = s.view(s.shape[0], self.n_classes, -1) # (N, n_classes, style_dim)
        else:
            pass # (N, style_dim)
        return s
