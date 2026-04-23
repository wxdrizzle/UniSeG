from collections.abc import Sequence
import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Norm


class ConvBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        strides: int = 1,
        dropout=0.0,
        norm=Norm.BATCH,
    ):
        super().__init__()
        layers = [
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=None,
                adn_ordering="NDA",
                act="relu",
                norm=norm,
                dropout=dropout,
            ),
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=1,
                padding=None,
                adn_ordering="NDA",
                act="relu",
                norm=norm,
                dropout=dropout,
            ),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c: torch.Tensor = self.conv(x)
        return x_c


class UpConv(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, kernel_size=3, strides=2, dropout=0.0,
                 norm=Norm.BATCH):
        super().__init__()
        self.up = Convolution(
            spatial_dims,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=kernel_size,
            act="relu",
            adn_ordering="NDA",
            norm=norm,
            dropout=dropout,
            is_transposed=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_u: torch.Tensor = self.up(x)
        return x_u


class AttentionBlock(nn.Module):
    def __init__(self, spatial_dims: int, f_int: int, f_g: int, f_l: int, dropout=0.0, norm=Norm.BATCH):
        super().__init__()
        self.W_g = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_g,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[norm, spatial_dims](f_int),
        )

        self.W_x = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_l,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[norm, spatial_dims](f_int),
        )

        self.psi = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_int,
                out_channels=1,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[norm, spatial_dims](1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi: torch.Tensor = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttentionLayer(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        submodule: nn.Module,
        up_kernel_size=3,
        strides=2,
        dropout=0.0,
        norm=Norm.BATCH,
    ):
        super().__init__()
        self.attention = AttentionBlock(spatial_dims=spatial_dims, f_g=in_channels, f_l=in_channels,
                                        f_int=in_channels // 2, norm=norm)
        self.upconv = UpConv(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=in_channels,
            strides=strides,
            kernel_size=up_kernel_size,
            norm=norm,
        )
        self.merge = Convolution(spatial_dims=spatial_dims, in_channels=2 * in_channels, out_channels=in_channels,
                                 dropout=dropout, norm=norm)
        self.submodule = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fromlower = self.upconv(self.submodule(x))
        att = self.attention(g=fromlower, x=x)
        att_m: torch.Tensor = self.merge(torch.cat((att, fromlower), dim=1))
        return att_m


class MyAttentionUnet(nn.Module):
    """
    Attention Unet based on
    Otkay et al. "Attention U-Net: Learning Where to Look for the Pancreas"
    https://arxiv.org/abs/1804.03999

    Args:
        spatial_dims: number of spatial dimensions of the input image.
        in_channels: number of the input channel.
        out_channels: number of the output classes.
        channels (Sequence[int]): sequence of channels. Top block first. The length of `channels` should be no less than 2.
        strides (Sequence[int]): stride to use for convolutions.
        kernel_size: convolution kernel size.
        up_kernel_size: convolution kernel size for transposed convolution layers.
        dropout: dropout ratio. Defaults to no dropout.
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Sequence[int] | int = 3,
        up_kernel_size: Sequence[int] | int = 3,
        dropout: float = 0.0,
        norm='batch',
    ):
        super().__init__()
        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.dropout = dropout
        if norm == 'batch':
            norm = Norm.BATCH
        elif norm == 'instance':
            norm = Norm.INSTANCE
        else:
            raise ValueError(f"unsupported norm type: {norm}")
        self.norm = norm

        head = ConvBlock(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=channels[0], dropout=dropout,
                         norm=norm)
        self.up_kernel_size = up_kernel_size

        def _create_block(channels: Sequence[int], strides: Sequence[int]) -> nn.Module:
            if len(channels) > 2:
                subblock = _create_block(channels[1:], strides[1:])
                return AttentionLayer(
                    spatial_dims=spatial_dims,
                    in_channels=channels[0],
                    out_channels=channels[1],
                    submodule=nn.Sequential(
                        ConvBlock(
                            spatial_dims=spatial_dims,
                            in_channels=channels[0],
                            out_channels=channels[1],
                            strides=strides[0],
                            dropout=self.dropout,
                            norm=norm,
                        ),
                        subblock,
                    ),
                    up_kernel_size=self.up_kernel_size,
                    strides=strides[0],
                    dropout=dropout,
                    norm=norm,
                )
            else:
                # the next layer is the bottom so stop recursion,
                # create the bottom layer as the subblock for this layer
                return self._get_bottom_layer(channels[0], channels[1], strides[0])

        encdec = _create_block(self.channels, self.strides)
        self.model = nn.Sequential(head, encdec)
        self.init_output_hook()

    def _get_bottom_layer(self, in_channels: int, out_channels: int, strides: int) -> nn.Module:
        return AttentionLayer(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            submodule=ConvBlock(
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                dropout=self.dropout,
                norm=self.norm,
            ),
            up_kernel_size=self.up_kernel_size,
            strides=strides,
            dropout=self.dropout,
            norm=self.norm,
        )

    def init_output_hook(self):
        assert not hasattr(self, "output")
        self.id_module2output = {}
        self.id_module2level = {}

        for name, module in self.named_modules():
            if isinstance(module, AttentionLayer):
                # 'model.1': level 0
                # 'model.1.submodule.1': level 1
                # 'model.1.submodule.1.submodule.1': level 2
                # ...
                level = name.count('submodule')
                self.id_module2level[str(id(module))] = f"{level}"

                def _hook(module, input, output):
                    self.id_module2output[str(id(module))] = output

                module.register_forward_hook(_hook)

                # bottleneck
                if level == len(self.channels) - 2:
                    self.id_module2level[str(id(module.submodule))] = f"{level+1}"
                    module.submodule.register_forward_hook(_hook)

    def get_level2output(self):
        out = {}
        for id_module, output in self.id_module2output.items():
            out[self.id_module2level[id_module]] = output
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model(x)
        level2output = self.get_level2output()
        for k in list(self.id_module2output.keys()):
            del self.id_module2output[k]
        # '{level}' -> [B, C, ...]
        return level2output # type: ignore


if __name__ == "__main__":
    model = MyAttentionUnet(spatial_dims=3, in_channels=1, out_channels=3, channels=(16, 32, 64, 128),
                            strides=(2, 2, 2, 2), norm='instance')
    # input_tensor = torch.rand((2, 1, 32, 32, 32))
    # '{level}' -> [B, C, ...]
    # level2output = model(input_tensor)
    print(model)
