import torch
import torch.nn as nn
from core.networks.blocks.stn import SpatialTransformer


class VectorIntegration(nn.Module):
    def __init__(self, size, int_steps=7):
        super().__init__()
        self.size = size
        self.int_steps = int_steps
        self.transform = SpatialTransformer(self.size)

    def forward(self, flow):
        """

        :param flow: tensor of shape [batch, dimension, *vol_shape]
        :return:
        """
        if self.int_steps:
            vec = torch.div(flow, 2**self.int_steps)
            # assert vec.norm(p=2, dim=1).max().item() < 0.5, "The maximal vector norm must be less than 0.5!"
            for _ in range(self.int_steps):
                vec = vec + self.transform(vec, vec)
            return vec
        else:
            return flow
