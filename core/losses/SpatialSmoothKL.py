# -*- coding: utf-8 -*-
"""
Deformation regularization module for image registration.

__author__ = Xinzhe Luo
__version__ = 0.1

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpatialSmoothKL(nn.Module):
    """
    Compute spatial smoothness KL-divergence on probabilistic flows.
    Adapted from https://github.com/voxelmorph/voxelmorph/blob/master/voxelmorph/tf/losses.py

    """
    def __init__(self, dimension, prior_lambda=10, **kwargs):
        super(SpatialSmoothKL, self).__init__()
        self.dimension = dimension
        self.prior_lambda = prior_lambda
        self.kwargs = kwargs
        self.disp_levels = self.kwargs.pop("disp_levels", (0, ))
        self.D = dict([(i, None) for i in self.disp_levels])

    def _adjacency_weight(self):
        """
        Compute the adjacency weight

        """
        weight_inner = np.zeros([3] * self.dimension)
        for j in range(self.dimension):
            o = [[1]] * self.dimension
            o[j] = [0, 2]
            weight_inner[np.ix_(*o)] = 1

        weight = np.zeros([self.dimension] + [1, *[3] * self.dimension])
        for i in range(self.dimension):
            weight[i, 0] = weight_inner

        return weight

    def _degree_matrix(self, vol_shape, device):
        z = torch.ones(1, self.dimension, *vol_shape, device=device)
        adj_weight = torch.as_tensor(self._adjacency_weight(), dtype=torch.float32, device=device)

        if self.dimension == 2:
            conv = F.conv2d
        elif self.dimension == 3:
            conv = F.conv3d
        else:
            raise NotImplementedError

        return conv(z, weight=adj_weight, padding=1, groups=self.dimension)

    def precision_loss(self, y_pred):
        """

        :params y_pred: tensor of shape [batch, dimension, *vol_shape]
        """
        sm = 0
        for i in range(self.dimension):
            y = y_pred.permute(i + 2, *range(0, i + 2), *range(i + 3, self.dimension + 2))
            df = y[1:, ] - y[:-1, ]
            sm += torch.mean(df**2)

        return 0.5 * sm / self.dimension

    def forward(self, y_pred, level=None, prior_lambda=None):
        """

        :params y_pred: tensor of shape [batch, dimension * 2, *vol_shape]
        """
        if level is None:
            assert self.disp_levels == (0, ), "Must have only one level when level is None!"
            level = 0

        if prior_lambda is None:
            prior_lambda = self.prior_lambda

        mu = y_pred[:, :self.dimension]
        logvar = y_pred[:, self.dimension:]

        if self.D[level] is None:
            self.D[level] = self._degree_matrix(y_pred.shape[2:], device=y_pred.device)

        var_term = prior_lambda * self.D[level] * torch.exp(logvar) - logvar
        var_term = torch.mean(var_term)

        prec_term = prior_lambda * self.precision_loss(mu)

        return 0.5 * self.dimension * (var_term + prec_term)
