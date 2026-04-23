import torch
import numpy as np


def jensen_shannon_divergence(p, q, normalize=False):
    # p, q: Probability vectors of shape (batch_size, K)
    # bounded between 0 and ln(2)
    m = 0.5 * (p+q)
    m = torch.clamp(m, min=1e-8)
    kl_pm = torch.sum(p * torch.log(p/m + 1e-8), dim=1)
    kl_qm = torch.sum(q * torch.log(q/m + 1e-8), dim=1)
    jsd = 0.5 * (kl_pm+kl_qm)
    if normalize:
        jsd = jsd / np.log(2)
    return jsd


def hellinger_distance(p, q):
    # p, q: Probability vectors of shape (batch_size, K)
    # bounded between 0 and 1

    h = torch.sum((torch.sqrt(p + 1e-6) - torch.sqrt(q + 1e-6))**2, dim=1)
    h = torch.sqrt(h + 1e-6) / np.sqrt(2)

    return h


def fisher_rao_distance(x, y):
    # x: Probability vectors of shape (B, N, K)
    # y: Probability vectors of shape (B, M, K)
    # bounded between 0 and pi
    assert len(x.shape) == len(y.shape) == 3
    assert x.shape[0] == y.shape[0]
    assert x.shape[2] == y.shape[2]

    x = x[:, :, None] # (B, N, 1, K)
    y = y[:, None] # (B, 1, M, K)
    sqrt_xy = torch.sqrt(x*y + 1e-6) # (B, N, M, K)
    inner_prod = torch.sum(sqrt_xy, dim=-1).clamp(max=1. - 1e-6) # (B, N, M)
    dist = 2. * torch.acos(inner_prod) # (B, N, M)
    return dist
