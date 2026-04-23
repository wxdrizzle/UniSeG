import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelSoftmaxEstimator(nn.Module):
    """
    Reparameterize categorical variables using the Gumbel-Softmax/Rao estimator.
    """
    def __init__(self, type='categorical', num_classes=2, tau=1, eps=1e-5, **kwargs):
        super(GumbelSoftmaxEstimator, self).__init__()
        assert type in ['categorical', 'bernoulli']
        self.type = type
        self.num_classes = num_classes
        self.tau = tau # non-negative scalar temperature
        self.eps = eps
        self.kwargs = kwargs
        self.straight_through = self.kwargs.pop('straight_through', True)
        self.sample_size = self.kwargs.pop('sample_size', 10) # Monte-Carlo approximation for Gumbel-Rao
        self.threshold = self.kwargs.pop('threshold', 50)

    def forward(self, logits):
        """

        :param logits: tensor of shape [B, *vol_shape, num_classes/None]
        :return: tensor of shape [B, *vol_shape, num_classes/None]
        """
        if self.type == 'categorical':
            assert logits.shape[-1] == self.num_classes

        # sample z from Categorical
        if self.type == 'categorical':
            z_idx = torch.distributions.Categorical(logits=logits).sample()
            z = F.one_hot(z_idx, self.num_classes).float()
        else:
            z = torch.distributions.Bernoulli(logits=logits).sample().float()

        # sample conditional gumbels and preserve gradient of logits
        y_cond = self.conditional_gumbel(logits, z)

        # Monte-Carlo approximation
        if self.type == 'categorical':
            y_soft = torch.softmax(y_cond / self.tau, dim=-1).mean(dim=0)
        else:
            y_soft = torch.sigmoid(y_cond / self.tau).mean(0)

        if self.straight_through:
            return z - y_soft.detach() + y_soft
        else:
            return y_soft

    def conditional_gumbel(self, logits, z):
        """

        :param logits: tensor of shape [B, *vol_shape, num_classes/None]
        :param z: tensor of shape [B, *vol_shape, num_classes/None], sampled from Categorical/Bernoulli
        :return: tensor of shape [sample_size, B, *vol_shape, num_classes/None]
        """

        # i.i.d. exponential, tensor of shape [sample_size, B, *vol_shape, num_classes/None]
        E = torch.distributions.Exponential(rate=torch.ones_like(logits)).sample([self.sample_size])
        if self.type == 'categorical':
            # E of the chosen class
            Ek = torch.sum(z * E, dim=-1, keepdim=True)
            # normalization factor
            Z = logits.exp().sum(dim=-1, keepdim=True) # [B, *vol_shape, 1]
        else:
            Ek = z * E
            Z = logits.exp() + 1

        y_cond = z * (- Ek.clamp_min(self.eps).log() + Z.clamp_min(self.eps).log()) + \
                 (1 - z) * (- torch.clamp_min(E / logits.exp() + Ek / Z, min=self.eps).log())

        return y_cond.detach() + logits - logits.detach()


if __name__ == '__main__':
    gumbel = GumbelSoftmaxEstimator(num_classes=5, sample_size=6)

    logits = torch.randn(4, 5, requires_grad=True)

    sample = gumbel(logits)
    print(sample.shape)
    print(sample.requires_grad)
    print(sample)
