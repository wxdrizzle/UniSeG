import torch
import torch.nn as nn


class GaussianDistance(nn.Module):
    def __init__(self, kern=4):
        super().__init__()
        self.kern = kern
        self.avgpool = nn.AvgPool2d(kernel_size=kern, stride=kern)

    def forward(self, mu_a, logvar_a, mu_b, logvar_b):
        mu_a = self.avgpool(mu_a)
        mu_b = self.avgpool(mu_b)
        var_a = self.avgpool(torch.exp(logvar_a)) / (self.kern * self.kern)
        var_b = self.avgpool(torch.exp(logvar_b)) / (self.kern * self.kern)

        mu_a1 = mu_a.view(mu_a.size(0), 1, -1)
        mu_a2 = mu_a.view(1, mu_a.size(0), -1)
        var_a1 = var_a.view(var_a.size(0), 1, -1)
        var_a2 = var_a.view(1, var_a.size(0), -1)

        mu_b1 = mu_b.view(mu_b.size(0), 1, -1)
        mu_b2 = mu_b.view(1, mu_b.size(0), -1)
        var_b1 = var_b.view(var_b.size(0), 1, -1)
        var_b2 = var_b.view(1, var_b.size(0), -1)

        vaa = torch.sum(
            torch.div(torch.exp(torch.mul(torch.div(torch.pow(mu_a1 - mu_a2, 2), var_a1 + var_a2), -0.5)),
                      torch.sqrt(var_a1 + var_a2)))
        vab = torch.sum(
            torch.div(torch.exp(torch.mul(torch.div(torch.pow(mu_a1 - mu_b2, 2), var_a1 + var_b2), -0.5)),
                      torch.sqrt(var_a1 + var_b2)))
        vbb = torch.sum(
            torch.div(torch.exp(torch.mul(torch.div(torch.pow(mu_b1 - mu_b2, 2), var_b1 + var_b2), -0.5)),
                      torch.sqrt(var_b1 + var_b2)))

        loss = vaa + vbb - torch.mul(vab, 2.0)

        return loss
