import torch
import torch.nn as nn
import torch.nn.functional as F


class ProductofExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.mode_logits2var = 'softplus'

    def logits2var(self, x):
        if self.mode_logits2var == 'softplus':
            x = F.softplus(x, beta=1)
            x = torch.clamp(x, min=1e-5)
            return x
        elif self.mode_logits2var == 'exp':
            x = torch.clamp(x, max=15)
            x = torch.exp(x)
            x = torch.clamp(x, min=1e-5)
            return x
        else:
            raise NotImplementedError

    def assert_no_nan(self, x):
        try:
            assert torch.isnan(x).sum() == 0, 'nan'
            assert torch.isinf(x).sum() == 0, 'inf'
        except Exception as e:
            raise e

    def get_poe(self, name2param_in, params_returned, ws=None, type_dist='Gaussian'):
        if type_dist == 'Gaussian':
            # name2param_in: name -> [B, M, C, ...]
            for name in name2param_in:
                assert name in ['mean', 'var', 'logits']

            mean = name2param_in['mean'] # [B, M, C, ...]
            if 'var' in name2param_in:
                var = name2param_in['var']
            elif 'logits' in name2param_in:
                logits = name2param_in['logits']
                var = self.logits2var(logits) # [B, M, C/2, ...]
            n_experts = mean.shape[1]
            if ws is not None:
                if ws.ndim == 1:
                    assert len(ws) == n_experts
                    ws = ws[(None, slice(None), *[None] * (mean.dim() - 2))]
                else:
                    assert ws.ndim == mean.ndim

            var_inv = 1 / var # [B, M, C/2, ...]

            name2param_out = {}
            if ws is None:
                var_prod = 1 / torch.sum(var_inv, dim=1) # [B, C/2, ...]
                mean_poe = torch.sum(mean * var_inv, dim=1) * var_prod # [B, C/2, ...]
                if 'mean' in params_returned:
                    name2param_out['mean'] = mean_poe # [B, C/2, ...]
                if 'var' in params_returned or 'std' in params_returned:
                    var_poe = var_prod * n_experts
            else:
                w_var_inv = var_inv * ws # [B, M, C/2, ...]
                var_poe = 1 / torch.sum(w_var_inv, dim=1) # [B, C/2, ...]
                mean_poe = var_poe * torch.sum(mean * w_var_inv, dim=1) # [B, C/2, ...]
                if 'mean' in params_returned:
                    name2param_out['mean'] = mean_poe
            if 'var' in params_returned or 'std' in params_returned or 'log_var' in params_returned:
                var_poe = torch.clamp(var_poe, min=1e-5) # [B, C/2, ...]
            if 'var' in params_returned:
                name2param_out['var'] = var_poe # [B, C/2, ...]
                # self.assert_no_nan(var_poe)
            if 'log_var' in params_returned:
                name2param_out['log_var'] = torch.log(var_poe)
                # self.assert_no_nan(name2param_out['log_var'])
            if 'std' in params_returned:
                name2param_out['std'] = torch.sqrt(var_poe) # [B, C/2, ...]
                # self.assert_no_nan(name2param_out['std'])
        else:
            raise NotImplementedError
        return name2param_out

    def get_sample(self, x):
        mean, std = x
        if self.training:
            sample = torch.randn_like(mean) * std + mean
        else:
            sample = mean
        return sample
