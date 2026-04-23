import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


class SpatialTransformer(nn.Module):
    def __init__(self, size, interp_mode='bilinear', padding_mode='zeros'):
        super().__init__()
        self.size = size
        self.dimension = len(self.size)

        vectors = [torch.arange(0, s) for s in self.size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.to(torch.float32)
        self.register_buffer('grid', grid, persistent=False)

        self.interp_mode = interp_mode
        self.padding_mode = padding_mode

    def getOverlapMask(self, flow, **kwargs):
        shape = flow.shape[2:]
        assert len(shape) == self.dimension, "Expected volume dimension %s, got %s!" % (self.dimension, len(shape))

        with torch.no_grad():
            new_locs = self._get_new_locs(flows=flow, **kwargs)
            mask = torch.zeros(flow.shape[0], 1, *self.size, device=flow.device, dtype=torch.uint8)
            for d in range(self.dimension):
                mask += new_locs[:, [d]].gt(shape[d]) + new_locs[:, [d]].le(0)

            mask = mask.eq(0).to(torch.float32)

        return mask

    def forward(self, src, flows=None, thetas=None, interp_mode=None, padding_mode=None, align_corners=True, **kwargs):
        shape = src.shape[2:]
        assert len(shape) == self.dimension, "Expected volume dimension %s, got %s!" % (self.dimension, len(shape))
        assert list(shape) == list(self.size), "The source has a different shape from target! " \
                                      "Target shape %s; Source shape %s." % (list(self.size), list(shape))
        # if not list(shape) == list(self.size):
        #     warnings.warn("The source has a different shape from target! "
        #                   "Target shape %s; Source shape %s." % (list(self.size), list(shape)))
        interp_mode = interp_mode if interp_mode is not None else self.interp_mode
        padding_mode = padding_mode if padding_mode is not None else self.padding_mode

        if flows is None and thetas is None:
            new_locs = kwargs.pop('new_locs', None)
            if new_locs is None:
                return src
        elif thetas is not None:
            raise NotImplementedError
        else:
            new_locs = self._get_new_locs(flows=flows, **kwargs)

        for i in range(self.dimension):
            new_locs[:, i] = 2 * (new_locs[:, i] / (shape[i] - 1) - 0.5)

        if self.dimension == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif self.dimension == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        else:
            raise NotImplementedError

        if padding_mode == 'wrap':
            new_locs = wrap_coordinates(new_locs)
            padding_mode = 'zeros'

        return F.grid_sample(src, new_locs, align_corners=align_corners, mode=interp_mode, padding_mode=padding_mode)

    def _get_new_locs(self, thetas=None, flows=None, **kwargs):
        grid = kwargs.pop('grid', None)
        if grid is None:
            grid = self.grid

        new_locs = grid.clone()

        if thetas is not None:
            raise NotImplementedError

        if flows is not None:
            if isinstance(flows, torch.Tensor):
                new_locs = new_locs + flows
            elif isinstance(flows, (list, tuple)):
                compose_type = kwargs.pop('compose_type', 'compositive')
                new_flow = self.compose_flows(flows, compose_type=compose_type)
                new_locs = new_locs + new_flow
            else:
                raise NotImplementedError

        return new_locs

    def compose_flows(self, flows, **kwargs):
        if isinstance(flows, (list, tuple)):
            new_flow = None
            compose_type = kwargs.pop('compose_type', 'compositive')
            for flow in flows:
                if flow is not None:
                    if new_flow is None:
                        new_flow = flow
                        continue
                    if compose_type == 'compositive':
                        new_flow = new_flow + self.forward(flow, new_flow, interp_mode='bilinear', padding_mode='zeros')
                    elif compose_type == 'additive':
                        new_flow = new_flow + flow
                    else:
                        raise NotImplementedError

        else:
            raise NotImplementedError

        return new_flow


def wrap_coordinates(x):
    y = torch.where(x > 1, x - 2 - 2 * torch.div(x - 1, 2, rounding_mode='trunc').detach(),
                    torch.where(x < -1, x + 2 + 2 * torch.div(-x - 1, 2, rounding_mode='trunc').detach(), x))

    return y
