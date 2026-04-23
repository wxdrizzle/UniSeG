import torch


def logits2seg(logits, dim_class=1):
    # logits: [..., K, ...]
    # return: [..., 1, ...]
    return torch.argmax(logits, dim=dim_class, keepdim=True)


def seg2onehot(seg, n_classes, dim_class=1):
    # seg: [..., 1, ...]
    # return: [..., K, ...]
    shape = list(seg.shape)
    assert shape[dim_class] == 1
    shape[dim_class] = n_classes
    o = torch.zeros(size=shape, device=seg.device)
    onehot = o.scatter_(dim=dim_class, index=seg.long(), value=1)
    return onehot
