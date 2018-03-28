import numpy as np
import torch
from torch.autograd import Variable
import torchx.utils as U


def th_median(t):
    """
    Find median of entire tensor or Variable
    """
    return U.to_float_tensor(t).view(-1).median(dim=0)[0][0]


def th_median_abs(t):
    return th_median(t.abs())


def th_ones_like(tensor):
    s = U.get_shape(tensor)
    assert s is not None
    return torch.ones(s)


def th_zeros_like(tensor):
    s = U.get_shape(tensor)
    assert s is not None
    return torch.zeros(s)


def th_normalize_feature(feats):
    """
    Normalize the whole dataset as one giant feature matrix
    """
    mean = feats.mean(0).expand_as(feats)
    std = feats.std(0).expand_as(feats)
    return (feats - mean) / std


def th_where(cond, x1, x2):
    """
    Similar to np.where and tf.where
    """
    cond = cond.type_as(x1)
    return cond * x1 + (1 - cond) * x2


def th_huber_loss_per_element(x, y=None, delta=1.0):
    """
    Args:
        if y is not None, compute huber_loss(x - y)
    """
    if y is not None:
        x = x - y
    x_abs = x.abs()
    return th_where(x_abs < delta,
                    0.5 * x * x,
                    delta * (x_abs - 0.5 * delta))


def th_norm(tensor, norm_type=2):
    """
    Supports infinity norm
    """
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        return tensor.abs().max()
    else:
        return tensor.norm(norm_type)


def th_clip_norm(tensor, clip, norm_type=2, in_place=False):
    """
    original src:
    http://pytorch.org/docs/0.2.0/_modules/pytorch/nn/utils/clip_grad.html#net_clip_grad_norm
    """
    norm = th_norm(tensor, norm_type)
    clip_coef = clip / (norm + 1e-6)
    if clip_coef < 1:
        if in_place:
            tensor.mul_(clip_coef)
        else:
            tensor = tensor * clip_coef
    return tensor


def th_flatten(x):
    """
    https://discuss.pytorch.org/t/runtimeerror-input-is-not-contiguous/930/4
    `.contiguous()` copies the tensor if the data isn't contiguous
    """
    return x.contiguous().view(x.size(0), -1)


def th_global_avg_pool(x):
    """
    https://arxiv.org/pdf/1312.4400.pdf
    Average each feature map HxW to one number.
    """
    N, C, H, W = x.size()
    return x.view(N, C, H * W).mean(dim=2).squeeze(dim=2)


def th_global_max_pool(x):
    N, C, H, W = x.size()
    # pytorch.max returns a tuple of (max, indices)
    return x.view(N, C, H * W).max(dim=2)[0].squeeze(dim=2)


