import numpy as np
import torch
import torch.nn as nn
import torchx.utils as U


def th_median(t):
    """
    Find median of entire tensor or Variable
    """
    return t.view(-1).median(dim=0)[0][0]


def th_median_abs(t):
    return th_median(t.abs())


def th_normalize_feature(feats):
    """
    Normalize the whole dataset as one giant feature matrix
    """
    mean = feats.mean(0).expand_as(feats)
    std = feats.std(0).expand_as(feats)
    return (feats - mean) / std


def th_flatten(x):
    """
    https://discuss.pytorch.org/t/runtimeerror-input-is-not-contiguous/930/4
    `.contiguous()` copies the tensor if the data isn't contiguous
    """
    return x.contiguous().view(x.size(0), -1)


def _get_param_list(obj):
    """
    Args:
        obj: if nn.Module instance, get its parameter list
            otherwise return itself
    """
    if isinstance(obj, nn.Module):
        return obj.parameters()
    else:
        return obj


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


def th_flatten_tensors(tensors_or_module):
    """
    Flatten tensors into a single contiguous 1D buffer
    https://github.com/pytorch/pytorch/blob/master/torch/_utils.py
    """
    tensors = list(_get_param_list(tensors_or_module))
    if len(tensors) == 1:
        return tensors[0].contiguous().view(-1)
    numels = [tensor.numel() for tensor in tensors]
    size = sum(numels)
    offset = 0
    flat = tensors[0].new_empty(size)
    for tensor, numel in zip(tensors, numels):
        flat.narrow(0, offset, numel).copy_(tensor, broadcast=False)
        offset += numel
    return flat


def th_unflatten_tensors(flat, tensors_or_module):
    """View a flat buffer using the sizes of tensors"""
    outputs = []
    offset = 0
    for tensor in _get_param_list(tensors_or_module):
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)


def th_copy_module(module1, module2):
    """
    Assign net1's parameters to net2
    """
    module2.load_state_dict(module1.state_dict())


def th_soft_update(target, source, tau):
    """
    Args:
        target: torch Module or list of tensors
        source: torch Module or list of tensors
    """
    target = _get_param_list(target)
    source = _get_param_list(source)
    with torch.no_grad():
        for target_param, param in zip(target, source):
            target_param.copy_(
                target_param * (1.0 - tau) + param * tau
            )


def th_hard_update(target, source):
    """
    Hard update parameters.

    Args:
        target: torch Module or list of tensors
        source: torch Module or list of tensors
    """
    target = _get_param_list(target)
    source = _get_param_list(source)
    with torch.no_grad():
        for target_param, param in zip(target, source):
            target_param.copy_(param)


def th_to_scalar(x):
    """
    To python native int/float type
    """
    if torch.is_tensor(x):
        assert x.numel() == 1, \
            'tensor must have only 1 element to convert to scalar'
        x = x.view(-1)[0]
        if x.dtype in [torch.float16, torch.float32, torch.float64]:
            return float(x)
        else:
            return int(x)
    elif U.is_np_array(x) or U.is_np_scalar(x):
        return np.asscalar(x)
    elif isinstance(x, (list, tuple)):
        assert len(x) == 1
        return x[0]
    else:
        return x


# ==================== Deprecated in v0.4 ====================
def th_where(cond, x1, x2):
    """
    Similar to np.where and tf.where

    Deprecated: torch v0.4 adds `torch.where()`
    """
    cond = cond.type_as(x1)
    return cond * x1 + (1 - cond) * x2


def th_huber_loss_per_element(x, y=None, delta=1.0):
    """
    Args:
        if y is not None, compute huber_loss(x - y)

    Deprecated: torch v0.4 adds `reduce=True/False` keyword to all loss functions
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
    Deprecated: torch v0.4 adds nn.utils.clip_grad_norm_
    http://pytorch.org/docs/stable/nn.html?highlight=clip#torch.nn.utils.clip_grad_norm_
    """
    norm = th_norm(tensor, norm_type)
    clip_coef = clip / (norm + 1e-6)
    if clip_coef < 1:
        if in_place:
            tensor.mul_(clip_coef)
        else:
            tensor = tensor * clip_coef
    return tensor


