"""
https://pytorch.org/docs/stable/distributed.html
https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
https://pytorch.org/docs/stable/distributed.html#launch-utility

Complete working example:
https://github.com/pytorch/examples/blob/master/imagenet/main.py

Wrapper around nn.DataParallel and nn.parallel.DistributedDataParallel
"""
import torch
import torch.nn as nn
from .device import device_to_int, get_torchx_device_dtype


def DataParallel(module, output_device=None, dim=0):
    """
    Reads the device scope from torchx.
    If the device in scope is CPU, this wrapper will be no op.
    """
    devices, dtype = get_torchx_device_dtype()
    if devices[0] == torch.device('cpu'):
        return module
    else:
        device_ids = [device_to_int(dev) for dev in devices]
        return nn.DataParallel(
            module,
            device_ids=device_ids,
            output_device=output_device,
            dim=dim
        )



