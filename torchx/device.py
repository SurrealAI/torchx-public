import sys
import contextlib
import torch
from collections import abc


# list of torch.device() objects
# if the list has more than 1 GPU, it's meant for torch.nn.DataParallel
_PYTORCH_DEVICES_ = []


# convenient constants
CPU_DEVICE = torch.device('cpu')
CUDA_DEVICE = torch.device('cuda')  # typically GPU #0


cuda_count = torch.cuda.device_count

has_cuda = torch.cuda.is_available


def get_devices_in_scope():
    return _PYTORCH_DEVICES_[-1] if _PYTORCH_DEVICES_ else [CPU_DEVICE]


def ids_to_devices(device):
    """
    -1 -> torch.device('cpu')
    [2, 3] -> torch.device('cuda:2'), torch.device('cuda:3')
    'cuda:1' -> torch.device('cuda:1')
    'cuda:all' -> [torch.device('cuda:0'), torch.device('cuda:1'), ...]

    Returns:
        list of torch.device()
    """
    if isinstance(device, int):
        if device < 0:
            return [CPU_DEVICE]
        else:
            return [torch.device('cuda:{}'.format(device))]
    elif not device:
        return [CPU_DEVICE]
    elif isinstance(device, type(torch.device(0))):
        return [device]
    elif isinstance(device, str):
        device = device.lower()
        if device == 'cuda:all':
            count = torch.cuda.device_count()
            if count == 0:
                return [CPU_DEVICE]
            else:
                return [torch.device('cuda:{}'.format(i)) for i in range(count)]
        else:
            return [torch.device(device)]
    elif isinstance(device, (list, tuple)):
        assert 'cuda:all' not in device, 'cannot have cuda:all in a list of devices'
        return [ids_to_devices(d)[0] for d in device]
    else:
        raise ValueError('unsupported: ', device)


@contextlib.contextmanager
def device_scope(device, override_parent=True, fallback=True):
    """
    All constructor functions, e.g. `torchx.zeros()` inside the scope will
    automatically send the value to the specified device. If you use the builtin
    constructors (e.g. torch.zeros()), the device transfer will NOT happen
    automatically. You have to specify `device=` explicitly.

    If you create torchx.nn.Module inside this scope, it will automatically
    invoke module.to(device) when you call the module

    If you have torchx.DataParallel inside a module definition, the device_ids
    will be inferred from the device scope. Do NOT use the builtin
    `torch.nn.DataParallel` if you want the automatic transfer.

    For DataParallel, the module itself must be transferred to
    the `devices[0]` in the device_ids list. It will be done automatically.

    Args:
        device: one of the following format
            - CPU: -1
            - int index of the GPU device.
            - torch device strings: "cpu", "cuda:0", etc.
            - torch.devce() object
            - "cuda:all" to use all available GPUs for torchx.DataParallel
            - list of any of the above for torchx.DataParallel
        override_parent: True to override the device in parent scope.
        fallback: fallback to CPU when GPU is unavailable
    """
    global _PYTORCH_DEVICES_
    count = torch.cuda.device_count()
    devices = ids_to_devices(device)
    if len(devices) > 1:
        assert CPU_DEVICE not in devices, 'cannot mix CPU with other devices'
        # must be either CPU or the first GPU
        # https://github.com/pytorch/pytorch/issues/1280
        if devices[0].index not in [0, None]:
            print('For multiGPU training, the first GPU index should be 0. ' 
                  'You can set CUDA_VISIBLE_DEVICES env variable to work around.',
                  file=sys.stderr)
    if fallback and count == 0 and devices[0].type != 'cpu':
        print('WARNING: no GPU found, fall back to CPU.', file=sys.stderr)
        devices = [CPU_DEVICE]

    if not override_parent and _PYTORCH_DEVICES_:
        devices = _PYTORCH_DEVICES_[-1]

    _PYTORCH_DEVICES_.append(devices)
    yield
    _PYTORCH_DEVICES_.pop()

