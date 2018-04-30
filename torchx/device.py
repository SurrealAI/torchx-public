"""
Data type 	            torch.dtype 	                Tensor types
---------               -----------                     ------------
32-bit floating point 	torch.float32 or torch.float 	torch.*.FloatTensor
64-bit floating point 	torch.float64 or torch.double 	torch.*.DoubleTensor
16-bit floating point 	torch.float16 or torch.half 	torch.*.HalfTensor
8-bit integer (unsigned) 	torch.uint8 	torch.*.ByteTensor
8-bit integer (signed) 	torch.int8 	torch.*.CharTensor
16-bit integer (signed) 	torch.int16 or torch.short 	torch.*.ShortTensor
32-bit integer (signed) 	torch.int32 or torch.int 	torch.*.IntTensor
64-bit integer (signed) 	torch.int64 or torch.long 	torch.*.LongTensor
"""
import sys
import contextlib
import torch


# list of (device_list, dtype) tuples.
# device_list is a list of torch.device() objects
# if the device list has more than 1 GPU, it's meant for torch.nn.DataParallel
_PYTORCH_DEVICES_ = []


# convenient constants
CPU_DEVICE = torch.device('cpu')
CUDA_DEVICE = torch.device('cuda')  # typically GPU #0


cuda_count = torch.cuda.device_count

has_cuda = torch.cuda.is_available


def cuda_memory(ids, unit='b', mode='alloc', is_max=False):
    """
    Get cuda memory
    http://pytorch.org/docs/master/cuda.html#torch.cuda.memory_allocated

    Args:
        ids: either a single GPU int index or a list of int indices.
            if "all", return a list of memory on all available GPUs
        unit: b, kb, mb, gb
        mode: alloc (allocated) or cache (cached)
        is_max: return the max GPU memory managed by pytorch
            http://pytorch.org/docs/master/cuda.html#torch.cuda.max_memory_cached

    Returns:
        if ids is a single int, return memory sizes on the device at the index
        if ids is an int list, return a list of memory sizes
        if ids is "all", return a list of memory sizes on all available GPUs
    """
    if isinstance(ids, (tuple, list)):
        return [cuda_memory(i, unit=unit, mode=mode, is_max=is_max)
                for i in ids]
    elif ids == 'all':
        return [cuda_memory(i, unit=unit, mode=mode, is_max=is_max)
                for i in range(cuda_count())]

    assert isinstance(ids, int) and ids >= 0
    unit = unit.lower()
    if unit == 'b':
        divisor = 1
    elif unit == 'kb':
        divisor = 1024
    elif unit == 'mb':
        divisor = 1024 * 1024
    elif unit == 'gb':
        divisor = 1024 * 1024 * 1024
    else:
        raise ValueError('invalid unit: [b, kb, mb, gb]')
    assert mode in ['alloc', 'allocated', 'cache', 'cached']
    is_alloc = mode in ['alloc', 'allocated']
    if is_alloc and is_max:
        query = torch.cuda.max_memory_allocated
    elif is_alloc and not is_max:
        query = torch.cuda.memory_allocated
    elif not is_alloc and is_max:
        query = torch.cuda.max_memory_cached
    else:
        query = torch.cuda.memory_cached
    return query(ids) // divisor


def id_to_device(device):
    """
    -1 -> torch.device('cpu')
    'cuda:1' -> torch.device('cuda:1')
    'gpu:3' -> torch.device('cuda:3')

    Returns:
        torch.device() object
    """
    if isinstance(device, int):
        if device < 0:
            return CPU_DEVICE
        else:
            return torch.device('cuda:{}'.format(device))
    elif not device:
        return CPU_DEVICE
    elif isinstance(device, type(torch.device(0))):
        return device
    elif isinstance(device, str):
        device = device.lower().replace('gpu', 'cuda')
        return torch.device(device)
    else:
        raise ValueError('unsupported: ', device)


def device_to_int(device):
    """
    Returns:
        -1 for CPU
        int index for GPU
    """
    device = id_to_device(device)
    if device == CPU_DEVICE:
        return -1
    else:
        return device.index if device.index else 0


def ids_to_devices(devices):
    """
    [2, 3] -> torch.device('cuda:2'), torch.device('cuda:3')
    'cuda:all' -> [torch.device('cuda:0'), torch.device('cuda:1'), ...]
    """
    if isinstance(devices, (list, tuple)):
        assert 'cuda:all' not in devices, 'cannot have cuda:all in a list of devices'
        return [id_to_device(d) for d in devices]
    elif devices == 'cuda:all':
        count = torch.cuda.device_count()
        if count == 0:
            return [CPU_DEVICE]
        else:
            return [torch.device('cuda:{}'.format(i)) for i in range(count)]
    else:
        return [id_to_device(devices)]


def get_default_device_dtype():
    """
    Returns:
        current (device, dtype) tuple
    """
    return torch.zeros((0,)).device, torch.get_default_dtype()


def set_default_device_dtype(device, dtype):
    """
    Sets device and dtype for all creation ops (e.g. torch.zeros()) afterwards
    """
    device = id_to_device(device)
    if device == CPU_DEVICE:
        tensor_type = torch.FloatTensor
    else:
        tensor_type = torch.cuda.FloatTensor
        torch.cuda.set_device(device.index if device.index else 0)
    # set CPU or GPU for all creation_ops (torch.zeros) afterwards
    torch.set_default_tensor_type(tensor_type)
    torch.set_default_dtype(dtype)


def get_torchx_device_dtype():
    if _PYTORCH_DEVICES_:
        return _PYTORCH_DEVICES_[-1]
    else:
        device, dtype = get_default_device_dtype()
        return [device], dtype


@contextlib.contextmanager
def device_scope(device, dtype=torch.float32, override_parent=True):
    """
    All constructor functions, e.g. `torch.zeros()` inside the scope will
    automatically create tensor with the specified dtype and device.
    This is achieved by setting the default device first by
    torch.set_default_tensor_type, and then set the default dtype by
    torch.set_default_dtype

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

    References:
        - http://pytorch.org/docs/stable/tensor_attributes.html
        - torch.set_default_dtype, torch.set_default_tensor_type
             http://pytorch.org/docs/stable/torch.html#torch.set_default_dtype
    """
    assert dtype in [torch.float32, torch.float64], 'torch v0.4 restrictions'
    global _PYTORCH_DEVICES_
    count = torch.cuda.device_count()
    devices = ids_to_devices(device)
    if len(devices) > 1:
        assert CPU_DEVICE not in devices, 'cannot mix CPU with other devices'
    if count == 0 and devices[0].type != 'cpu':
        print('WARNING: no GPU found, fall back to CPU.', file=sys.stderr)
        devices = [CPU_DEVICE]

    if not override_parent and _PYTORCH_DEVICES_:
        devices, dtype = _PYTORCH_DEVICES_[-1]

    _PYTORCH_DEVICES_.append((devices, dtype))
    old_device, old_dtype = get_default_device_dtype()
    set_default_device_dtype(devices[0], dtype)

    yield devices[0]

    _PYTORCH_DEVICES_.pop()
    set_default_device_dtype(old_device, old_dtype)

