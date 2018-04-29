"""
Numpy utils
"""
import numpy as np
import torch
from torch.autograd import Variable
from .numpy_utils import is_np_array, is_np_scalar, np_cast
from .shape import numel


def get_torch_type(x):
    if isinstance(x, list):
        return 'list'
    elif is_np_array(x) or is_np_scalar(x):
        return 'numpy'
    elif torch.is_tensor(x):
        return 'tensor'
    else:
        return 'scalar'


def to_float_tensor(x, copy=True):
    """
    FloatTensor is the most used pytorch type, so we create a special method for it
    """
    typ = get_torch_type(x)
    if typ == 'tensor':
        assert isinstance(x, torch.FloatTensor)
        return x
    elif typ == 'variable':
        x = x.data
        assert isinstance(x, torch.FloatTensor)
        return x
    elif typ != 'numpy':
        x = np.array(x, copy=False)
    x = np_cast(x, np.float32)
    if copy:
        return torch.FloatTensor(x)
    else:
        return torch.from_numpy(x)


def to_scalar(x):
    typ = get_torch_type(x)
    if typ == 'tensor':
        assert numel(x) == 1, \
            'tensor must have only 1 element to convert to scalar'
        return x.view(-1)[0]
    elif typ == 'numpy':
        return np.asscalar(x)
    elif typ == 'list':
        assert len(x) == 1
        return x[0]
    else:
        return x

