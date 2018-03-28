"""
Shape inference methods
"""
import math
import numpy as np
from functools import partial
from .nested import recursive_map, recursive_combine, recursive_compare
from .numpy_utils import product


def is_simple_shape(shape):
    "Returns: whether a shape is list of ints"
    return (
        isinstance(shape, (list, tuple)) and
        all(map(lambda d : isinstance(d, int) and d > 0, shape))
    )


def is_multi_shape(shape):
    """
    For RNN shape: (output, h, c)
    Max two-level of nesting
    """
    return (
        isinstance(shape, (list, tuple)) and
        all(map(is_simple_shape, shape))
    )


def is_dict_shape(shape):
    """
    One level of dict of shapes
    """
    return (
        isinstance(shape, dict) and
        all(is_multi_shape(s) or is_simple_shape(s) for s in shape.values())
    )


def is_valid_shape(shape, *,
                   allow_simple=True, allow_multi=True, allow_dict=True):
    return (
        (allow_simple and is_simple_shape(shape)) or
        (allow_multi and is_multi_shape(shape)) or
        (allow_dict and is_dict_shape(shape))
    )


def _get_shape(x):
    "single object"
    if isinstance(x, np.ndarray):
        return tuple(x.shape)
    else:
        return tuple(x.size())


def get_shape(struct):
    """
    Recursively walks a data structure (can be tuples, lists, or dict) and
    replaces a tensor/variable/np array with its shape (tuple of int)
    """
    return recursive_map(
        struct,
        func=_get_shape,
        is_base=None,
        leave_none=True
    )


def numel(x):
    """
    Returns:
        number of elements in tensor x. Name "numel" comes from Matlab
    """
    return product(_get_shape(x))


def print_shape(struct, **kwargs):
    """
    For debugging
    """
    print(get_shape(struct), **kwargs)


def get_dim_from_shape(struct):
    """
    Recursively walks a data structure (can be tuples, lists, or dict) and
    replaces a shape (int tuple) with its dim, i.e. len(shape)
    """
    return recursive_map(
        struct,
        func=lambda shape: len(shape),
        is_base=is_simple_shape,
        leave_none=True,
    )


def get_dim(struct):
    """
    Recursively walks a data structure (can be tuples, lists, or dict) and
    replaces a tensor/variable/np array with its dim (int)
    """
    return recursive_map(
        struct,
        func=lambda tensor: len(_get_shape(tensor)),
        is_base=None,
        leave_none=True
    )


def shape_equals(struct1, struct2):
    """
    Recursively compare nested shape, tuple and list are treated as the same type
    """
    return recursive_compare(
        struct1, struct2,
        comparator=lambda x, y: tuple(x) == tuple(y),
        is_base=is_simple_shape
    )


def _expands(dim, *xs):
    "repeat vars like kernel and stride to match dim"
    def _expand(x):
        if isinstance(x, int):
            return (x,) * dim
        else:
            assert len(x) == dim
            return x
    return map(lambda x: _expand(x), xs)


def infer_shape_convnd(dim,
                       input_shape,
                       out_channels,
                       kernel_size,
                       stride=1,
                       padding=0,
                       dilation=1,
                       has_batch=False):
    """
    http://pytorch.org/docs/nn.html#conv1d
    http://pytorch.org/docs/nn.html#conv2d
    http://pytorch.org/docs/nn.html#conv3d
    
    Args:
        dim: supports 1D to 3D
        input_shape: 
        - 1D: [channel, length]
        - 2D: [channel, height, width]
        - 3D: [channel, depth, height, width]
        has_batch: whether the first dim is batch size or not
    """
    assert is_simple_shape(input_shape)
    if has_batch:
        assert len(input_shape) == dim + 2, \
            'input shape with batch should be {}-dimensional'.format(dim+2)
    else:
        assert len(input_shape) == dim + 1, \
            'input shape without batch should be {}-dimensional'.format(dim+1)
    if stride is None:
        # for pooling convention in PyTorch
        stride = kernel_size
    kernel_size, stride, padding, dilation = \
        _expands(dim, kernel_size, stride, padding, dilation)
    if has_batch:
        batch = input_shape[0]
        input_shape = input_shape[1:]
    else:
        batch = None
    _, *img = input_shape
    new_img_shape = [
        math.floor(
            (img[i] + 2 * padding[i] - dilation[i] * (kernel_size[i]- 1) - 1) // stride[i] + 1
        )
        for i in range(dim)
    ]
    assert is_simple_shape(new_img_shape), \
        'shape after convolution is invalid: {}'.format(new_img_shape)
    return ((batch,) if has_batch else ()) + (out_channels, *new_img_shape)


def infer_shape_poolnd(dim,
                       input_shape,
                       kernel_size,
                       stride=None,
                       padding=0,
                       dilation=1,
                       has_batch=False):
    """
    The only difference from infer_shape_convnd is that `stride` default is None
    """
    if has_batch:
        out_channels = input_shape[1]
    else:
        out_channels = input_shape[0]
    return infer_shape_convnd(dim, input_shape, out_channels,
                              kernel_size, stride, padding, dilation, has_batch)


def infer_shape_transpose_convnd(dim,
                                 input_shape,
                                 out_channels,
                                 kernel_size,
                                 stride=1,
                                 padding=0,
                                 output_padding=0,
                                 dilation=1,
                                 has_batch=False):
    """
    http://pytorch.org/docs/nn.html#convtranspose1d
    http://pytorch.org/docs/nn.html#convtranspose2d
    http://pytorch.org/docs/nn.html#convtranspose3d

    Args:
        dim: supports 1D to 3D
        input_shape:
        - 1D: [channel, length]
        - 2D: [channel, height, width]
        - 3D: [channel, depth, height, width]
        has_batch: whether the first dim is batch size or not
    """
    assert is_simple_shape(input_shape)
    if has_batch:
        assert len(input_shape) == dim + 2, \
            'input shape with batch should be {}-dimensional'.format(dim+2)
    else:
        assert len(input_shape) == dim + 1, \
            'input shape without batch should be {}-dimensional'.format(dim+1)
    kernel_size, stride, padding, output_padding, dilation = \
        _expands(dim, kernel_size, stride, padding, output_padding, dilation)
    if has_batch:
        batch = input_shape[0]
        input_shape = input_shape[1:]
    else:
        batch = None
    _, *img = input_shape
    new_img_shape = [
        (img[i] - 1) * stride[i] - 2 * padding[i] + kernel_size[i] + output_padding[i]
        for i in range(dim)
    ]
    assert is_simple_shape(new_img_shape), \
        'shape after transposed convolution is invalid: {}'.format(new_img_shape)
    return ((batch,) if has_batch else ()) + (out_channels, *new_img_shape)


infer_shape_conv1d = partial(infer_shape_convnd, 1)
infer_shape_conv2d = partial(infer_shape_convnd, 2)
infer_shape_conv3d = partial(infer_shape_convnd, 3)


infer_shape_maxpool1d = partial(infer_shape_poolnd, 1)
infer_shape_maxpool2d = partial(infer_shape_poolnd, 2)
infer_shape_maxpool3d = partial(infer_shape_poolnd, 3)


"""
http://pytorch.org/docs/nn.html#avgpool1d
http://pytorch.org/docs/nn.html#avgpool2d
http://pytorch.org/docs/nn.html#avgpool3d
"""
infer_shape_avgpool1d = partial(infer_shape_maxpool1d, dilation=1)
infer_shape_avgpool2d = partial(infer_shape_maxpool2d, dilation=1)
infer_shape_avgpool3d = partial(infer_shape_maxpool3d, dilation=1)

