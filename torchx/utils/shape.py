"""
Shape inference methods
"""
import math
import numpy as np
import torch
from functools import partial
from .nested import recursive_map, recursive_all, recursive_compare
from .numpy_utils import product


def is_simple_shape(shape):
    "Returns: whether a shape is list of ints"
    return (
        isinstance(shape, (list, tuple)) and
        all(map(lambda d : (isinstance(d, int) and d > 0) or d is None, shape))
    )


def is_sequence_shape(shape):
    "shapes like [shape1, shape2, shape3]"
    return (
        isinstance(shape, (list, tuple))
        and all(is_valid_shape(s) for s in shape)
    )


def is_valid_shape(shape):
    return recursive_all(
        shape,
        func=lambda _: True,
        is_base=is_simple_shape,
        unknown_type_handler=lambda _: False
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


def shape_equals(struct1, struct2, ignore_batch_dim=False):
    """
    Recursively compare nested shape, tuple and list are treated as the same type.

    Args:
        ignore_batch_dim: default False
          if ignored, only compare shapes starting from the second dim
    """
    if ignore_batch_dim:
        comparator = lambda shape1, shape2: tuple(shape1)[1:] == tuple(shape2)[1:]
    else:
        comparator = lambda shape1, shape2: tuple(shape1) == tuple(shape2)
    return recursive_compare(
        struct1, struct2,
        comparator=comparator,
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


_HELPER_TENSOR = torch.zeros((1,))


def shape_slice(input_shape, slice):
    """
    Credit to Adam Paszke for the trick. Shape inference without instantiating
    an actual tensor.
    The key is that `.expand()` does not actually allocate memory
    Still needs to allocate a one-element HELPER_TENSOR.
    """
    return tuple(_HELPER_TENSOR.expand(*input_shape)[slice].size())


class ShapeSlice:
    """
    shape_slice inference with easy []-operator
    """
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def __getitem__(self, slice):
        return shape_slice(self.input_shape, slice)


def shape_view(input_shape, *view_args):
    """
    Can only have at most one "-1" dimension to be inferred.
    The expand() trick doesn't work because the input must be contiguous.
    """
    # TODO handle input shape with None
    view_args = list(view_args)
    assert view_args.count(-1) <= 1, 'can have at most one -1 for inferring shape'
    old_elems = product(input_shape)
    new_elems = product(view_args)
    if new_elems < 0:  # -1 exists
        new_elems *= -1
        assert old_elems % new_elems == 0, \
            'new shape must be compatible with old shape'
        view_args[view_args.index(-1)] = old_elems // new_elems  # infer -1
    else:
        assert old_elems == new_elems, \
            'new shape must have the same total number of elements as old shape'
    return tuple(view_args)


def shape_convnd(dim,
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


def shape_poolnd(dim,
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
    return shape_convnd(dim, input_shape, out_channels,
                        kernel_size, stride, padding, dilation, has_batch)


def shape_transpose_convnd(dim,
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


shape_conv1d = partial(shape_convnd, 1)
shape_conv2d = partial(shape_convnd, 2)
shape_conv3d = partial(shape_convnd, 3)


shape_maxpool1d = partial(shape_poolnd, 1)
shape_maxpool2d = partial(shape_poolnd, 2)
shape_maxpool3d = partial(shape_poolnd, 3)


"""
http://pytorch.org/docs/nn.html#avgpool1d
http://pytorch.org/docs/nn.html#avgpool2d
http://pytorch.org/docs/nn.html#avgpool3d
"""
shape_avgpool1d = partial(shape_maxpool1d, dilation=1)
shape_avgpool2d = partial(shape_maxpool2d, dilation=1)
shape_avgpool3d = partial(shape_maxpool3d, dilation=1)

