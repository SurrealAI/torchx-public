import torch

import torchx.utils as U
from .core import Layer


class MergeLayer(Layer):
    def _build(self, input_shape):
        """
        MergeLayer has only stateless operations
        subclass should check for input_shape validity
        """
        self.get_output_shape(input_shape)
        pass

    def _check_shape(self, input_shape):
        assert U.is_multi_shape(input_shape), \
            self.__class__.__name__ + ' input_shape should be a sequence of tuples'

    __call__ = U.method_decorator(U.enable_varargs)(Layer.__call__)


class Concat(MergeLayer):
    def __init__(self, axis=-1, *, input_shape=None, **kwargs):
        """
        Args:
          axis: if 0, concat over batch dim will not affect output shape because
            `input_shape` does not include batch dim
          input_shape: must be a multi shape (i.e. list of simple shapes)
        """
        self.axis = axis
        super().__init__(input_shape=input_shape, **kwargs)

    def forward(self, x_list):
        return torch.cat(x_list, self.axis)

    def get_output_shape(self, input_shape):
        self._check_shape(input_shape)
        out_shape = list(input_shape[0])
        axis = self.axis
        if axis < 0:
            axis += len(out_shape)
        for shape in input_shape[1:]:
            for d in range(len(out_shape)):  # over dimensions
                if d == axis:
                    out_shape[d] += shape[d]
                else:
                    assert out_shape[d] == shape[d], \
                        ('concat dim error', input_shape)
        return tuple(out_shape)

    def __repr__(self):
        return 'Concat({})'.format(self.input_shape)


class ElementwiseMerge(MergeLayer):
    """
    Takes a list of input tensors of the same shape, and do elementwise merge
    """
    def get_output_shape(self, input_shape):
        self._check_shape(input_shape)
        shape_0 = input_shape[0]
        for shape in input_shape:
            assert U.shape_equals(shape, shape_0), \
                (self.__class__.__name__ +
                 ' must have the same shape for all input tensors')
        return shape_0


class Add(ElementwiseMerge):
    def forward(self, x_list):
        return sum(x_list)


class Subtract(ElementwiseMerge):
    def forward(self, x_list):
        assert len(x_list) == 2
        return x_list[0] - x_list[1]


class Multiply(ElementwiseMerge):
    def forward(self, x_list):
        output = x_list[0]
        for x in x_list[1:]:
            output = output * x
        return output


class Maximum(ElementwiseMerge):
    def forward(self, x_list):
        output = x_list[0]
        for x in x_list[1:]:
            output = torch.max(output, x)
        return output


class Average(ElementwiseMerge):
    def forward(self, x_list):
        return sum(x_list) / (1. * len(x_list))


# ==================== functional forms ====================
@U.enable_varargs
def concat(x_list, axis=-1):
    return Concat(axis=axis)(*x_list)


@U.enable_varargs
def add(x_list):
    return Add()(x_list)


def subtract(x1, x2):
    return Subtract()(x1, x2)


@U.enable_varargs
def multiply(x_list):
    return Multiply()(x_list)


@U.enable_varargs
def maximum(x_list):
    return Maximum()(x_list)


@U.enable_varargs
def average(x_list):
    return Average()(x_list)
