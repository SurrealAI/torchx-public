import torch
import torchx.nn as nnx
import torchx.utils as U
from .base import Layer


class Flatten(Layer):
    def _build(self, input_shape):
        pass

    def forward(self, x):
        return nnx.th_flatten(x)

    def get_output_shape(self, input_shape):
        output_size = 1
        for s in input_shape[1:]:  # exclude batch dim
            output_size *= s
        return (input_shape[0], output_size)


class View(Layer):
    def __init__(self, *args, input_shape=None, **kwargs):
        super().__init__(input_shape=input_shape, **kwargs)
        self.viewargs = args

    def _build(self, input_shape):
        pass

    def forward(self, x):
        return x.view(self.viewargs)

    def get_output_shape(self, input_shape):
        return U.shape_view(input_shape, self.viewargs)


class Slice(Layer):
    def __init__(self, slice, *, input_shape=None, **kwargs):
        super().__init__(input_shape=input_shape, **kwargs)
        self.slice = slice

    def _build(self, input_shape):
        pass

    def forward(self, x):
        return x[self.slice]

    def get_output_shape(self, input_shape):
        return U.shape_slice(input_shape, self.slice)


# ==================== functional forms ====================
def slice(x, slice):
    return Slice(slice)(x)

