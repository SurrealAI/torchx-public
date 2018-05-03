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
    def __init__(self, *args):
        super().__init__()
        self._view_args = args

    def _build(self, input_shape):
        pass

    def forward(self, x):
        return x.view(self._view_args)

    def get_output_shape(self, input_shape):
        return U.shape_view(input_shape, self._view_args)


class Slice(Layer):
    def __init__(self, slice):
        super().__init__()
        self._slice = slice

    def _build(self, input_shape):
        pass

    def forward(self, x):
        return x[self._slice]

    def get_output_shape(self, input_shape):
        return U.shape_slice(input_shape, self._slice)


# ==================== functional forms ====================
def flatten(x):
    return Flatten()(x)


def view(x, *args):
    return View(*args)(x)


def slice(x, slice):
    return Slice(slice)(x)
