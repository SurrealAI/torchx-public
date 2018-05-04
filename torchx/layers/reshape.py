import torchx.nn as nnx
import torchx.utils as U
from .base import Layer


class Flatten(Layer):
    def __init__(self, start_dim=1):
        """
        Args:
            start_dim: flatten all dimensions starting from `start_dim`
                default=1 means collapsing all but the batch dim
        """
        super().__init__()
        self._start_dim = start_dim

    def _build(self, input_shape):
        assert U.is_simple_shape(input_shape)
        assert self._start_dim < len(input_shape), \
            'start_dim must be smaller than the number of dimensions'

    def forward(self, x):
        return nnx.th_flatten(x, start_dim=self._start_dim)

    def get_output_shape(self, input_shape):
        collapsed = 1
        for s in input_shape[self._start_dim:]:
            collapsed *= s
        return (*input_shape[:self._start_dim], collapsed)


class View(Layer):
    def __init__(self, *args):
        super().__init__()
        self._view_args = args

    def _build(self, input_shape):
        pass

    def forward(self, x):
        return x.view(self._view_args)

    def get_output_shape(self, input_shape):
        return U.shape_view(input_shape, *self._view_args)


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
def flatten(x, start_dim=1):
    return Flatten(start_dim=start_dim)(x)


def view(x, *args):
    return View(*args)(x)


def slice(x, slice):
    return Slice(slice)(x)
