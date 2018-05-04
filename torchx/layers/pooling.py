import torch.nn as nn
import torchx.utils as U
from .base import Layer


class MaxPoolNd(Layer):
    def __init__(self, dim, kernel_size, **kwargs):
        super().__init__()
        self.kernel_size = kernel_size
        assert 1 <= dim <= 3
        self.dim = dim
        self.pool_kwargs = kwargs
        self._PoolClass = [nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d][dim - 1]
        self._pool = None

    def _build(self, input_shape):
        self._pool = self._PoolClass(
            self.kernel_size,
            **self.pool_kwargs
        )

    def forward(self, x):
        return self._pool(x)

    def get_output_shape(self, input_shape):
        return U.shape_poolnd(
            self.dim,
            input_shape,
            kernel_size=self.kernel_size,
            has_batch=True,
            **self.pool_kwargs
        )

    def get_native(self):
        return self._pool


class MaxPool1d(MaxPoolNd):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)


class MaxPool2d(MaxPoolNd):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)


class MaxPool3d(MaxPoolNd):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)


class AvgPoolNd(MaxPoolNd):
    def __init__(self, dim, kernel_size, **kwargs):
        super().__init__(dim, kernel_size, **kwargs)
        self._PoolClass = [nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d][dim - 1]

    def get_output_shape(self, input_shape):
        return U.shape_poolnd(
            self.dim,
            input_shape,
            kernel_size=self.kernel_size,
            dilation=1,  # difference between MaxPool and AvgPool
            has_batch=True,
            **self.pool_kwargs
        )


class AvgPool1d(AvgPoolNd):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)


class AvgPool2d(AvgPoolNd):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)


class AvgPool3d(AvgPoolNd):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)
