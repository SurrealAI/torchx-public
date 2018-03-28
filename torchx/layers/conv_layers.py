import torch.nn as nn
from .core import Layer
import torchx.utils as U


class ConvNd(Layer):
    def __init__(self, dim, out_channels, kernel_size,
                 *, input_shape=None, **kwargs):
        super().__init__(input_shape=input_shape, **kwargs)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert 1 <= dim <= 3
        self.dim = dim
        self.ConvClass = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dim - 1]

    def _build(self, input_shape):
        in_channels = input_shape[1]
        self.conv = self.ConvClass(
            in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            **self.init_kwargs
        )

    def forward(self, x):
        return self.conv(x)

    def get_output_shape(self, input_shape):
        return U.shape_convnd(
            self.dim,
            input_shape,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            has_batch=True,
            **self.init_kwargs
        )


class Conv1D(ConvNd):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)


class Conv2D(ConvNd):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)


class Conv3D(ConvNd):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)


class ConvTransposeNd(Layer):
    def __init__(self, dim, out_channels, kernel_size,
                 *, input_shape=None, **kwargs):
        super().__init__(input_shape=input_shape, **kwargs)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert 1 <= dim <= 3
        self.dim = dim
        self.ConvTransposeClass = \
            [nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d][dim - 1]

    def _build(self, input_shape):
        in_channels = input_shape[1]
        self.conv_t = self.ConvTransposeClass(
            in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            **self.init_kwargs
        )

    def forward(self, x):
        return self.conv_t(x)

    def get_output_shape(self, input_shape):
        return U.shape_transpose_convnd(
            self.dim,
            input_shape,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            has_batch=True,
            **self.init_kwargs
        )


class ConvTranspose1D(ConvTransposeNd):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)


class ConvTranspose2D(ConvTransposeNd):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)


class ConvTranspose3D(ConvTransposeNd):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)


class MaxPoolNd(Layer):
    def __init__(self, dim, kernel_size,
                 *, input_shape=None, **kwargs):
        super().__init__(input_shape=input_shape, **kwargs)
        self.kernel_size = kernel_size
        assert 1 <= dim <= 3
        self.dim = dim
        self.PoolClass = [nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d][dim - 1]

    def _build(self, input_shape):
        self.pool = self.PoolClass(
            self.kernel_size,
            **self.init_kwargs
        )

    def forward(self, x):
        return self.pool(x)

    def get_output_shape(self, input_shape):
        return U.shape_poolnd(
            self.dim,
            input_shape,
            kernel_size=self.kernel_size,
            has_batch=True,
            **self.init_kwargs
        )


class MaxPool1D(MaxPoolNd):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)


class MaxPool2D(MaxPoolNd):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)


class MaxPool3D(MaxPoolNd):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)


class AvgPoolNd(MaxPoolNd):
    def __init__(self, dim, kernel_size,
                 *, input_shape=None, **kwargs):
        super().__init__(dim, kernel_size, input_shape=input_shape, **kwargs)
        self.PoolClass = [nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d][dim - 1]

    def get_output_shape(self, input_shape):
        return U.shape_poolnd(
            self.dim,
            input_shape,
            kernel_size=self.kernel_size,
            dilation=1,  # difference between MaxPool and AvgPool
            has_batch=True,
            **self.init_kwargs
        )


class AvgPool1D(AvgPoolNd):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)


class AvgPool2D(AvgPoolNd):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)


class AvgPool3D(AvgPoolNd):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)
