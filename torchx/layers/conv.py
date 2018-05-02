import torch.nn as nn
import torchx.utils as U
from .base import Layer


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


class Conv1d(ConvNd):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)


class Conv2d(ConvNd):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)


class Conv3d(ConvNd):
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


class ConvTranspose1d(ConvTransposeNd):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)


class ConvTranspose2d(ConvTransposeNd):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)


class ConvTranspose3d(ConvTransposeNd):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)

