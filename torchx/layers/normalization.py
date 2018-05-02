import torch.nn as nn
from .base import Layer


class BatchNormNd(Layer):
    def __init__(self, dim,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 *, input_shape=None):

        """
        http://pytorch.org/docs/stable/nn.html#batchnorm1d

        Args:
            dim: 1d, 2d, or 3d BatchNorm
            input_shape: required by torchx.Layer
            eps: nn.BatchNorm parameter
            momentum: nn.BatchNorm parameter
            affine: nn.BatchNorm parameter
            track_running_stats: nn.BatchNorm parameter
        """
        super().__init__(
            input_shape=input_shape,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats
        )
        self.NormClass = [
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d
        ][dim - 1]

    def _build(self, input_shape):
        """
        C from [N, C, ...] input shape        
        """
        self.batch_norm = self.NormClass(input_shape[1], **self.init_kwargs)

    def forward(self, x):
        return self.batch_norm(x)

    def get_output_shape(self, input_shape):
        return input_shape


class BatchNorm1d(BatchNormNd):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)


class BatchNorm2d(BatchNormNd):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)


class BatchNorm3d(BatchNormNd):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)


class InstanceNormNd(Layer):
    def __init__(self, dim,
                 eps=1e-5,
                 momentum=0.1,
                 affine=False,
                 track_running_stats=False,
                 *, input_shape=None):

        """
        http://pytorch.org/docs/stable/nn.html#instancenorm1d
        
        Args:
            dim: 1d, 2d, or 3d InstanceNorm
            input_shape: required by torchx.Layer
            eps: nn.InstanceNorm parameter
            momentum: nn.InstanceNorm parameter
            affine: nn.InstanceNorm parameter
            track_running_stats: nn.InstanceNorm parameter

        Warnings:
            the default args `affine=False` and `track_running_stats=False`
            are different from BatchNorm
        """
        super().__init__(
            input_shape=input_shape,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats
        )
        self.NormClass = [
            nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d
        ][dim - 1]

    def _build(self, input_shape):
        """
        C from [N, C, ...] input shape
        """
        self.instance_norm = self.NormClass(input_shape[1], **self.init_kwargs)

    def forward(self, x):
        return self.instance_norm(x)

    def get_output_shape(self, input_shape):
        return input_shape


class InstanceNorm1d(InstanceNormNd):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)


class InstanceNorm2d(InstanceNormNd):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)


class InstanceNorm3d(InstanceNormNd):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)

