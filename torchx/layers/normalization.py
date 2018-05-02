import torch.nn as nn
from .base import Layer
from .container import Lambda


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
            eps: nn.BatchNorm parameter
            momentum: nn.BatchNorm parameter
            affine: nn.BatchNorm parameter
            track_running_stats: nn.BatchNorm parameter
            input_shape: see torchx.Layer
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
        channel = input_shape[1]
        self.batch_norm = self.NormClass(channel, **self.init_kwargs)

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
            eps: nn.InstanceNorm parameter
            momentum: nn.InstanceNorm parameter
            affine: nn.InstanceNorm parameter
            track_running_stats: nn.InstanceNorm parameter
            input_shape: see torchx.Layer

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
        channel = input_shape[1]
        self.instance_norm = self.NormClass(channel, **self.init_kwargs)

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


class LayerNorm(Layer):
    def __init__(self, num_normalize_dim,
                 eps=1e-5,
                 elementwise_affine=True,
                 *, input_shape=None):
        """
        http://pytorch.org/docs/stable/nn.html#layernorm

        Args:
          num_normalize_dim: the number of dimensions to be normalized over,
            starting from the last dim in the input tensor.
            e.g. input tensor size [batch=16, C=10, H=20, W=30]
            - num_dim=1: normalize over [W=30], affine parameters gamma and beta
                will have shape [30]
            - num_dim=2: normalize over [H, W], gamma and beta shape [20, 30]
            - num_dim=3: normalize over [C, H, W], gamma and beta shape [10, 20, 30]
          eps: nn.LayerNorm parameter
          elementwise_affine: nn.LayerNorm parameter
          input_shape: see torchx.Layer
        """
        super().__init__(
            input_shape=input_shape,
            eps=eps,
            elementwise_affine=elementwise_affine
        )
        self.num_normalize_dim = num_normalize_dim

    def _build(self, input_shape):
        """
        e.g. input tensor size [batch=16, C=10, H=20, W=30]
        - num_dim=1: normalize over [W=30], affine parameters gamma and beta
            will have shape [30]
        - num_dim=2: normalize over [H, W], gamma and beta shape [20, 30]
        - num_dim=3: normalize over [C, H, W], gamma and beta shape [10, 20, 30]
        """
        assert self.num_normalize_dim <= len(input_shape) - 1, \
            'num_normalize_dim must be at most input dim - 1'
        normalized_shape = input_shape[-self.num_normalize_dim:]
        self.layer_norm = nn.LayerNorm(normalized_shape, **self.init_kwargs)

    def forward(self, x):
        return self.layer_norm(x)

    def get_output_shape(self, input_shape):
        return input_shape


class GroupNorm(Layer):
    def __init__(self, num_groups,
                 eps=1e-5,
                 affine=True,
                 *, input_shape=None):
        """
        Args:
          num_groups: equivalent to LayerNorm if num_groups == 1
            equivalent to InstanceNorm if num_groups == num_channels
          eps: nn.GroupNorm parameter
          elementwise_affine: nn.GroupNorm parameter
          input_shape: see torchx.Layer
        """
        super().__init__(
            input_shape=input_shape,
            eps=eps,
            affine=affine
        )
        self.num_groups = num_groups

    def _build(self, input_shape):
        """
        self.num_groups must divide input channel
        """
        channel = input_shape[1]
        assert channel % self.num_groups == 0, \
            'number of groups {} must divide number of channels {}'.format(
                self.num_groups, channel)
        self.group_norm = nn.GroupNorm(
            self.num_groups, channel, **self.init_kwargs)

    def forward(self, x):
        return self.group_norm(x)

    def get_output_shape(self, input_shape):
        return input_shape


# ========= Same shape layers with no learnable parameters ===============
_wrap = Lambda.wrap_same_shape_class

LocalResponseNorm = _wrap(nn.LocalResponseNorm)
CrossMapLRN2d = _wrap(nn.CrossMapLRN2d)
