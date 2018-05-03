import inspect
import torch.nn as nn
import torchx.utils as U
from torchx.nn import Module
from .placeholder import PlaceholderStruct

_LAYER_REGISTRY = {}


def _register_layer_cls(layer_cls):
    _LAYER_REGISTRY[layer_cls.__name__] = layer_cls


def _get_layer_cls(cls_name):
    # TODO: FIX
    """
    Prioritize user-defined Layer classes (in _LAYER_REGISTRY)
    if not found, get module class from `torch.nn` and apply SameShapeAdapter
    """
    if cls_name in _LAYER_REGISTRY:
        return _LAYER_REGISTRY[cls_name]
    elif hasattr(nn, cls_name):
        cls = getattr(nn, cls_name)
        assert callable(cls), 'torch.nn.{} invalid'.format(cls_name)
        return SameShapeAdapter(cls)
    else:
        raise ValueError('Layer class "{}" not found'.format(cls_name))


class _LayerMeta(U.SaveInitArgsMeta):
    # inherit to avoid metaclass conflict with U.Module
    def __new__(cls, name, bases, class_dict):
        cls = type.__new__(cls, name, bases, class_dict)
        _register_layer_cls(cls)
        return cls


def get_layer_registry():
    return _LAYER_REGISTRY


class Layer(Module, metaclass=_LayerMeta):
    """
    API is very similar to Keras sequential mode, used in `SequentialModel`
    https://keras.io/getting-started/sequential-model-guide/
    Layer is just an annotation. Call build() to instantiate the actual PyTorch module

    Abstract methods:
        _build(input_shape)
        get_output_shape(input_shape)
        forward(*args, **kwargs): the regular PyTorch nn.Module.forward()
    """
    def __init__(self):
        super().__init__()
        self.input_shape = None
        # https://keras.io/getting-started/functional-api-guide/#the-concept-of-layer-node
        self.input_placeholder_nodes = []
        self.output_placeholder_nodes = []

    def _build(self, input_shape):
        """
        Returns:
          instantiated pytorch module (U.Module)
        """
        raise NotImplementedError

    def get_output_shape(self, input_shape):
        raise NotImplementedError

    @property
    def is_built(self):
        return self.input_shape is not None

    def build(self, input_shape):
        """
        This method is not supposed to be overriden. Please override _build() instead.
        Only the first call to build() is effective. Subsequent calls will be no-ops.

        Returns:
            self
        """
        # TODO extend is_valid_shape
        assert U.is_valid_shape(input_shape)
        if not self.is_built:
            self.input_shape = input_shape
            self._build(self.input_shape)
        else:
            if not U.shape_equals(
                input_shape, self.input_shape, ignore_batch_dim=True
            ):
                raise RuntimeError(
                    'build() has already been called, subsequent calls will be no-op. '
                    "However, this call's shape {} does not match input_shape {} "
                    'at the first build() call.'.format(input_shape, self.input_shape)
                )
        return self

    def _pack_call_args(self, args, kwargs):
        """
        Handle variable positional and keyword args.
        Can be overriden to change behavior for __call__ signature

        Returns:
          pack args into a single nested structure of placeholders or tensors,
          depending on the actual call
        """
        assert bool(args) != bool(kwargs), \
            'either positional or keyword, but not both'
        assert U.is_signature_compatible(self.forward, *args, **kwargs), \
            'self.forward() should have the same signature as placeholder call'

        if kwargs:
            return kwargs
        elif len(args) == 1:
            return args[0]  # auto-expand the lone *arg
        else:
            return args

    def _handle_placeholder_call(self, *args, **kwargs):
        """
        Only the following __call__() signatures are allowed:
        1. (*args): for multi-input
        2. (arg_list): for multi-input
        3. (**kwargs): for dict-input
        3. (kwarg_dict): for dict-input

        Signature must be compatible with self.forward

        Returns:
          When args/kwargs has placeholder, return another Placeholder object
          with computed output shape
        """
        placeholders = self._pack_call_args(args, kwargs)
        pstruct = PlaceholderStruct(placeholders)
        self.input_placeholder_nodes.append(pstruct)
        self.build(pstruct.get_shape())  # will only build once
        output_shape = self.get_output_shape(self.input_shape)
        output = PlaceholderStruct.from_shape(
            output_shape,
            inbound_layer=self,
            node_index=len(self.input_placeholder_nodes) - 1
        )
        assert isinstance(output, PlaceholderStruct)
        self.output_placeholder_nodes.append(output)
        return output.get()  # plain nested structure of placeholders

    def __call__(self, *args, **kwargs):
        """
        If input_shape specified in __init__, automatically call build()
        otherwise self.build(input_shape) must be called explicitly before forward prop

        Returns:
          - computed tensor
          - or Placeholder object if args/kwargs has placeholder
        """
        if PlaceholderStruct.exists(args) or PlaceholderStruct.exists(kwargs):
            return self._handle_placeholder_call(*args, **kwargs)

        tensors = self._pack_call_args(args, kwargs)  # nested tensors
        tensors_shape = U.get_shape(tensors)
        self.build(tensors_shape)  # will only build once
        return super().__call__(*args, **kwargs)


def get_torch_builtin_modules(pkg_name):
    """
    Args:
        pkg_name: pkgs in torch.nn.modules

    Returns:
        dict {module_name: module_class}
    """
    assert hasattr(nn.modules, pkg_name), \
        'pkg must be one of '+str([m for m in dir(nn.modules) if not m.startswith('_')])
    exclude = ['Parameter', 'F', 'Module']
    builtins = {}
    pkg = getattr(nn.modules, pkg_name)
    for cls_name in dir(pkg):
        if cls_name[0].isupper() and cls_name not in exclude:
            builtins[cls_name] = getattr(pkg, cls_name)
    return builtins
