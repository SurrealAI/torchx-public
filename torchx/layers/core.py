import inspect
import torch.nn as nn
import torchx.utils as U
from torchx.nn import Module
from .placeholder import PlaceholderStruct

_LAYER_REGISTRY = {}


def _register_layer_cls(layer_cls):
    _LAYER_REGISTRY[layer_cls.__name__] = layer_cls


def _get_layer_cls(cls_name):
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
    def __init__(self, *, input_shape=None, **kwargs):
        """
        Args:
            input_shape: None if this is not the first layer. See
        """
        super().__init__()
        self.input_shape = input_shape
        self.init_kwargs = kwargs
        self.is_built = False
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

    def build(self, input_shape=None):
        """
        Only the first call has effect. Subsequent calls are no-ops
        Please override _build() abstract method
        """
        # print('DEBUG BUILD', self.__class__.__name__, input_shape, '->', self.out_features if hasattr(self, 'out_features') else '')
        if self.is_built:
            assert self.input_shape is not None, \
                'internal error, self.input_shape should have been set after build'
            return
        if self.input_shape is not None:
            # TODO extend is_valid_shape
            assert U.is_valid_shape(self.input_shape)
            if input_shape is not None:
                assert U.shape_equals(input_shape, self.input_shape,
                                      ignore_batch_dim=True), \
                    'self.input_shape has already been set, ' \
                    'must be the same as `input_shape` arg of build(input_shape)'
        else:
            assert U.is_valid_shape(input_shape)
            self.input_shape = input_shape
        self._build(self.input_shape)
        self.is_built = True
        assert self.input_shape is not None, 'internal error'

    def _handle_placeholder(self, input_pstruct, output_shape, node_index):
        """
        Can be overriden by subclasses

        Args:
          input_pstruct: PlaceholderStruct
          output_shape: computed as self.get_output_shape(self.input_shape)
          node_index: https://keras.io/getting-started/functional-api-guide/#the-concept-of-layer-node

        Returns:
          PlaceholderStruct
        """
        # print('DEBUG INSIDE HANDLE', self.__class__.__name__, placeholders, input_shape)
        return PlaceholderStruct.from_shape(
            output_shape,
            inbound_layer=self,
            node_index=node_index
        )

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
        if self.input_shape is None:
            self.input_shape = pstruct.get_shape()  # for build()
        else:
            assert U.shape_equals(self.input_shape, pstruct.get_shape()), \
                'multiple placeholder calls should have the same input_shape'
        output_shape = self.get_output_shape(self.input_shape)
        output = self._handle_placeholder(
            pstruct,
            output_shape=output_shape,
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
        if self.input_shape is not None:
            self.build()
        else:
            # TODO enable auto-shape inference and auto build
            raise RuntimeError(self.__class__.__name__ +
               ' input_shape not specified and Layer.build() has not been called yet.')
        tensors = self._pack_call_args(args, kwargs)  # nested tensors
        tensors_shape = U.get_shape(tensors)
        assert U.shape_equals(
            tensors_shape, self.input_shape, ignore_batch_dim=True
        ), ('actual tensor shape does not match input_shape at build time',
            tensors_shape, self.input_shape)
        return super().__call__(*args, **kwargs)


def SameShapeAdapter(layer):
    if inspect.isclass(layer):
        if issubclass(layer, Layer):
            return layer

        class _SameShapeAdapter(Layer):
            """
            Adapts an existing torch.nn.<layer>, assume the same input and output shape
            """
            def __init__(self, *args, **kwargs):
                # input_shape does not matter, just set a non-None value
                super().__init__(input_shape='_placeholder_', **kwargs)
                self.init_args = args

            def _build(self, input_shape):
                return layer(*self.init_args, **self.init_kwargs)

            def get_output_shape(self, input_shape):
                assert U.is_simple_shape(input_shape)
                return input_shape

        return _SameShapeAdapter

    else:  # layer is an instantiated object
        if isinstance(layer, Layer):
            return layer
        # duck typing, fake the abstract class methods
        layer.input_shape = None
        layer.build = lambda input_shape=None: None
        layer.get_output_shape = lambda input_shape: input_shape
        return layer


