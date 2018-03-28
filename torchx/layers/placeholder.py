import pprint
import torchx.utils as U


class Placeholder:
    """
    Placeholder class. Traces the call stack to reconstruct the graph
    Similar to Keras' functional API:
    https://keras.io/getting-started/functional-api-guide/

    Attributes:
      shape

    """
    def __init__(self, shape,
                 inbound_layer=None,
                 trigger_build=False):
        assert U.is_simple_shape(shape)
        self.shape = shape
        self.inbound_layer = inbound_layer
        # flag that will build the layer that operates on this placeholders
        self.trigger_build = trigger_build
        self.tensor = None  # actual tensor value

    def __repr__(self):
        return 'P{}'.format(self.shape)


def _is_placeholder(x):
    return isinstance(x, Placeholder)


class PlaceholderStruct:
    """
    Recursive struct of placeholder.
    Supports arbitrary nesting of list, tuple, and dict
    """
    def __init__(self, struct):
        assert PlaceholderStruct.exists(struct), 'No placeholder found in struct'
        self.struct = struct

    def get(self):
        """
        Returns:
          The underlying plain nested structure of placeholders
        """
        return self.struct

    @staticmethod
    def exists(struct):
        """
        Returns:
          True if any object in the recursive data structure is a Placeholder instance
        """
        return U.recursive_any(
            struct,
            func=lambda x: isinstance(x, Placeholder),
            is_base=None,
        )

    def get_shape(self):
        return U.recursive_map(
            self.struct,
            func=lambda x: x.shape,
            is_base=_is_placeholder,
        )

    def flatten(self):
        return U.recursive_flatten(self.struct)

    def setattrs(self, **attrs):
        """
        Recursively set attribute of placeholder objects
        """
        def _setattrs(x):
            for key, value in attrs.items():
                setattr(x, key, value)
        U.recursive_map(
            self.struct,
            func=_setattrs,
            is_base=_is_placeholder
        )

    @classmethod
    def from_shape(cls, shape_struct, **kwargs):
        """
        Convert a recursive shape into a PlaceholderStruct

        Args:
          struct: recursive shape struct
          **kwargs: init args to Placeholder constructor
        """
        struct = U.recursive_map(
            shape_struct,
            func=lambda shape: Placeholder(shape, **kwargs),
            is_base=U.is_simple_shape,
        )
        return cls(struct)

    def bind_tensors(self, tensors):
        """
        Bind tensor values to placeholders for actual layer computation
        """
        def _binder(placeholder, tensor):
            placeholder.tensor = tensor

        U.recursive_binary_combine(
            self.struct, tensors,
            combinator=_binder,
            is_base=_is_placeholder,
        )

    def all_tensor_bound(self):
        return all(p.tensor is not None for p in self.flatten())

    def to_tensors(self):
        """
        Extract the actual tensors (`.tensor` attribute) from placeholders
        keep the recursive structure
        """
        return U.recursive_map(
            self.struct,
            func=lambda p: p.tensor,
            is_base=_is_placeholder,
        )

    def __repr__(self):
        return self.to_string()

    def to_string(self):
        return self._to_string(self.struct)

    def _to_string(self, struct):
        if isinstance(struct, Placeholder):
            return str(struct)
        elif U.is_mapping(struct):
            obj = {name: self._to_string(value)
                   for name, value in struct.items()}
            return pprint.pformat(obj, indent=2)
        elif U.is_sequence(struct):
            obj = type(struct)(self._to_string(value) for value in struct)
            return pprint.pformat(obj, indent=2)
        else:
            raise ValueError('invalid placeholder struct type', type(struct))
