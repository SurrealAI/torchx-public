import pprint
import torchx.utils as U


class Placeholder:
    """
    Placeholder class. Traces the call stack to reconstruct the graph
    Similar to Keras' functional API:
    https://keras.io/getting-started/functional-api-guide/
    https://keras.io/getting-started/functional-api-guide/#the-concept-of-layer-node
    """
    def __init__(self, shape,
                 inbound_layer=None,
                 node_index=0):
        assert U.is_simple_shape(shape)
        self.shape = shape
        self.inbound_layer = inbound_layer
        # https://keras.io/getting-started/functional-api-guide/#the-concept-of-layer-node
        self.node_index = node_index
        self.tensor = None  # actual tensor value

    def __repr__(self):
        return 'P{}'.format(self.shape)

    def __getitem__(self, key):
        from .misc_layers import slice
        return slice(self, key)

    def __add__(self, p2):
        from .merge_layers import add
        return add(self, p2)

    def __sub__(self, p2):
        from .merge_layers import subtract
        return subtract(self, p2)

    def __mul__(self, p2):
        from .merge_layers import multiply
        return multiply(self, p2)

    def __truediv__(self, p2):
        from .merge_layers import divide
        return divide(self, p2)


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
        return U.recursive_flatten(
            self.struct,
            is_base=_is_placeholder
        )

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
