import pprint

from torchx.utils.shape import is_simple_shape


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
        assert is_simple_shape(shape)
        self.shape = shape
        self.inbound_layer = inbound_layer
        # flag that will build the layer that operates on this placeholders
        self.trigger_build = trigger_build
        self.tensor = None  # actual tensor value

    def __repr__(self):
        return 'P{}'.format(self.shape)


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
        if isinstance(struct, Placeholder):
            return True
        elif isinstance(struct, dict):
            return any(PlaceholderStruct.exists(value) for value in struct.values())
        elif isinstance(struct, (tuple, list)):
            return any(PlaceholderStruct.exists(value) for value in struct)
        else:
            return False

    def get_shape(self):
        return self._get_shape(self.struct)

    def _get_shape(self, struct):
        """
        Recursively get shape, maintain the same data structure
        """
        if isinstance(struct, Placeholder):
            return struct.shape
        elif isinstance(struct, dict):
            return {name: self._get_shape(value)
                    for name, value in struct.items()}
        elif isinstance(struct, (tuple, list)):
            return type(struct)(self._get_shape(value) for value in struct)
        else:
            raise ValueError('invalid placeholder struct type', type(struct))

    def flatten(self):
        placeholder_list = []
        self._flatten(self.struct, placeholder_list)
        return placeholder_list

    def _flatten(self, struct, placeholder_list):
        if isinstance(struct, Placeholder):
            placeholder_list.append(struct)
        elif isinstance(struct, dict):
            for value in struct.values():
                self._flatten(value, placeholder_list)
        elif isinstance(struct, (tuple, list)):
            for value in struct:
                self._flatten(value, placeholder_list)
        else:
            raise ValueError('invalid placeholder struct type', type(struct))

    def setattrs(self, **attrs):
        """
        Recursively set attribute of placeholder objects
        """
        self._setattrs(self.struct, **attrs)

    def _setattrs(self, struct, **attrs):
        if isinstance(struct, Placeholder):
            for key, value in attrs.items():
                setattr(struct, key, value)
        elif isinstance(struct, dict):
            for value in struct.values():
                self._setattrs(value, **attrs)
        elif isinstance(struct, (tuple, list)):
            for value in struct:
                self._setattrs(value, **attrs)
        else:
            raise ValueError('invalid placeholder struct type', type(struct))

    @classmethod
    def from_shape(cls, shape_struct, **kwargs):
        """
        Convert a recursive shape into a PlaceholderStruct

        Args:
          struct: recursive shape struct
          **kwargs: init args to Placeholder constructor
        """
        return cls(PlaceholderStruct._from_shape(shape_struct, **kwargs))

    @staticmethod
    def _from_shape(struct, **kwargs):
        if is_simple_shape(struct):
            return Placeholder(struct, **kwargs)
        elif isinstance(struct, dict):
            return {name: PlaceholderStruct._from_shape(value, **kwargs)
                    for name, value in struct.items()}
        elif isinstance(struct, (tuple, list)):
            return type(struct)(PlaceholderStruct._from_shape(value, **kwargs)
                                for value in struct)
        else:
            raise ValueError('invalid shape struct type', type(struct))

    def bind_tensors(self, tensors):
        """
        Bind tensor values to placeholders for actual layer computation
        """
        self._bind_tensors(self.struct, tensors)

    def _bind_tensors(self, struct, tensors):
        if isinstance(struct, Placeholder):
            struct.tensor = tensors
        elif isinstance(struct, dict):
            assert isinstance(tensors, dict)
            assert len(struct) == len(tensors), \
                'size of placeholder dict must match size of tensor dict'
            for key, value in struct.items():
                self._bind_tensors(value, tensors[key])
        elif isinstance(struct, (tuple, list)):
            assert isinstance(tensors, (tuple, list))
            assert len(struct) == len(tensors), \
                'number of placeholders must match number of tensors'
            for value, t in zip(struct, tensors):
                self._bind_tensors(value, t)
        else:
            raise ValueError('invalid placeholder struct type', type(struct))

    def all_tensor_bound(self):
        return all(p.tensor is not None for p in self.flatten())

    def to_tensors(self):
        """
        Extract the actual tensors (`.tensor` attribute) from placeholders
        keep the recursive structure
        """
        return self._to_tensors(self.struct)

    def _to_tensors(self, struct):
        if isinstance(struct, Placeholder):
            return struct.tensor
        elif isinstance(struct, dict):
            return {name: self._to_tensors(value)
                    for name, value in struct.items()}
        elif isinstance(struct, (tuple, list)):
            return type(struct)(self._to_tensors(value) for value in struct)
        else:
            raise ValueError('invalid placeholder struct type', type(struct))

    def __repr__(self):
        return self._to_string(self.struct)

    def _to_string(self, struct):
        if isinstance(struct, Placeholder):
            return str(struct)
        elif isinstance(struct, dict):
            obj = {name: self._to_string(value)
                   for name, value in struct.items()}
            return pprint.pformat(obj, indent=2)
        elif isinstance(struct, (tuple, list)):
            obj = type(struct)(self._to_string(value) for value in struct)
            return pprint.pformat(obj, indent=2)
        else:
            raise ValueError('invalid placeholder struct type', type(struct))
