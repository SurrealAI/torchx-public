"""
Utils to handle nested data structures
"""
import collections
from functools import partial


def is_sequence(obj):
    """
    Returns:
      True if the sequence is a collections.Sequence and not a string.
    """
    return (isinstance(obj, collections.Sequence)
            and not isinstance(obj, str))


def is_mapping(obj, only_mutable=False):
    """
    Args:
      obj:
      only_mutable: only allows mutable mapping, otherwise allow anything that
        extends collections.Mapping
    """
    if only_mutable:
        return isinstance(obj, collections.MutableMapping)
    else:
        return isinstance(obj, collections.Mapping)


def _is_sequence(struct, allow_any_seq_type):
    return (allow_any_seq_type and is_sequence(struct)
            or not allow_any_seq_type and isinstance(struct, (list, tuple)))


def _is_mapping(struct, allow_any_dict_type):
    return (allow_any_dict_type and is_mapping(struct, only_mutable=True)
            or not allow_any_dict_type and isinstance(struct, dict))


def recursive_map(struct,
                  func,
                  is_base=None,
                  allow_any_seq_type=True,
                  allow_any_dict_type=True,
                  leave_none=False):
    """
    Recursively walks a data structure (can be Sequence or dict) and
    replaces base objects with func(obj) results.
    The result retains the original nested structure

    Args:
      struct: recursive data structure
      func: maps a base object to a return value
      is_base: function that takes an object and returns True if it's a base object
        if is_base=None, all non-Sequence and non-dict objects will be treated
        as base object
      allow_any_seq_type: True (default) to allow all data structure that
        implements the collections.Sequence interface (except for str).
        False to allow only list and tuple
      allow_any_dict_type: True (default) to allow all data structure that
        implements the collections.MutableMapping interface.
        False to allow only dict
      leave_none: if True, returns None if an object is None.
        False (default) to handle None's yourself.
    """
    if is_base and is_base(struct):
        return func(struct)
    elif _is_sequence(struct, allow_any_seq_type):
        return_seq = [
            recursive_map(
                struct=value,
                func=func,
                is_base=is_base,
                allow_any_seq_type=allow_any_seq_type,
                allow_any_dict_type=allow_any_dict_type,
                leave_none=leave_none
            )
            for value in struct
        ]
        return type(struct)(return_seq)
    elif _is_mapping(struct, allow_any_dict_type):
        # not using dict comprehension because if the struct is OrderedDict,
        # the return value should also retain order
        return_dict = type(struct)()
        for key, value in struct.items():
            return_dict[key] = recursive_map(
                struct=value,
                func=func,
                is_base=is_base,
                allow_any_seq_type=allow_any_seq_type,
                allow_any_dict_type=allow_any_dict_type,
                leave_none=leave_none
            )
        return return_dict
    elif leave_none and struct is None:
        return None
    elif is_base is None:  # pass all non-Sequence and non-dict objects
        return func(struct)
    else:  # if is_base is not None and struct is not Sequence or dict or base object
        raise ValueError('Unknown data structure type: {}'.format(type(struct)))


def recursive_reduce(struct,
                     reduce_op,
                     func=None,
                     is_base=None,
                     allow_any_seq_type=True,
                     allow_any_dict_type=True):
    """
    Recursively walks a data structure (can be Sequence or dict), maps each
    base object to func(obj), and reduce the results over sequence or dict.values()

    Args:
      struct: recursive data structure
      reduce_op: must be an associative operator that takes a list of values
        and returns a single value. Should be able to handle the case where the
        list only has one value.
      func: maps a base object to value that will be reduced.
        if None, default to identity function `lambda x: x`
      is_base: function that takes an object and returns True if it's a base object
        if is_base=None, all non-Sequence and non-dict objects will be treated
        as base object
      allow_any_seq_type: True (default) to allow all data structure that
        implements the collections.Sequence interface (except for str).
        False to allow only list and tuple
      allow_any_dict_type: True (default) to allow all data structure that
        implements the collections.MutableMapping interface.
        False to allow only dict
    """
    if func is None:
        func = lambda x: x
    if is_base and is_base(struct):
        return func(struct)
    elif (_is_sequence(struct, allow_any_seq_type)
          or _is_mapping(struct, allow_any_dict_type)):
        if _is_mapping(struct, allow_any_dict_type):
            values = struct.values()
        else:
            values = struct
        return reduce_op([
            recursive_reduce(
                struct=value,
                reduce_op=reduce_op,
                func=func,
                is_base=is_base,
                allow_any_seq_type=allow_any_seq_type,
                allow_any_dict_type=allow_any_dict_type,
            )
            for value in values
        ])
    elif is_base is None:  # pass all non-Sequence and non-dict objects
        return func(struct)
    else:  # if is_base is not None and struct is not Sequence or dict or base object
        raise ValueError('Unknown data structure type: {}'.format(type(struct)))


def flatten(struct, **kwargs):
    """
    Flattens base objects into a list
    """
    flattened_list = []

    def map_func(obj):
        flattened_list.append(obj)

    recursive_map(
        struct=struct,
        func=map_func,
        **kwargs
    )
    return flattened_list


def recursive_any(struct,
                  func=None,
                  **kwargs):
    """
    Recursively walks a data structure (can be Sequence or dict)
    return True if any func(obj) evaluates to True
    Nested version of the Python builtin any()

    Args:
      struct: recursive data structure
      func: maps a base object to True/False
      **kwargs: see recursive_reduce
    """
    return recursive_reduce(
        struct=struct,
        reduce_op=any,
        func=func,
        **kwargs
    )


def recursive_all(struct,
                  func=None,
                  **kwargs):
    """
    Recursively walks a data structure (can be Sequence or dict)
    return True if and only if all func(obj) evaluates to True
    Nested version of the Python builtin all()

    Args:
      struct: recursive data structure
      func: maps a base object to True/False
      **kwargs: see recursive_reduce
    """
    return recursive_reduce(
        struct=struct,
        func=func,
        reduce_op=all,
        **kwargs
    )


def recursive_sum(struct,
                  func=None,
                  **kwargs):
    """
    Recursively walks a data structure (can be Sequence or dict)
    Sums all func(obj)
    Nested version of the Python builtin sum()

    Args:
      struct: recursive data structure
      func: maps a base object to value
      **kwargs: see recursive_reduce
    """
    return recursive_reduce(
        struct=struct,
        func=func,
        reduce_op=sum,
        **kwargs
    )


def recursive_max(struct,
                  func=None,
                  **kwargs):
    """
    Recursively walks a data structure (can be Sequence or dict)
    Max of all func(obj)
    Nested version of the Python builtin max()

    Args:
      struct: recursive data structure
      func: maps a base object to value
      **kwargs: see recursive_reduce
    """
    return recursive_reduce(
        struct=struct,
        func=func,
        reduce_op=max,
        **kwargs
    )
