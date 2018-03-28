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


def _unknown_type_handler(struct, handler):
    if handler is None:
        raise ValueError('Unknown structure type: {}'.format(type(struct)))
    else:
        return handler(struct)


def recursive_map(struct,
                  func,
                  is_base=None,
                  allow_any_seq_type=True,
                  allow_any_dict_type=True,
                  unknown_type_handler=None,
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
      unknown_type_handler: function that handles an unknown struct 
        default None: raise Exception
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
                unknown_type_handler=unknown_type_handler,
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
                unknown_type_handler=unknown_type_handler,
                leave_none=leave_none
            )
        return return_dict
    elif leave_none and struct is None:
        return None
    elif is_base is None:  # pass all non-Sequence and non-dict objects
        return func(struct)
    else:  
        # if is_base is not None and struct is not Sequence or dict or base object
        return _unknown_type_handler(struct, unknown_type_handler)


def recursive_reduce(struct,
                     reduce_op,
                     func=None,
                     is_base=None,
                     allow_any_seq_type=True,
                     allow_any_dict_type=True,
                     unknown_type_handler=None):
    """
    Recursively walks a data structure (can be Sequence or dict), maps each
    base object to func(obj), and reduce the results over sequence or dict.values()

    Args:
      struct: recursive data structure
      reduce_op: must be an associative operator that takes a *generator* of values
        and returns a single value. Should be able to handle the case where the
        generator only has one value.
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
        return reduce_op(
            recursive_reduce(
                struct=value,
                reduce_op=reduce_op,
                func=func,
                is_base=is_base,
                allow_any_seq_type=allow_any_seq_type,
                allow_any_dict_type=allow_any_dict_type,
                unknown_type_handler=unknown_type_handler,
            )
            for value in values
        )
    elif is_base is None:  # pass all non-Sequence and non-dict objects
        return func(struct)
    else:
        # if is_base is not None and struct is not Sequence or dict or base object
        return _unknown_type_handler(struct, unknown_type_handler)


def recursive_flatten(struct, **kwargs):
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


class StructureMismatch(Exception):
    pass


def recursive_combine(*structs,
                      combinator,
                      is_base=None,
                      allow_any_seq_type=True,
                      allow_any_dict_type=True,
                      unknown_type_handler=None):
    """
    Combines multiple nested structs with a combinator function.
    All structures must match: seq of same lengths, dict of same keys, etc.

    Args:
      *structs: multiple nested structs
      combinator: f(tuple of base objects) -> combined value
      see `recursive_reduce` for other args
    """
    assert len(structs) >= 1, 'must have at least one struct'
    s0 = structs[0]
    if is_base and is_base(s0):
        return combinator(structs)
    elif _is_sequence(s0, allow_any_seq_type):
        for s in structs[1:]:
            if not _is_sequence(s, allow_any_seq_type):
                raise StructureMismatch('should all be sequences', s0, s)
            if len(s) != len(s0):
                raise StructureMismatch('unequal sequence length', len(s0), len(s))
        return_seq = [
            recursive_combine(
                *s_zipped,
                combinator=combinator,
                is_base=is_base,
                allow_any_seq_type=allow_any_seq_type,
                allow_any_dict_type=allow_any_dict_type,
                unknown_type_handler=unknown_type_handler,
            )
            for s_zipped in zip(*structs)
        ]
        return type(s0)(return_seq)
    elif _is_mapping(s0, allow_any_dict_type):
        for s in structs[1:]:
            if not _is_mapping(s, allow_any_dict_type):
                raise StructureMismatch('should both be mappings', s0, s)
            if len(s) != len(s0):
                raise StructureMismatch('unequal dict size', len(s0), len(s))
        # not using dict comprehension because if the struct is OrderedDict,
        # the return value should also retain order
        return_dict = type(s0)()
        for key0, value0 in s0.items():
            value_zipped = [value0]
            for s in structs[1:]:
                if key0 not in s:
                    raise StructureMismatch('struct key `{}` not in {}'.format(key0, s))
                value_zipped.append(s[key0])
            return_dict[key0] = recursive_combine(
                *value_zipped,
                combinator=combinator,
                is_base=is_base,
                allow_any_seq_type=allow_any_seq_type,
                allow_any_dict_type=allow_any_dict_type,
                unknown_type_handler=unknown_type_handler,
            )
        return return_dict
    elif is_base is None:  # pass all non-Sequence and non-dict objects
        return combinator(structs)
    else:
        # if is_base is not None and s0 is not Sequence or dict or base object
        return _unknown_type_handler(s0, unknown_type_handler)


def recursive_zip(*structs, **kwargs):
    """
    Combines each base object in two structs into tuple (obj1, obj2)

    Args:
      see `recursive_combine` for other args
    """
    return recursive_combine(
        *structs,
        combinator=lambda s_tuple: s_tuple,
        **kwargs
    )


def recursive_binary_combine(struct1, struct2,
                             *, combinator,
                             **kwargs):
    """
    Binary version of `recursive_combine`. The only difference is that the
    combinator function now takes 2 args and returns the combined value

    Args:
      combinator: (x, y) -> combined
    """
    return recursive_combine(
        struct1, struct2,
        combinator=lambda xy: combinator(xy[0], xy[1]),
        **kwargs
    )


def recursive_compare(struct1, struct2,
                      comparator=None,
                      mode='all',
                      **kwargs):

    """
    Compares two nested structs

    Args:
      comparator: (obj1, obj2) -> True/False
      mode: 'all' or 'any'
      see `recursive_combine` for other args
    """
    reducer = {
        'all': recursive_all,
        'any': recursive_any
    }
    assert mode in reducer
    reducer = reducer[mode]
    cmp_results = recursive_binary_combine(
        struct1, struct2,
        combinator=comparator,
        **kwargs
    )
    return reducer(cmp_results)
