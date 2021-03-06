import sys
import os
import inspect
import collections
import functools
import re
import pprint
from enum import Enum, EnumMeta


def _get_qualified_type_name(type_):
    name = str(type_)
    r = re.compile("<class '(.*)'>")
    match = r.match(name)
    if match:
        return match.group(1)
    else:
        return name


def assert_type(x, expected_type, message=''):
    assert isinstance(x, expected_type), (
        (message + ': ' if message else '')
        + 'expected type `{}`, actual type `{}`'.format(
            _get_qualified_type_name(expected_type),
            _get_qualified_type_name(type(x))
        )
    )
    return True


def print_(*objs, h='', **kwargs):
    """
    Args:
      *objs: objects to be pretty-printed
      h: header string
      **kwargs: other kwargs to pass on to ``pprint()``
    """
    if h:
        print('=' * 20, h, '=' * 20)
    for obj in objs:
        print(pprint.pformat(obj, indent=4, **kwargs))
    if h:
        print('=' * (42 + len(h)))


class _GetItemEnumMeta(EnumMeta):
    """
    Hijack the __getitem__ method from metaclass, because subclass cannot
        override magic methods. More informative error message.
    """
    def __getitem__(self, option):
        enum_class = None
        for v in self.__members__.values():
            enum_class = v.__class__
            break
        assert enum_class is not None, \
            'must have at least one option in StringEnum'
        return get_enum(enum_class, option)


class StringEnum(Enum, metaclass=_GetItemEnumMeta):
    """
    https://docs.python.org/3.4/library/enum.html#duplicatefreeenum
    The created options will automatically have the same string value as name.

    Support [] subscript, i.e. MyFruit['orange'] -> MyFruit.orange
    """
    def __init__(self, *args, **kwargs):
        self._value_ = self.name


def create_string_enum(class_name, option_names):
    assert_type(option_names, str)
    assert_type(option_names, list)
    return StringEnum(class_name, option_names)


def get_enum(enum_class, option):
    """
    Args:
        enum_class:
        option: if the value doesn't belong to Enum, throw error.
            Can be either the str name or the actual enum value
    """
    assert issubclass(enum_class, StringEnum)
    if isinstance(option, enum_class):
        return option
    else:
        assert_type(option, str)
        option = option.lower()
        options = enum_class.__members__
        if option not in options:
            raise ValueError('"{}" is not a valid option for {}. '
                             'Available options are {}.'
             .format(option, enum_class.__name__, list(options)))
        return options[option]


def fformat(float_num, precision):
    """
    https://stackoverflow.com/a/44702621/3453033
    """
    assert isinstance(precision, int) and precision > 0
    return ('{{:.{}f}}'
            .format(precision)
            .format(float_num)
            .rstrip('0')
            .rstrip('.'))


def is_sequence(obj):
    """
    Returns:
      True if the sequence is a collections.Sequence and not a string.
    """
    return (isinstance(obj, collections.Sequence)
            and not isinstance(obj, str))


def include_keys(include, d):
    """
    Pick out the `include` keys from a dict

    Args:
      include: list or set of keys to be included
      d: raw dict that might have irrelevant keys
    """
    assert is_sequence(include)
    return {k: v for k, v in d.items() if k in set(include)}


def exclude_keys(exclude, d):
    """
    Remove the `exclude` keys from a kwargs dict.

    Args:
      exclude: list or set of keys to be excluded
      d: raw dict that might have irrelevant keys
    """
    assert is_sequence(exclude)
    return {k: v for k, v in d.items() if k not in set(exclude)}


def iter_last(iterable):
    """
    For processing the last element differently
    Yields: (is_last=bool, element)
    """
    length = len(iterable)
    return ((i == length-1, x) for i, x in enumerate(iterable))


def case_insensitive_match(items, key):
    """
    Args:
        items: iterable of keys
        key: search for the key case-insensitively

    Returns:
        matched original key (with cases), None if no match
    """
    for k in items:
        if k.lower() == key.lower():
            return k
    return None


def _get_bound_args(func, *args, **kwargs):
    """
    https://docs.python.org/3/library/inspect.html#inspect.BoundArguments
    def f(a, b, c=5, d=6): pass
    get_bound_args(f, 3, 6, d=100) -> {'a':3, 'b':6, 'c':5, 'd':100}

    Returns:
        OrderedDict of bound arguments
    """
    arginfo = inspect.signature(func).bind(*args, **kwargs)
    arginfo.apply_defaults()
    return arginfo.arguments


class _Deprecated_SaveInitArgsMeta(type):
    """
    Bounded arguments:
    https://docs.python.org/3/library/inspect.html#inspect.BoundArguments

    Store the captured constructor arguments to <instance>._init_args_dict
    as OrderedDict. Can be retrieved by the property method <obj>.init_args_dict
    Includes both the positional args (with the arg name) and kwargs
    """
    def __init__(cls, name, bases, attrs):
        # WARNING: must add class method AFTER super.__init__
        # adding attrs['new-method'] before __init__ has no effect!
        super().__init__(name, bases, attrs)
        @property
        def init_args_dict(self):
            return self._init_args_dict
        cls.init_args_dict = init_args_dict

    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        try:
            obj._init_args_dict = _get_bound_args(obj.__init__, *args, **kwargs)
        except TypeError:  # __init__ has special stuff like *args
            obj._init_args_dict = None
        return obj


class SaveInitArgsMeta(type):
    """
    Store __init__ call args in self.init_args_dict
    Positional args to __init__ will be stored under `None` key
    Keyword args to __init__ will be stored as-is in init_args_dict
    """
    def __init__(cls, name, bases, attrs):
        # WARNING: must add class method AFTER super.__init__
        # adding attrs['new-method'] before __init__ has no effect!
        super().__init__(name, bases, attrs)
        @property
        def init_args_dict(self):
            return self._init_args_dict
        cls.init_args_dict = init_args_dict

    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        init_dict = {}
        if args:
            init_dict[None] = list(args)  # convert to list for YAML compatibility
        init_dict.update(kwargs)
        obj._init_args_dict = init_dict
        return obj


class SaveInitArgs(metaclass=SaveInitArgsMeta):
    """
    Either use metaclass hook:
        class MyObj(metaclass=SaveInitArgsMeta)
    or simply inherit
        class MyObj(SaveInitArgs)
    """
    pass


class AutoInitializeMeta(type):
    """
    Call the special method ._initialize() after __init__.
    Useful if some logic must be run after the object is constructed.
    For example, the following code doesn't work because `self.y` does not exist
    when super class calls self._initialize()

    class BaseClass():
        def __init__(self):
            self._initialize()

        def _initialize():
            self.x = self.get_x()

        def get_x(self):
            # abstract method that only subclass

    class SubClass(BaseClass):
        def __init__(self, y):
            super().__init__()
            self.y = y

        def get_x(self):
            return self.y * 3

    Fix:
    class BaseClass(metaclass=AutoInitializeMeta):
        def __init__(self):
            pass
            # self._initialize() is now automatically called after __init__

        def _initialize():
            print('INIT', self.x)

        def get_x(self):
            # abstract method that only subclass
            raise NotImplementedError
    """
    def __call__(self, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        assert hasattr(obj, '_initialize'), \
            'AutoInitializeMeta requires that subclass implements _initialize()'
        obj._initialize()
        return obj


class noop_context:
    """
    Placeholder context manager that does nothing.
    We could have written simply as:

    @contextmanager
    def noop_context(*args, **kwargs):
        yield

    but the returned context manager cannot be called twice, i.e.
    my_noop = noop_context()
    with my_noop:
        do1()
    with my_noop: # trigger generator error
        do2()
    """
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def meta_wrap(decor):
    """
    a decorator decorator, allowing the wrapped decorator to be used as:
    @decorator(*args, **kwargs)
    def callable()
      -- or --
    @decorator  # without parenthesis, args and kwargs will use default
    def callable()

    Args:
      decor: a decorator whose first argument is a callable (function or class
        to be decorated), and the rest of the arguments can be omitted as default.
        decor(f, ... the other arguments must have default values)

    Warning:
      decor can NOT be a function that receives a single, callable argument.
      See stackoverflow: http://goo.gl/UEYbDB
    """
    single_callable = (lambda args, kwargs:
                       len(args) == 1 and len(kwargs) == 0 and callable(args[0]))
    @functools.wraps(decor)
    def new_decor(*args, **kwargs):
        if single_callable(args, kwargs):
            # this is the double-decorated f.
            # It should not run on a single callable.
            return decor(args[0])
        else:
            # decorator arguments
            return lambda real_f: decor(real_f, *args, **kwargs)

    return new_decor


@meta_wrap
def deprecated(func, msg='', action='warning'):
    """
    Function/class decorator: designate deprecation.

    Args:
      msg: string message.
      action: string mode
      - 'warning': (default) prints `msg` to stderr
      - 'noop': do nothing
      - 'raise': raise DeprecatedError(`msg`)
    """
    action = action.lower()
    if action not in ['warning', 'noop', 'raise']:
        raise ValueError('unknown action type {}'.format(action))
    if not msg:
        msg = 'This is a deprecated feature.'

    # only does the deprecation when being called
    @functools.wraps(func)
    def _deprecated(*args, **kwargs):
        if action == 'warning':
            print(msg, file=sys.stderr)
        elif action == 'raise':
            raise DeprecationWarning(msg)
        return func(*args, **kwargs)
    return _deprecated


def pack_varargs(args):
    """
    Pack *args or a single list arg as list

    def f(*args):
        arg_list = pack_varargs(args)
        # arg_list is now packed as a list
    """
    assert isinstance(args, tuple), 'please input the tuple `args` as in *args'
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        return args[0]
    else:
        return args


def enable_list_arg(func):
    """
    Function decorator.
    If a function only accepts varargs (*args),
    make it support a single list arg as well
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = pack_varargs(args)
        return func(*args, **kwargs)
    return wrapper


def enable_varargs(func):
    """
    Function decorator.
    If a function only accepts a list arg, make it support varargs as well
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = pack_varargs(args)
        return func(args, **kwargs)
    return wrapper


def pack_kwargs(args, kwargs):
    """
    Pack **kwargs or a single dict arg as dict

    def f(*args, **kwargs):
        kwdict = pack_kwargs(args, kwargs)
        # kwdict is now packed as a dict
    """
    if len(args) == 1 and isinstance(args[0], dict):
        assert not kwargs, 'cannot have both kwargs and a dict arg'
        return args[0]  # single-dict
    else:
        assert not args, 'cannot have positional args if kwargs exist'
        return kwargs


def enable_dict_arg(func):
    """
    Function decorator.
    If a function only accepts varargs (*args),
    make it support a single list arg as well
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs = pack_kwargs(args, kwargs)
        return func(**kwargs)
    return wrapper


def enable_kwargs(func):
    """
    Function decorator.
    If a function only accepts a dict arg, make it support kwargs as well
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs = pack_kwargs(args, kwargs)
        return func(kwargs)
    return wrapper


def method_decorator(decorator):
    """
    Decorator of decorator: transform a decorator that only works on normal
    functions to a decorator that works on class methods
    From Django form: https://goo.gl/XLjxKK
    """
    @functools.wraps(decorator)
    def wrapped_decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            def bound_func(*args2, **kwargs2):
                return method(self, *args2, **kwargs2)
            return decorator(bound_func)(*args, **kwargs)
        return wrapper
    return wrapped_decorator


def accepts_varargs(func):
    """
    If a function accepts *args
    """
    params = inspect.signature(func).parameters
    return any(param.kind == inspect.Parameter.VAR_POSITIONAL
               for param in params.values())


def accepts_kwargs(func):
    """
    If a function accepts **kwargs
    """
    params = inspect.signature(func).parameters
    return any(param.kind == inspect.Parameter.VAR_KEYWORD
               for param in params.values())


def is_signature_compatible(func, *args, **kwargs):
    sig = inspect.signature(func)
    try:
        sig.bind(*args, **kwargs)
        return True
    except TypeError:
        return False

