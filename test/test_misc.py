from test.utils import *


def test_save_init_meta():
    class MyObj(metaclass=U.SaveInitArgsMeta):
        def __init__(self, a, b, c=30, *, d=-5, e, **mykwargs):
            pass

    a = MyObj(10, 20, e=-10, f1='otherf1', f2='otherf2')
    assert a.init_args_dict == {
        None: [10, 20],
        'e': -10, 'f1': 'otherf1', 'f2': 'otherf2'
    }
