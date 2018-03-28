from test.utils import *


class MyList(Sequence):
    def __init__(self, L):
        self.L = L

    def __iter__(self):
        return iter(self.L)

    def __len__(self):
        return len(self.L)

    def __getitem__(self, item):
        return self.L[item]

    def __repr__(self):
        return 'MyList({})'.format(self.L)

    def __eq__(self, other):
        return self.L == other.L


class MyMap(MutableMapping):
    def __init__(self, L=None):
        self.L = L if L else {}

    def __iter__(self):
        return iter(self.L)

    def __len__(self):
        return len(self.L)

    def __getitem__(self, item):
        return self.L[item]

    def __setitem__(self, key, value):
        self.L[key] = value

    def __delitem__(self, key):
        del self.L[key]

    def __repr__(self):
        return 'MyMap({})'.format(self.L)

    def __eq__(self, other):
        return self.L == other.L


def test_recursive_map():
    struct = {
        'a': [
            {'a1': 3},
            MyMap({'a2': MyList([4, 5, 6]),
                 'a3': None}),
        ],
        'b': MyMap({'b1': (7, 8)}),
        'c': 9,
        'd': (None, 10),
        'e': OrderedDict([
            ('e6', [11, 12]),
            ('e3', MyList([13, 14])),
            ('e5', 15),
            ('e1', -1),
            ('e7', -2),
            ('e2', None),
        ])
    }

    ans = U.recursive_map(
        struct,
        lambda x: x ** 2,
        # is_base=lambda x: isinstance(x, int),
        is_base=None,
        allow_any_seq_type=True,
        allow_any_dict_type=True,
        leave_none=True
    )
    pprint(ans)

    correct = {
        'a': [{'a1': 9}, MyMap({'a2': MyList([16, 25, 36]), 'a3': None})],
        'b': MyMap({'b1': (49, 64)}), 'c': 81, 'd': (None, 100),
        'e': OrderedDict(
            [('e6', [121, 144]), ('e3', MyList([169, 196])), ('e5', 225),
             ('e1', 1), ('e7', 4), ('e2', None)])
    }
    assert ans == correct

    ans = U.recursive_sum(
        struct,
        lambda x: abs(x) if x else 0,
        is_base=None,
        allow_any_seq_type=True,
        allow_any_dict_type=True,
    )
    assert ans == 120

    ans = U.recursive_max(
        struct,
        lambda x: abs(x) if x else 0,
        is_base=None,
        allow_any_seq_type=True,
        allow_any_dict_type=True,
    )
    assert ans == 15

    ans = U.recursive_any(
        struct,
        lambda x: x > 15 if x else False,
        is_base=None,
        allow_any_seq_type=True,
        allow_any_dict_type=True,
    )
    assert ans == False

    ans = U.recursive_all(
        struct,
        lambda x: x > -3 if x else True,
        is_base=None,
    )
    assert ans == True

    pprint(U.flatten(struct))


run_all_tests(globals())
