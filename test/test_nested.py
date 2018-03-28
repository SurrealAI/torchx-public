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


def test_recursive_combine():
    struct1 = {
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
    struct2 = {
        'a': [
            {'a1': 30},
            MyMap({'a2': MyList([40, 50, 60]),
                   'a3': None}),
        ],
        'b': MyMap({'b1': (70, 80)}),
        'c': 90,
        'd': (None, 100),
        'e': OrderedDict([
            ('e6', [110, 120]),
            ('e3', MyList([130, 140])),
            ('e5', 150),
            ('e1', -10),
            ('e7', -20),
            ('e2', None),
        ])
    }
    struct3 = {
        'a': [
            {'a1': 300},
            MyMap({'a2': MyList([400, 500, 600]),
                   'a3': None}),
        ],
        'b': MyMap({'b1': (700, 800)}),
        'c': 900,
        'd': (None, 1000),
        'e': OrderedDict([
            ('e6', [1100, 1200]),
            ('e3', MyList([1300, 1400])),
            ('e5', 1500),
            ('e1', -100),
            ('e7', -200),
            ('e2', None),
        ])
    }

    ans = U.recursive_combine(
        struct1, struct2, struct3,
        combinator=lambda xs: sum(xs) if xs[0] else None,
        is_base=None,
        allow_any_seq_type=True,
        allow_any_dict_type=True,
    )
    assert ans == {'a': [{'a1': 333}, MyMap({'a2': MyList([444, 555, 666]), 'a3': None})],
                   'b': MyMap({'b1': (777, 888)}),
                   'c': 999,
                   'd': (None, 1110),
                   'e': OrderedDict([('e6', [1221, 1332]),
                                     ('e3', MyList([1443, 1554])),
                                     ('e5', 1665),
                                     ('e1', -111),
                                     ('e7', -222),
                                     ('e2', None)])}

    ans = U.recursive_zip(
        struct1, struct2, struct3,
    )
    assert ans == {
        'a': [{'a1': (3, 30, 300)},
              MyMap({
                        'a2': MyList(
                            [(4, 40, 400), (5, 50, 500), (6, 60, 600)]), 'a3': (
                  None, None, None)
                    })],
        'b': MyMap({'b1': ((7, 70, 700), (8, 80, 800))}),
        'c': (9, 90, 900),
        'd': ((None, None, None), (10, 100, 1000)),
        'e': OrderedDict([('e6', [(11, 110, 1100), (12, 120, 1200)]),
                          ('e3', MyList([(13, 130, 1300), (14, 140, 1400)])),
                          ('e5', (15, 150, 1500)),
                          ('e1', (-1, -10, -100)),
                          ('e7', (-2, -20, -200)),
                          ('e2', (None, None, None))])
    }

run_all_tests(globals())
