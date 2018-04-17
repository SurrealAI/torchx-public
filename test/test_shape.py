from test.utils import *


class GetSlice:
    "Easily get slice object"
    def __getitem__(self, item):
        return item


G = GetSlice()


def test_view():
    def check_view(input_shape, *view_args):
        inferred = U.shape_view(input_shape, *view_args)
        correct = torch.zeros(input_shape).view(*view_args).size()
        assert inferred == correct, (inferred, correct)
        print(inferred)

    check_view([111, 40, 50], 8, 37, -1)
    check_view([111, 40, 50], -1, 3, 10)
    check_view([111, 40, 50], 100, -1, 222)
    check_view([111, 40, 50], 250, 2, 444)


def test_slice():
    def check_slice(input_shape, slice):
        inferred = U.shape_slice(input_shape, slice)
        correct = torch.zeros(input_shape)[slice].size()
        assert inferred == correct, (inferred, correct)
        print(inferred)

    check_slice([50, 60, 17], G[3:-1:3, 0, :])
    check_slice([50, 6, 170], G[..., 2:160:3])

