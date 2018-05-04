from test.utils import *


def test_flatten():
    x = new_tensor((3, 4, 2, 5))
    assert nnx.th_flatten(x).size() == (3, 4*2*5)
    assert nnx.th_flatten(x, start_dim=0).size() == (3*4*2*5, )
    assert nnx.th_flatten(x, start_dim=2).size() == (3, 4, 2*5)
    assert nnx.th_flatten(x, start_dim=3).size() == (3, 4, 2, 5)
    with pytest.raises(ValueError):
        nnx.th_flatten(x, start_dim=4)
