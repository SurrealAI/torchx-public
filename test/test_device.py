from test.utils import *
from torchx.device import *


def test_devices_to_ids():
    assert ids_to_devices(None) == [torch.device('cpu')]
    assert ids_to_devices('cuda:4') == [torch.device('cuda:4')]
    if not has_cuda():
        assert ids_to_devices('cuda:all') == [torch.device('cpu')]
    assert ids_to_devices(7) == [torch.device('cuda:7')]
    assert ids_to_devices(-1) == [torch.device('cpu')]
    assert ids_to_devices([0, 'cuda:5', torch.device('cpu')]) == \
           [torch.device('cuda:0'), torch.device('cuda:5'), torch.device('cpu')]


def test_device_scope():
    D = torch.device
    with device_scope('cuda:0', fallback=False):
        assert get_devices_in_scope() == [D('cuda:0')]
    with device_scope(-1):
        assert get_devices_in_scope() == [CPU_DEVICE]
    with device_scope(None):
        assert get_devices_in_scope() == [CPU_DEVICE]

    with device_scope([3, 1, 0, 2], fallback=False):
        assert get_devices_in_scope() == [D('cuda:3'), D('cuda:1'), D('cuda:0'), D('cuda:2')]

    with device_scope([3, 1], fallback=False):
        with device_scope(2, fallback=False):
            assert get_devices_in_scope() == [D('cuda:2')]

    with device_scope([3, 1], fallback=False):
        with device_scope(2, fallback=False, override_parent=False):
            assert get_devices_in_scope() == [D('cuda:3'), D('cuda:1')]
