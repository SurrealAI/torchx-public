from torch import nn


def get_same_shape_layers():
    same_shape_layers = []
    same_shape_modules = [nn.modules.dropout, nn.modules.activation, nn.modules.batchnorm, nn.modules.instancenorm]
    no_include = ['Parameter', 'F', 'Module']
    for module in same_shape_modules:
        for attr in dir(module):
            if attr[0].isupper() and attr not in no_include:
                same_shape_layers.append(getattr(module, attr))
    return same_shape_layers


if __name__ == '__main__':
    print(get_same_shape_layers())
