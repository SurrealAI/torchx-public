# Installation

PyTorch is great. PyTorch on steroids is even better. 

This README is incomplete. I have only documented the most essential features. 
More to come later.

```
git clone https://github.com/SurrealAI/TorchX.git torchx
pip install -e torchx/
```

# Keras-inspired API

One problem with PyTorch is that you have to specify the shapes for each module, even though some of the shape parameters can be inferred from upstream modules. This is especially annoying if you need to tune the network architecture. 

Consider this sequential CNN architecture for 10-way classification:

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 20, kernel_size=5, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(20, 30, kernel_size=7),  # 20 is redundant, can be inferred from above
    nn.BatchNorm2d(30),  # 30 from the previous conv
    nn.ReLU(),
    nn.Conv2d(30, 40, kernel_size=3),  # 30 can be inferred from second conv
    nn.BatchNorm2d(40),  # 40 from the previous conv
    nn.ReLU(),
    Flatten(),  # you have to write your own flatten
    # Dim after flatten has to be manually calculated (nontrivial!)
    # What's worse, every time you change some layer's architectural 
    # parameter above,  you will have to recalculate.
    nn.Linear(1960, 80),
    nn.ReLU(),
    nn.Linear(80, 10),  # 80 is once again redundant
)

x = torch.randn((8, 3, 32, 32))
y = model(x)
print(y.size())  # (8, 10)
```

## TorchX Layer class

TorchX features a shape inference engine. The modules will not be instantiated until they have enough information to be constructed. To accomplish this, TorchX provides the `layers` package that contains most of the modules in `torch.nn`, but are wrapped as subclasses of `torchx.layers.Layer` instead. 

All `Layer`s inherit from the standard `nn.Module`, so they are perfectly interoperable with PyTorch in case you'd like to switch back and forth. Except for the "Functional API" (discussed later), TorchX `Layer`s can be interleaved with standard modules when you define your own `nn.Module`. 

What's more, you can always retrieve the underlying torch module by `mylayer.native_module()`

To use a single layer:

```python
import torchx.layers as L

x = torch.zeros((16, 10))  # batch size 16, input feature size 10
model = L.Linear(20)  # output feature size 20
y = model(x) # model weight and bias are lazily instantiated when you invoke it
print(y.size())  # (16, 20)
```

## TorchX Sequential API

Just like the builtin `torch.nn.Sequential`, TorchX features a Sequential container that eliminates the tedious shape tracking once and for all.

We take the CNN architecture in the previous section and rewrite it with TorchX:

```python
import torchx.layers as L

model = L.Sequential(
    L.Conv2d(20, kernel_size=5, stride=2, padding=1),
    L.ReLU(),
    L.Conv2d(30, kernel_size=7),  # just tell me the output channel size
    L.BatchNorm2d(),  # input channel dim is inferred
    L.ReLU(),
    L.Conv2d(40, kernel_size=3),
    L.BatchNorm2d(),
    L.ReLU(),
    L.Flatten(),  # output dim after flatten is calculated by TorchX
    L.Linear(80),  # just tell me the hidden size! 
    L.ReLU(),
    L.Linear(10),
)

x = torch.randn((8, 3, 32, 32))
y = model(x)
print(y.size())  # (8, 10)
```

No sweat! 

## TorchX Functional API

Now we want to define more complex connectivity than `nn.Sequential`, such as multi-input multi-output models, directed acyclic graphs, or models with shared layers. In standard PyTorch, you typically have to follow 3 steps:

1. Subclass `nn.Module`.
2. In `__init__()`, define all the layers with learnable parameters as class attributes. You have to manually declare or calculate all the shapes upfront. 
3. Override `forward()` method to specify the connectivity of your network.
 
This design gives rise to redundancy and inconvenience, especially when you want to change any significant part of the architecture. You will have to update the attribute declarations in `__init__`, recalculate the shapes, and make sure the corresponding lines in `forward()` are kept consistent. 

We illustrate with a diamond-shaped CNN followed by 2 FCs:

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # note the unnecessary shape parameter duplications, similar to Sequential
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5, stride=2)
        # diamond edges:
        self.conv2_1 = nn.Conv2d(10, 30, kernel_size=7, padding=2)
        self.conv2_2 = nn.Conv2d(10, 30, kernel_size=5, padding=1)
        self.fc1 = nn.Linear(4320, 80)  # 4320 is a non-trivial calculate!
        self.fc2 = nn.Linear(80, 10)

    def forward(self, x):
        x = self.conv1(x)
        branch1 = self.conv2_1(x)
        branch2 = self.conv2_2(x)
        x = branch1 * branch2
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc1(x)
        x = self.fc2(x)
        return x


model = MyModel()

x = torch.randn((8, 3, 32, 32))
y = model(x)
print(y.size())  # (8, 10)
```

Now let's attempt to tune the architecture.

Suppose you want to change the output channel of `conv1` to 20, you will have to change _three_ places in the code: `conv1`'s output channel, `conv2_1`'s input channel, and `conv2_2`'s input channel. 

To add another FC layer between `fc1` and `fc2`, you will have to change _four_ places: `fc1`'s output dim, `fc2`'s input dim, define a new `fc3` in `__init__`, and finally add `x = self.fc3(x)` to `forward()`.

Because the module definitions and their connectivity are decoupled, you will have to scroll back and forth between `__init__` and `forward` to know what exactly are `conv2_1` and `fc3`. This is not a problem here, but would be a headache in bigger modules that span hundreds of lines. 

In TorchX, we introduce the **functional API** to automate shape deduction and bring module definitions and connectivity together. We start with a lightweight markup object, `Placeholder` that represents the input tensor shape. 

```python
import torchx.layers as L

# input image shape: (batch_size, channel, H, W)
xp_input = L.Placeholder((8, 3, 32, 32))

# definition and connectivity right next to each other!
# layers take a placeholder and return another placeholder
xp = L.Conv2d(10, kernel_size=5, stride=2)(xp_input)
branch1 = L.Conv2d(30, kernel_size=7, padding=2)(xp)  # no need to specify input channel
branch2 = L.Conv2d(30, kernel_size=5, padding=1)(xp)
xp = branch1 * branch2
xp = xp.flatten()
xp = L.Linear(80)(xp)  # no need to calculate the flattened shape
xp = L.Linear(10)(xp)

# `inputs` and `outputs` keywords can each take a single placeholder, 
# a list, or even a dict of placeholders. 
# this defines the signature of model.__call__()
model = L.Functional(inputs=xp_input, outputs=xp)

# model is now fully instantiated, we can give it real tensors
x = torch.randn((8, 3, 32, 32))
y = model(x)
print(y.size())  # (8, 10)
```

The functional API makes updating architecture so much easier. Let's repeat the exercise above and note the difference from standard PyTorch:
 
To change the output channel of `branch1` to 20, you only need to touch one line:

```python
xp = L.Conv2d(20, kernel_size=5, stride=2)(xp_input)  # only change
branch1 = L.Conv2d(30, kernel_size=7, padding=2)(xp)  # exactly the same as before
branch2 = L.Conv2d(30, kernel_size=5, padding=1)(xp)  # same
```

To add another FC layer between `fc1` and `fc2`, just add one line:
```python
xp = L.Linear(80)(xp)  # same as before
xp = L.Linear(50)(xp)  # add a Linear layer of 50 hidden units
xp = L.Linear(10)(xp)  # same
```


## Non-standard layers

### TimeDistributed

TODO (give example)

This container is useful for RNNs. It applies a sequence of layers to every temporal slice of an input.

The input should be at least 3D, and the dimension at index one will be considered to be the temporal dimension: `(batch_size, time_steps, ...features...)`

### Lambda

TODO

