# Refactor the model with classes

Going to write each of the `layers` as a class, going to write `__call__` implementations. What does `__call__` do?


## Aside: `__call__`:

From geeksforgeeks:

```python
class A:
    def __init__(self):
        print("created class")

    def __call__(self):
        print("hi")

z = A()
z()
# created class
# hi
```

## Comments about the Backward Pass

The backward pass needs some information **stored** in the class to do the calculus calculations. This is why some data will be stored in each of these classes (vs. pure `python` functions)

```python
class Relu():
    """
    Attributes:
        inp: stores the input passed
        out: stores the output 
    """
    def __call__(self, inp):
        """
        Args:
            inp: torch.Tensor
        """
        self.inp = inp
        self.out = inp.clamp_min(0.)
        return self.out
    
    def backward(self):
        # the gradient calculation, update the input
        # implementing chain rule
        self.inp.g = (self.inp > 0).float() * self.out.g
```

Implementing the linear layer the same way:

- as a class
- storing values within the layer
- including the forward pass function
- and the backwards pass function

```python
class Lin():
    def __init__(self, w: torch.Tensor, b: torch.Tensor):
        """
        for the general equation:
            z = wx + b

        w: layer weights
        b: layer bias 
        inp: the batch / input data
        """
        self.w = w
        self.b = b
        self.inp = None  # will be set during call
        self.out = None  # will be set during call

    @staticmethod
    def lin(x, w, b):
        """
        x: the batch / data input
        w: the layer weights
        b: the layer bias
        """
        return x @ w + b

    def __call__(self, inp):
        """Does a standard forward pass
        z = wx + b

        """
        self.inp = inp
        self.out = self.lin(inp, self.w, self.b)
        return self.out

    def backward(self):
        """
        Does the backward gradient calculation
        """
        self.inp.g = self.out.g @ self.w.t()
        self.w.g = self.inp.t() @ self.out.g
        self.b.g = self.out.g.sum(0)
```

Implementing Mean Squared Error (MSE) in the class approach:

- implementing forward + backwards in a single class
- including data storage

```python
class Mse():
    def __init__(self):
        self.inp = None
        self.targ = None
        self.out = None

    def __call__(self, inp, targ):
        self.inp,self.targ = inp,targ
        self.out = self.mse(inp, targ)
        return self.out
    
    def backward(self):
        self.inp.g = 2. * (self.inp.squeeze() - self.targ).unsqueeze(-1) / self.targ.shape[0]

    @staticmethod
    def mse(output, targ):
        return (output[:,0]-targ).pow(2).mean()
```

Implementing `Model`

```python
class Model():
    """A simple 2 layered model"""

    def __init__(self, w1, b1, w2, b2):
        """
        w1: torch.TensorType
        b1: torch.TensorType
        w2: torch.TensorType
        b2: torch.TensorType
        """
        self.layers = [
            Lin(w1, b1),
            Relu(),
            Lin(w2, b2),
        ]
        self.loss = Mse()
        
    def __call__(self, x, targ):
        for l in self.layers:
            x = l(x)
        return self.loss(x, targ)
    
    def backward(self):
        """run the backwards prop for each layer"""
        self.loss.backward()
        for l in reversed(self.layers):
            l.backward()
```


# How can we generalize this even more?

Looking at the class definitions above, there are some common implementation patterns in all the layer classes.

- all have `__call__` implementations
- all have `backward` calculus implementations

Let's generalize with a super class

```python
# abstract base class
from abc import ABC, abstractmethod

class Module(ABC):
    def __call__(self, *args):
        self.args = args
        self.out = self.forward(*args)
        return self.out

    @abstractmethod
    def forward(self):
        pass

    def backward(self):
        self.bwd(self.out, *self.args)

    @abstractmethod
    def bwd(self):
        pass
```

## Now re-implement all the above layers inheriting from our module class

Now if all the above layers are re-implemented by inheriting our `Module`, some of the code clutter can be reduced:

```python
class Relu(Module):
    def forward(self, inp):
        return inp.clamp_min(0.)

    def bwd(self, out, inp):
        inp.g = (inp>0).float() * out.g


class Lin(Module):
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def forward(self, inp):
        return inp @ self.w + self.b 

    def bwd(self, out, inp):
        inp.g = self.out.g @ self.w.t()
        self.w.g = inp.t() @ self.out.g
        self.b.g = self.out.g.sum(0)


class Mse(Module):
    def forward (self, inp, targ):
        return (inp.squeeze() - targ).pow(2).mean()

    def bwd(self, out, inp, targ):
        inp.g = 2*(inp.squeeze()-targ).unsqueeze(-1) / targ.shape[0]
```

# How is the done in pytorch? `Autograd`!

```python
from torch import nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.w = torch.randn(n_in,n_out).requires_grad_()
        self.b = torch.zeros(n_out).requires_grad_()

    def forward(self, inp):
        return inp @ self.w + self.b


class Model(nn.Module):
    def __init__(self, n_in, nh, n_out):
        super().__init__()
        self.layers = [
            Linear(n_in,nh),
            nn.ReLU(),
            Linear(nh,n_out)
        ]
        
    def __call__(self, x, targ):
        for l in self.layers:
            x = l(x)
        return F.mse_loss(x, targ[:, None])

model = Model(m, nh, 1)
loss = model(x_train, y_train)
loss.backward()

l0 = model.layers[0]
l0.b.grad
```
