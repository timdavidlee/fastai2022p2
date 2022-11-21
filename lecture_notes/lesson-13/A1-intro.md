# Backpropogation

Much of the code for this lesson can be found here:

[https://github.com/fastai/course22p2/blob/master/nbs/03_backprop.ipynb](https://github.com/fastai/course22p2/blob/master/nbs/03_backprop.ipynb)

Imports for this lesson:

```python
import pickle, gzip  # for reading data
import time
import os, shutil # for navigating the file system
import math, numpy as np # for math operations
import matplotlib as mpl # for plotting
from pathlib import Path # pathing to files + folders

import torch
from torch import tensor
from fastcore.test import test_close
torch.manual_seed(42)

# sets the color scheme for plotting
mpl.rcParams['image.cmap'] = 'gray'
# adjusting line width + precision when viewing data
torch.set_printoptions(precision=2, linewidth=125, sci_mode=False)
np.set_printoptions(precision=2, linewidth=125)


path_data = Path('data')
path_gz = path_data/'mnist.pkl.gz'

# loading the mnist dataset again
with gzip.open(path_gz, 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(
        f, encoding='latin-1'
    )

# turning all of the arrays into pytorch tensors
x_train, y_train, x_valid, y_valid = map(
    tensor, [x_train, y_train, x_valid, y_valid]
)
```


## General Overview

-- hand written notes here --


## Setting up the matrix operations

```python
# from last time
# n = 50_000 samples
# m = 784 (28 x 28) the pixels into a single line
# c = 10 the number of classes, digits 0-9
n, m = x_train.shape
c = y_train.max() + 1
n, m, c

# number of hidden nodes (or rectified lines)
nh = 50
```

Lets setup linear operations

```sh
# create weight(w) and bias(b) matrix
w1 = torch.randn(m, nh)
b1 = torch.zeros(nh)

# 1 is used here to simplify the design, with a single output 
# will hope to predict a "2" if the image is a 2, and so forth using MSE
w2 = torch.randn(nh,1)
b2 = torch.zeros(1)

def linear(x, w, b):
    """our matrix multiply
    x: your input
    w: the weights
    b: the bias
    """
    return x @ w + b

def relu(x):
    """ any negative values will be 0
        relu(tensor([-1., 2., 3.])) -> tensor([0., 2., 3.])
        relu(tensor([0.25, 2., 3.])) -> tensor([0.25, 2.00, 3.00])
    """
    return x.clamp_min(0.)
```

Testing out the linear function for shape compatibility:

```python
# doing a quick test
print(x_valid.shape)
print(w1.shape)
linear(x_valid, w1, b1).shape

>>> torch.Size([10000, 784])
    torch.Size([784, 50])
    torch.Size([10000, 50])
```


## Defining a model's forward pass

```python
def model(x_batch, w1=w1, b1=b1, w2=w2, b2=b2):
    l1 = linear(x_batch, w1, b1)
    l2 = relu(l1)
    return linear(l2, w2, b2)

# remember, the definition of w2, outputs 1 dim
# 10,000 here is the number of valid records
res_valid = model(x_valid)
print(res_valid.shape)
>>> torch.Size([10000, 1])
```

### Defining Mean Squared Error (MSE)

Its acknowledge that `MSE` is not a proper choice for our problem. For now, consider how to construct the formula

- need to subtract the prediction - truth
- prediction out of the `model`, for the validation set is `[10_000, x 1]`
- truth is for the validation set is `[10_000]`
- expectation (for validation set) is `10_000 x 1`

This results in an incorrect shape

```python
(res_valid - y_valid).shape
>>> torch.Size([10000, 10000])
```

In the background, the last dimension is `1` from `10_000 x 1`, and since the truth has dim `[10_000]`, it was compatible and defaulted to broadcasting. So to avoid this:

- add dim to truth for the validation set is `[10_000]` -> `[10_000 x 1]` 
    - option 1 `matrix[:, None]`
- or remove dim in the prediction `[10_000, x 1]` -> `[10_000]`
    - option 2 `matrix[:, 0]`
    - option 3 `matrix.squeeze()`

```python
def mse(output, target):
    sq_err = (output - target[:, None]).pow(2)
    return sq_err.mean()

print(mse(res_valid, y_valid))
>>> tensor(1595.00)
```

## Gradients and Backward Pass

--handwritten notes--

```python
# library that will help doing the diff calculus for you
from sympy import symbols, diff

# define the symbols
x, y = symbols("x y")

# diff 3x^2 +9 with respect to X
diff(3 * x**2 + 9, x)
```

### Sympy - calculus helper

```python
# library that will help doing the diff calculus for you
from sympy import symbols, diff

# define the symbols
x, y = symbols("x y")

# diff 3x^2 +9 with respect to X
diff(3 * x**2 + 9, x)
>>> 6x
```

--[]--

## Calculate Grad funciton

```python
def linear_grad(inp, out, w, b):
    """Note that all the operations in this function
    are in-place, returns nothing"""
    
    # the input gradient(rate of change) is related to the
    # output's gradient (rate of change) by the weights, because 
    # of the linear layer
    inp.g = out.g @ w.t()
    
    # import pdb; pdb.set_trace()
    # then the weight's gradient is updated 
    w.g = (inp.unsqueeze(-1) * out.g.unsqueeze(1)).sum(dim=0)
    
    # derivate of the bias, is the derivate of the outputs
    b.g = out.g.sum(dim=0)
    
    
def forward_and_backward(inp, targ, w1, b1, w2, b2):
    # forward pass, first 3 lines are similar to model
    z1 = linear(inp, w1, b1)
    z2 = relu(z1)
    # the out here, is the same as the earlier model() output
    out = linear(z2, w2, b2)
    diff = out[:, 0] - targ
    loss = diff.pow(2).mean()

    # backward pass
    out.g = 2. * diff[:, None] / inp.shape[0]
    linear_grad(z2, out, w2, b2)
    z1.g = (z1 > 0).float() * z2.g
    linear_grad(inp, z1, w1, b1)
```


### Using pythons debugger

Python comes with its own interactive debugger, `pdb`

```python
import pdb
```

Check out the documentation here [https://docs.python.org/3/library/pdb.html](https://docs.python.org/3/library/pdb.html)

```sh
# if we add the debuffer line here.
def linear_grad(inp, out, w, b):
    """Note that all the operations in this function
    are in-place, returns nothing"""
    
    # the input gradient(rate of change) is related to the
    # output's gradient (rate of change) by the weights, because 
    # of the linear layer
    inp.g = out.g @ w.t()
    pdb.set_trace()  # START DEBUG
    # then the weight's gradient is updated 
    w.g = (inp.unsqueeze(-1) * out.g.unsqueeze(1)).sum(dim=0)
    
    # derivate of the bias, is the derivate of the outputs
    b.g = out.g.sum(dim=0)
```

Executing the `forward_and_backward` will call the `linear_grad` twice:

```
forward_and_backward(x_train, y_train)
> /tmp/ipykernel_58/1909617223.py(12)linear_grad()
     10     import pdb; pdb.set_trace()
     11     # then the weight's gradient is updated
---> 12     w.g = (inp.unsqueeze(-1) * out.g.unsqueeze(1)).sum(dim=0)
     13 
     14     # derivate of the bias, is the derivate of the outputs

ipdb> 
```

Take a look at the help

```
ipdb>  h

Documented commands (type help <topic>):
========================================
EOF    commands   enable    ll        pp       s                until 
a      condition  exit      longlist  psource  skip_hidden      up    
alias  cont       h         n         q        skip_predicates  w     
args   context    help      next      quit     source           whatis
b      continue   ignore    p         r        step             where 
break  d          interact  pdef      restart  tbreak         
bt     debug      j         pdoc      return   u              
c      disable    jump      pfile     retval   unalias        
cl     display    l         pinfo     run      undisplay      
clear  down       list      pinfo2    rv       unt            

Miscellaneous help topics:
==========================
exec  pdb

```

Using the debugger console, the variables at that time step can be checked

```
ipdb>  p out.shape, out.g.shape
(torch.Size([50000, 1]), torch.Size([784, 1]))

ipdb > inp.unsqueeze(-1).shape
torch.Size([50000, 50, 1])

ipdb >   out.g.unsqueeze(1).shape
torch.Size([784, 1, 1])

ipdb > (inp.unsqueeze(-1) * out.g.unsqueeze(1)).shape
torch.Size([50000, 50, 1])

ipdb>  (inp.unsqueeze(-1) * out.g.unsqueeze(1)).sum(dim=0).shape
torch.Size([50, 1])
```

Other helpful `pdb` commands; a helpful reference: [https://realpython.com/python-debugging-pdb/](https://realpython.com/python-debugging-pdb/)

Continues on till next call

```
ipdb> w 
```

Excutes the next call

```
ipdb> n 
```
