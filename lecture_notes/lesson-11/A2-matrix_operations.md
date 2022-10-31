# Matrix multiplication from foundations

```python
from pathlib import Path  # helps navigate files 
import itertools          # usefull tools for working with collections + iterators
import urllib             # for calling websites, or downloading files
import pickle, gzip       # for opening + saving files, different format
import math, time
import os, shutil         # doing file system things, copy, move, mkdir
# plotting libraries
import matplotlib as mpl, matplotlib.pyplot as plt

import torch
from torch import tensor  # our tensor + deeplearning framework

# set some printing options
torch.set_printoptions(precision=2, linewidth=140, sci_mode=False)


MNIST_URL='https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/mnist.pkl.gz?raw=true'
path_data = Path('data')
path_data.mkdir(exist_ok=True)
path_gz = path_data/'mnist.pkl.gz'

# get the file, only if it hasn't be downloaded
if not path_gz.exists():
    urllib.request.urlretrieve(MNIST_URL, path_gz)

with gzip.open(path_gz, 'rb') as f:
    # the gzip contains 4 arrays + some metadata
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')

    #  x_train.shape, y_train.shape, x_valid.shape, y_valid.shape 
    #  ((50000, 784), (50000,), (10000, 784), (10000,))

# convert all downloaded data into pytorch tensors
x_trn_tensor, y_trn_tensor, x_val_tensor, y_val_tensor = map(tensor, (x_train, y_train, x_valid, y_valid))

# reshape the training input into 28 x 28 images
imgs = x_trn_tensor.reshape((-1, 28, 28))
```

## 1. A preview of setting up linear weight multiplication

<img src="https://i.ibb.co/w4yWq73/Screenshot-2022-10-30-at-6-51-13-PM.jpg" alt="Screenshot-2022-10-30-at-6-51-13-PM" border="2">

```python
torch.manual_seed(1)
weights = torch.randn(784, 10)
bias = torch.zeros(10)
```

since we will be doing a small mini-batch for illustration purposes

```python
A = x_valid[: 5]  # 5 x 784
B = weights       # 784 x 10
```

### 1.1 General rule when multiplying matrices

1. The inner dimension must match
2. the output will collapse the common dimension

```python
A_rows, A_cols = A.shape
B_rows, B_cols = B.shape

# will be our output holder
output = torch.zeros(A_rows, B_cols)
```

Lets write the double loop for this (inefficient i know!)

```python
for i in range(A_rows):
    for j in range(B_cols):
        for k in range(A_cols):   # same as B_rows
            output[i, j] += A[i, k] * B[k, j]

output
"""
tensor([[-10.94,  -0.68,  -7.00,  -4.01,  -2.09,  -3.36,   3.91,  -3.44, -11.47,  -2.12],
        [ 14.54,   6.00,   2.89,  -4.08,   6.59, -14.74,  -9.28,   2.16, -15.28,  -2.68],
        [  2.22,  -3.22,  -4.80,  -6.05,  14.17,  -8.98,  -4.79,  -5.44, -20.68,  13.57],
        [ -6.71,   8.90,  -7.46,  -7.90,   2.70,  -4.73, -11.03, -12.98,  -6.44,   3.64],
        [ -2.44,  -6.40,  -2.40,  -9.04,  11.18,  -5.77,  -8.92,  -3.79,  -8.98,   5.28]])
"""
```

### 1.2 Wrapping all into a function:

```python
def matmul(A, B):
    A_rows, A_cols = A.shape
    B_rows, B_cols = B.shape
    output = torch.zeros(A_rows, B_cols)
    for i in range(A_rows):
        for j in range(B_cols):
            for k in range(A_cols):   # same as B_rows
                output[i, j] += A[i, k] * B[k, j]
    return output
```

Do a quick timing test

```
%%timeit
matmul(A, B)
#  741 ms ± 1.64 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

Egad that is slow, lets try and speed it up

### 1.3 Numba

`numba` is a system that takes python and turns it into machine code

```python
from numba import njit
from numpy import array
```

An example

```python
@njit
def dot(a, b):
    res = 0.
    for i in range(len(a)):
        res += a[i] * b[i]
    return res
```

Time it twice, the first time will compile it, the 2nd time will let it run

```python
%%timeit
dot(
    array([1., 2., 3.]),
    array([4., 5., 6.])
)
# 5.11 µs ± 7.3 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

For the second test, runs much faster!

```python
%%timeit
dot(
    array([1., 2., 3.]),
    array([4., 5., 6.])
)
#  1.85 µs ± 8.05 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
```

Now if we update the function:

```python
def matmul_numba(A, B):
    A_rows, A_cols = A.shape
    B_rows, B_cols = B.shape
    output = torch.zeros(A_rows, B_cols)
    for i in range(A_rows):
        for j in range(B_cols):
            # substitue the numba-powered array calculation
            output[i, j] = dot(A[i, :], B[:, j])
    return output
```

and now time the process

```python
%%timeit
matmul(A, B)
# 846 ms ± 27.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%%timeit
matmul_numba(A.numpy(), B.numpy())
# 340 µs ± 888 ns per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
```

Note the units! its a 2000x speed up