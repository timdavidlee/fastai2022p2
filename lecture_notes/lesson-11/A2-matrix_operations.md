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
            output[i, j] = dot(A[i, :]), B[:, j]
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

### 1.4 Elementwise operations

```python
a = torch.tensor([10, 6, 4])
b = torch.tensor([2, 8, 7])
```

Torch has a lot of operations built in

#### Addition

```
a + b
#  tensor([12, 14, 11])
```

#### Comparison

```
a > b
#  tensor([ True, False, False])
```

A note about these `boolean` expressions: they are stored as integers: `1, 0 == True, False`. As a result, can be summed, averaged etc.

```
(a > b).sum()
# tensor(1)

(a > b).float().mean()
# tensor(0.33)
```

#### Matrix Norm:

Make a 3 x 3 Matrix

```python
mtx = torch.tensor(range(1, 10))
mtx = mtx.reshape(3, 3)
"""
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
"""
```

Frobenius norm:

- for each element, square the value, add up all squares then square root the last total

```python
(mtx * mtx).sum().sqrt()
#  tensor(16.88)
```

### 1.5 Navigating a matrix

```python
mtx = torch.tensor(range(1, 10))
mtx = mtx.reshape(3, 3)
"""
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
"""
```

View a row

```python
mtx[1, :]
#  tensor([4, 5, 6])
```

View a column

```python
mtx[:, 1]
tensor([2, 5, 8])
```

Note that if only one dimension is provided:

```python
mtx[1]
# tensor([4, 5, 6])
```

### 1.6 Using tensor operations in Matmul

```python
def matmul_tensor(A, B):
    A_rows, A_cols = A.shape
    B_rows, B_cols = B.shape
    output = torch.zeros(A_rows, B_cols)
    for i in range(A_rows):
        for j in range(B_cols):
            # substitue the tensor-powered array calculation
            output[i, j] = (A[i, :] * B[:, j]).sum()
    return output

# optional adjustment
def matmul_tensor(A, B):
    A_rows, A_cols = A.shape
    B_rows, B_cols = B.shape
    output = torch.zeros(A_rows, B_cols)
    for i in range(A_rows):
        for j in range(B_cols):
            # substitue the tensor-powered array calculation
            output[i, j] = torch.dot(A[i, :], B[:, j])
    return output
```

```python
%%timeit
matmul(A, B)
# 846 ms ± 27.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%%timeit
matmul_numba(A.numpy(), B.numpy())
# 340 µs ± 888 ns per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

%%timeit
matmul_tensor(A, B)
# 731 µs ± 2.88 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

```


## Broadcasting

The term **broadcasting** what if you have arrays with different shapes (rows / columns)?

From the numpy [documentation](https://numpy.org/doc/stable/user/basics.broadcasting.html):

    The term broadcasting describes how NumPy treats arrays with different shapes during arithmetic operations. Subject to certain constraints, the smaller array is “broadcast” across the larger array so that they have compatible shapes. Broadcasting provides a means of vectorizing array operations so that looping occurs in C instead of Python. It does this without making needless copies of data and usually leads to efficient algorithm implementations. 

### Broadcasting with a Scalars

Even though there is only 1 value, it is **broadcasted** across all the elements

```python
mtx = torch.tensor(range(1, 10))
mtx = mtx.reshape(3, 3)
"""
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
"""
```

Addition

```python
mtx + 1
"""
tensor([[ 2,  3,  4],
        [ 5,  6,  7],
        [ 8,  9, 10]])
"""
```

Comparison

```python
mtx > 5
"""
tensor([[False, False, False],
        [False, False,  True],
        [ True,  True,  True]])
"""
```

### Broadcasting Vectors onto a matrix

Say we want to multiply `vec` to every **row** in a matrix

```python
vec = torch.tensor([10., 20., 30.])
vec.shape, mtx.shape
# (torch.Size([3]), torch.Size([3, 3]))
```

and even though the shapes are different, watch as they are added together

```python
mtx + vec
"""
tensor([[11., 22., 33.],
        [14., 25., 36.],
        [17., 28., 39.]])
"""

vec + mtx
"""
tensor([[11., 22., 33.],
        [14., 25., 36.],
        [17., 28., 39.]])
"""
```

Whats going on underneath? a function called `.expand_as()`, and though it LOOKs like its copying the data, it is not.

```python
t = vec.expand_as(mtx)
"""
tensor([[10., 20., 30.],
        [10., 20., 30.],
        [10., 20., 30.]])
"""
```

Digging into the intermediate tensor: its the same values with a stride for "walking"

```python
t.storage() 
"""
 10.0
 20.0
 30.0
[torch.storage._TypedStorage(dtype=torch.float32, device=cpu) of size 3]
"""
t.stride()
#  (0, 1)
```
