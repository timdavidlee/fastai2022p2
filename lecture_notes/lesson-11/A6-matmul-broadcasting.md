# Matmul with broadcasting

From before, here's all the functions we have collected

```python
import torch
from numba import njit
from torch import tensor

def matmul(A, B):
    A_rows, A_cols = A.shape
    B_rows, B_cols = B.shape
    output = torch.zeros(A_rows, B_cols)
    for i in range(A_rows):
        for j in range(B_cols):
            for k in range(A_cols):   # same as B_rows
                output[i, j] += A[i, k] * B[k, j]
    return output


@njit
def dot(a, b):
    res = 0.
    for i in range(len(a)):
        res += a[i] * b[i]
    return res


def matmul_numba(A, B):
    """Note, should convert to numpy before giving to this function"""
    A_rows, A_cols = A.shape
    B_rows, B_cols = B.shape
    output = torch.zeros(A_rows, B_cols)
    for i in range(A_rows):
        for j in range(B_cols):
            # substitue the numba-powered array calculation
            output[i, j] = dot(A[i, :]), B[:, j]
    return output


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

For fun, this author decided to make a pure numba function: one modification, the output array needs to be allocated outside of the loop

```python
@njit
def matmul_pure_numba(A, B, output):
    A_rows, A_cols = A.shape
    B_rows, B_cols = B.shape
    for i in range(A_rows):
        for j in range(B_cols):
            for k in range(A_cols):   # same as B_rows
                output[i, j] += A[i, k] * B[k, j]
    return output
```


Lets re-write the function again with broadcasting

```python
def matmul_broadcasting(A, B):
    A_rows, A_cols = A.shape
    B_rows, B_cols = B.shape
    output = torch.zeros(A_rows, B_cols)
    for i in range(A_rows):
        output[i] = (A[i, :, None] * B).sum(dim=0)
    return output
```

#### Discussion

```python
output[i] = (A[i, :, None] * B).sum(dim=0)

A
# (5 x 784) minibatch for an image

A[i, :, None]
# take the first image, and expand the last dimension
# (784 x 1)

B
# (784 x 10)

(A[i, :, None] * B)
"""
Because of the dimension expansion, A is broadcast against B, 
and the shape is retained
(784 x 10)
"""

(A[i, :, None] * B).sum(dim=0)
"""
Finally this is summed "across rows (dim=0)", resulting in 10 numbers
"""
```



Lets do some timing comparisons

```python
from timeit import timeit

fake_img = torch.randn(5, 784)  # 5 was an arbitrary batch size
weights = torch.randn(784, 10)  # 10 was the number of classes we are trying to predict
```

```python
%%timeit
matmul(fake_img, weights)
#  684 ms ± 2.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%%timeit
matmul_numba(fake_img.numpy(), weights.numpy())
#  358 µs ± 37 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)

%%timeit
matmul_tensor(fake_img, weights)
# 693 µs ± 944 ns per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

%%timeit
matmul_broadcasting(fake_img, weights)
#  104 µs ± 260 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

%%timeit
output = np.zeros(shape=[fake_img.shape[0], weights.shape[-1]])
matmul_pure_numba(fake_img.numpy(), weights.numpy(), output)
#  48.4 µs ± 4.38 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
```