```python
import torch
```

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