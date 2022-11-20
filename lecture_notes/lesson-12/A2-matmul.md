# Finishing up Matmul

## Recap

The goal of this walkthrough is to remove as many loops as possible:

## Einstein Summation

Einsten summation  or `torch.einsum` is a compact representation for combining products and sums in a general way.

So consider the following, we have two matrices: `m1` and `m2`

- `m1` is `5 x 784` which represents the 5 images of `28 x 28`
- `m2` is `784 x 10` which represents the weights, each one of those pixels relates to the 10 classes `(0-9) digits`

```python
import torch

mr = torch.einsum('ik,kj->ikj', m1, m2)
mr.shape

>>> torch.Size([5, 784, 10])
```

This can also be thought of as this:

```
m1, row1 x m2, col1 -> 784 elements

m1, row2 x m2, col1 -> 784 elements

... and so on

Resulting in a stack of 5 x 784 x 10
```

Consider the following:

- the first row of `m1` times the first column of `m2`

```python
first_by_first = (m1[0] * m2[:, 0])
first_by_first.sum()

>>> tensor(-10.94)
```


As proof, the first sample in the result `mr` looks like the following:

```python
mr(1)
tensor([[-10.94,  -0.68,  -7.00,  -4.01,  -2.09,  -3.36,   3.91,  -3.44, -11.47,  -2.12],
        [ 14.54,   6.00,   2.89,  -4.08,   6.59, -14.74,  -9.28,   2.16, -15.28,  -2.68],
        [  2.22,  -3.22,  -4.80,  -6.05,  14.17,  -8.98,  -4.79,  -5.44, -20.68,  13.57],
        [ -6.71,   8.90,  -7.46,  -7.90,   2.70,  -4.73, -11.03, -12.98,  -6.44,   3.64],
        [ -2.44,  -6.40,  -2.40,  -9.04,  11.18,  -5.77,  -8.92,  -3.79,  -8.98,   5.28]])
```

And that upper left hand value matches or row x column multiplication

So to recap:

```python
torch.einsum('ik,kj->ikj', m1, m2)
"""
Will leave the elements in the middle (k)
"""

torch.einsum('ik,kj->ij', m1, m2)
"""
Since the middle indice is not available, will sum in that dimension
"""
```

Using this we can re-write `matmul` using `einsum`

```python
def matmul(a, b):
    return torch.einsum('ik,kj->ij', a, b)
```

What about speed?

```python
test_close(tr, matmul(x_train, weights), eps=1e-3)

%timeit -n 5 _=matmul(x_train, weights)
>>> 15.1 ms ± 176 µs per loop (mean ± std. dev. of 7 runs, 5 loops each)
```

## Pytorch Matmul Speed

```python
test_close(tr, x_train @ weights, eps=1e-3)
%timeit -n 5 _=torch.matmul(x_train, weights)
>>> 15.2 ms ± 96.2 µs per loop (mean ± std. dev. of 7 runs, 5 loops each)
```

## What about GPU + CUDA Matmul?

GPUs generally have lower clock speeds but are great at parallel calculations. To help illustrate this, consider the following function that only fills in one part of the grid (similar to `first_by_first` idea above)

```python
# the previous result of matmul
tensor([[-10.94,  -0.68,  -7.00,  -4.01,  -2.09,  -3.36,   3.91,  -3.44, -11.47,  -2.12],
        [ 14.54,   6.00,   2.89,  -4.08,   6.59, -14.74,  -9.28,   2.16, -15.28,  -2.68],
        [  2.22,  -3.22,  -4.80,  -6.05,  14.17,  -8.98,  -4.79,  -5.44, -20.68,  13.57],
        [ -6.71,   8.90,  -7.46,  -7.90,   2.70,  -4.73, -11.03, -12.98,  -6.44,   3.64],
        [ -2.44,  -6.40,  -2.40,  -9.04,  11.18,  -5.77,  -8.92,  -3.79,  -8.98,   5.28]])
```

Below is the function design just to calculate 1 value in the above array

```python
def matmul(grid, a, b, c):
    """
    grid: a tuple of  (x,y) coordinates
    a: tensor to multiply
    b: tensor to multiply
    c: a empty tensor to hold results
    """
    i, j = grid
    if i < c.shape[0] and j < c.shape[1]:
        tmp = 0.
        for k in range(a.shape[1]):
            tmp += a[i, k] * b[k, j]
        c[i, j] = tmp
```

Below is the pattern to use the "coordinate calculation function"

```python
# results holder
res = torch.zeros(ar, bc)

# 0,0 will be the upper left value to calculat
matmul((0, 0), m1, m2, res)
```

Which results in 

```python
tensor([[-10.94,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00],
        [  0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00],
        [  0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00],
        [  0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00],
        [  0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00]])
```

So because this is localized (only writes to one section of the result), this can considered a`kernel`. The below is a general framework to loop through + apply a `kernel` or our function.

```python
def launch_kernel(kernel, grid_x, grid_y, *args, **kwargs):
    for i in range(grid_x):
        for j in range(grid_y):
            kernel((i,j), *args, **kwargs)
```

```python
res = torch.zeros(ar, bc)
launch_kernel(matmul, ar, bc, m1, m2, res)
res
```

```python
tensor([[-10.94,  -0.68,  -7.00,  -4.01,  -2.09,  -3.36,   3.91,  -3.44, -11.47,  -2.12],
        [ 14.54,   6.00,   2.89,  -4.08,   6.59, -14.74,  -9.28,   2.16, -15.28,  -2.68],
        [  2.22,  -3.22,  -4.80,  -6.05,  14.17,  -8.98,  -4.79,  -5.44, -20.68,  13.57],
        [ -6.71,   8.90,  -7.46,  -7.90,   2.70,  -4.73, -11.03, -12.98,  -6.44,   3.64],
        [ -2.44,  -6.40,  -2.40,  -9.04,  11.18,  -5.77,  -8.92,  -3.79,  -8.98,   5.28]])
```

But it is not fast, because the above is base python. How can this be written in `cuda` or gpu-language?

`numba` has tools to actual generate `cuda` code, but it comes with some alterations.

```python
from numba import cuda

@cuda.jit  # compiles to GPU code
def matmul(a,b,c):
    i, j = cuda.grid(2)
    if i < c.shape[0] and j < c.shape[1]:
        tmp = 0.
        for k in range(a.shape[1]): tmp += a[i, k] * b[k, j]
        c[i,j] = tmp
```

So lets try it out! 

The important step will be copying the data over to the GPU (by default it is sitting on the CPU)

```python
# will be our results holder on the GPU
r = np.zeros(tr.shape)

m1g = cuda.to_device(x_train)
m2g = cuda.to_device(weights)
rg = cuda.to_device(r)

# or 
m1g, m2g, rg = map(cuda.to_device, (x_train, weights, r))
```
In `cuda`, there is a concept called `TPB` (or threads per block)

```python
# turns grid into blocks
TPB = 16
rr, rc = r.shape

# (3125, 1)
blockspergrid = (math.ceil(rr / TPB), math.ceil(rc / TPB))

# this is our Cuda version of the function, along with additional params in []
matmul[blockspergrid, (TPB, TPB)](m1g,m2g,rg)

# copy the results back to the CPU
r = rg.copy_to_host()

test_close(tr, r, eps=1e-3)
```

Doing a quick speed benchmark

```python
%%timeit -n 10
matmul[blockspergrid, (TPB, TPB)](m1g, m2g, rg)
r = rg.copy_to_host()

>>> 3.61 ms ± 708 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

Which is already much faster, lets try out the native torch methods for using the GPU

```python
m1c = x_train.cuda()
m2c = weights.cuda()
```

```python
# do the multiplication then copy back to cpu
r = (m1c @ m2c).cpu()
```

```python
%timeit -n 10 r=(m1c@m2c).cpu()
>> 458 µs ± 93.1 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

`0.5ms` compared to the previous `500ms` is a HUGE improvement.







 
