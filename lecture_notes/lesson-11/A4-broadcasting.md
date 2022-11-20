
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

## About 1D Vectors + expanding dimensions

There is a distinction between the following:
- `tensor size(3)`
- `tensor size(1, 3)`
- `tensor size(3, 1)`
- This distinction will be essential to master : matrix to matrix operations

```python
vec = torch.tensor([10., 20., 30.])

vec.shape
"""
torch.Size([3])
"""
```

How to convert a vector into a single row matrix

```python
vec_adj = vec.unsqueeze(0)
vec_adj.shape
"""
torch.Size([1, 3])
"""

vec_adj = vec[None, :]
vec_adj.shape
"""
torch.Size([1, 3])
"""

vec_adj
"""
tensor([[10., 20., 30.]])
"""
```

How to convert a vector into a single column matrix

```python
vec_adj = vec.unsqueeze(1)
vec_adj.shape
"""
torch.Size([3, 1])
"""

vec_adj = vec[:, None]
vec_adj.shape
"""
torch.Size([3, 1])
"""

vec_adj
"""
tensor([[10.],
        [20.],
        [30.]])
"""
```


### Note: trailing dimensions

`torch` has a specific syntax to skip dimensions with `...`

A quick example:

```python
some_tensor = torch.randn(2, 3, 4)
some_tensor.shape
# torch.Size([2, 3, 4])

some_tensor[..., None].shape
# torch.Size([2, 3, 4, 1])

some_tensor[None, ...].shape
# torch.Size([1, 2, 3, 4])
```


### How to control the broadcasting dimension

since we combining a `size(3)` and a `size(3, 3)`, there's a question of if its doing it row-wise or column wise. This can be controlled by adjusting the vector ahead of time.

```python
vec = torch.tensor([10., 20., 30.])
mtx = torch.tensor(range(1, 10))
mtx = mtx.reshape(3, 3)
```

Lets look at the default:

```python
vec + mtx
"""
tensor([[11., 22., 33.],
        [14., 25., 36.],
        [17., 28., 39.]])
"""
```

Now lets expand the dimensions of the vector

```python
vec[None, ...] + mtx
# (1, 3)
"""
tensor([[11., 22., 33.],
        [14., 25., 36.],
        [17., 28., 39.]])
"""

vec[..., None] + mtx
# (3, 1)
"""
tensor([[11., 12., 13.],
        [24., 25., 26.],
        [37., 38., 39.]])
"""
```

- Expanding the first dimension, applies the vector row wise
- Expanding the last dimension applies the vector column wise


### Other Broadcasing Examples

```python
vec[None, :]
# torch.Size([1, 3])

vec[:, None]
# 
```