### Other Broadcasing Examples

```python
import torch
from torch import tensor

vec = torch.tensor([10., 20., 30.])

vec[None, :]
"""
tensor([[10., 20., 30.]])
torch.Size([1, 3])
"""

vec[:, None]
"""
tensor([[10.],
        [20.],
        [30.]])
torch.Size([3, 1])
""" 
```

What happens when a column is multiplied against the row?

```python
vec[:, None] * vec[None, :]
"""
tensor([[100., 200., 300.],
        [200., 400., 600.],
        [300., 600., 900.]])

torch.Size([3, 1]) x torch.Size([1, 3]) => torch.Size([3, 3])
"""
```

What if the order was reversed?

```python
vec[None, :] * vec[:, None]
"""
tensor([[100., 200., 300.],
        [200., 400., 600.],
        [300., 600., 900.]])

torch.Size([3, 1]) x torch.Size([1, 3]) => torch.Size([3, 3])
"""
```

From the notebook:

    When operating on two arrays/tensors, Numpy/PyTorch compares their shapes element-wise. It starts with the **trailing dimensions**, and works its way forward. Two dimensions are **compatible** when

    - they are equal, or
    - one of them is 1, in which case that dimension is broadcasted to make it the same size

    Arrays do not need to have the same number of dimensions. For example, if you have a `256*256*3` array of RGB values, and you want to scale each color in the image by a different value, you can multiply the image by a one-dimensional array with 3 values. Lining up the sizes of the trailing axes of these arrays according to the broadcast rules, shows that they are compatible:

        Image  (3d array): 256 x 256 x 3
        Scale  (1d array):             3
        Result (3d array): 256 x 256 x 3

    The [numpy documentation](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html#general-broadcasting-rules) includes several examples of what dimensions can and can not be broadcast together.