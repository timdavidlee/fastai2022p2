# Matrices + Tensors

From before:

```python
from pathlib import Path  # helps navigate files 
import itertools          # usefull tools for working with collections + iterators
import urllib             # for calling websites, or downloading files
import pickle, gzip       # for opening + saving files, different format
import math, time
import os, shutil         # doing file system things, copy, move, mkdir
# plotting libraries
import matplotlib as mpl, matplotlib.pyplot as plt
```

And loading the same handwritten dataset from before

```python
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
```

Using the first image as an example, we create a list of lists. For ease of reading, the code has been expanded to show what is being called by what

```python
first_image_as_list = list(x_train[0])
image_iterator = iter(first_image_as_list)
img_as_list_of_lists = list(
    iter(
        lambda:
            list(
                itertools.islice(image_iterator, 28)
            ),
        []
    )
)
```

If we look at the first row, we can select a particular element

```python
# row
img_as_list_of_lists[20]

"""
>>> [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09375, 0.4453125, 0.86328125, 0.98828125, 0.98828125, 0.98828125, 0.98828125, 0.78515625, 0.3046875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
"""

# element
img_as_list_of_lists[20][15]

"""
>>> 0.98828125
"""
```

But the preferred "usage" or syntax is:
- `arr[20, 15]` preferred
- `arr[20][15]` curreent

## Better ways of doing matrix

Let's make a `class`

```python
class Matrix:
    def __init__(self, xs):
        self.xs = xs

    def __getitem__(self, idxs):
        # note this assumes that at least 2 indices are passed
        return self.xs[idxs[0]][idxs[1]]
```

- `class` indicates we are defining / declaring a new class, by the name `Matrix`. Python documentation on classes [here](https://docs.python.org/3/tutorial/classes.html#) 

- `__<something>__` has two handlebars on both sides called `double-under` or `dunder`
- here's a list of some common python `dunders`: https://www.pythonmorsels.com/dunder-variables/
- `__init__` is the `constructor`, which means it is run everytime a new `instance` of the class is created
- `self` will be covered in a later section

```python
mtx = Matrix(img_as_list_of_lists)  # this is calling the __init__
mtx[20, 15]   # calls __getitem__([20, 15])
# >>> 0.98828125
```

## pytorch -> tensors

Lets see how this is done in pytorch

```python
import torch
from torch import tensor

one_d = tensor([1, 2, 3])  # 1D

two_d = tensor([
    [1, 2, 3],
    [4, 5, 6],
])  # 2D matrix

one_d, two_d
"""
(tensor([1, 2, 3]),
 tensor([[1, 2, 3],
         [4, 5, 6]]))
"""

img_as_tensor = tensor(img_as_list_of_lists)
```

Note that all this work above was done for a single image array.