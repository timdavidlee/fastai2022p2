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

Note that all this work above was done for a single image array. Lets use python's `map` function to apply it to many things at once. Consider a quick example below:

```python
a, b, c, d = map(lambda x: x + 1, [1, 2, 3, 4])
"""
a, b, c, d
(2, 3, 4, 5)
a
2
"""
```

- The map will go through and apply our function `x + 1` against each item in the list.
- Also notice, since we are returning 4 values, if we put exactly 4 variables on the left side, we can do individual assignments

Now lets use this method on the training + validation datasets

```python
x_trn_tensor, y_trn_tensor, x_val_tensor, y_val_tensor = map(tensor, (x_train, y_train, x_valid, y_valid))
x_trn_tensor.shape
# >>> torch.Size([50000, 784])
```

**properties of torch.tensor**

- `.shape`: tells the size of the different dimensions
- `.type()`: tells the type, `float`, `long`, `int` and will also tell the precision `16, 32, 64`
- `.reshape(shape=new_size)`: can re-arrange the elements into a different shape

```python
t1 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
t2 = t1.reshape(shape=(2, 6))
t3 = t1.reshape(shape=(3, 4))
t4 = t1.reshape(shape=(12, 1))

t1.shape, t2.shape, t3.shape, t4.shape
"""
(torch.Size([12]), torch.Size([2, 6]), torch.Size([3, 4]), torch.Size([12, 1]))
"""

t2
"""
tensor([[ 1,  2,  3,  4,  5,  6],
        [ 7,  8,  9, 10, 11, 12]])
"""
```

**Caveat**: pay close attention to 1-Dish tensors: `shape=(12,)` is not the same as `shape=(12, 1)` or `shape=(1, 12)`

```python
# note the original was [50000, 784]
imgs = x_trn_tensor.reshape((-1, 28, 28))
imgs.shape
"""
torch.Size([50000, 28, 28])
"""

plt.imshow(imgs[0])
```

Note that the `-1` means "keep the first dimension"

--image--

## A quick word on vocab

`APL` is a programming language developed closer to the expressions found in math. (https://tryapl.org/)[https://tryapl.org/]. In APL, they don't use the word `tensor` they use the word `arrays`. `numpy` which was heavily influenced by `APL` also borrowed the language and called these `arrays`. Pytorch which was heavily influenced by numpy, for some reason calls them `tensors`.

- `1-D tensor`: is like a vector, or list
- `2-D tensor`: is like a matrix, or spreadsheet
- `3-D tensor`: is like a cube, a batch of matrices, or a stack of spreadsheets
- and can be much higher order dimensions!

`Fast.ai` has a [APL study Forum](https://forums.fast.ai/t/apl-array-programming/97188)

## Language of Tensors 

### Rank - how many dimensions are there?

Here's an example of a rank-1 tensor

```python
z = torch.tensor([1, 2, 3, 4])
z.shape
# torch.Size([4])
```

Note the extra nested list

```python
z = torch.tensor([[1, 2, 3, 4],])
z.shape
# torch.Size([1, 4])
```

Now considering all our images, will be rank-3

```python
imgs.shape
# torch.Size([50000, 28, 28])

```

A single image would be a rank-2 matrix

```python
imgs[0].shape
# torch.Size([28, 28])
```

## Extract information about the dataset from the tensors

```python
n_records, n_pixels = x_trn_tensor.shape

# how many targets
min(y_trn_tensor)   # 9
max(y_trn_tensor)   # 0
```

So there are a total of 10 classes, and they will be labels