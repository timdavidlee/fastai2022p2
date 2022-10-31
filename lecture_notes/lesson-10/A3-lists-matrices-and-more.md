## After the Break

[https://github.com/fastai/course22p2/blob/master/nbs/01_matmul.ipynb](https://github.com/fastai/course22p2/blob/master/nbs/01_matmul.ipynb)

**Goal: get to stable diffusion from the foundations**

Will start with the following:

- base python
- matplotlib
- python standard librarys (math, os, string, pickle ...)
- jupyter notebooks
- **CAVEAT**: once an external library or function is created from scratch, its permitted to use the imported version in its place

**But I don't have a million dollars to spend on a gpu farm!**

This course will focus on a smaller version of the technology, and after that point, the larger public models will be used. Will aim to create our own:

- `CLIPEncoder`
- `VAE (Autoencoder)`

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

## Get Data - Handwritten images

For demo purposes the popular `MNIST` dataset will be used, which are black-and-white low-resolution images of hand-written digits

[https://github.com/datapythonista/mnist](https://github.com/datapythonista/mnist)

There's also another small black + white dataset called `fashion-mnist`

[https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)

*Other alternatives can be `wget.download` or `requests`

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

Quick note on `with`, which opens a "context". This is a shorted way of writing the longer version

```python
file_stream =  gzip.open(path_gz, 'rb')
pickle.load(file_open)
file_stream.close()

# the same thing
with gzip.open(path_gz, 'rb') as file_stream:
    # as long as the indentation exists, the file_stream is opern
    pickle.load(file_stream)


pickle.load(file_stream) # ERROR, since this is outside the indentation, the filestream has been closed
```

Lets look at the first image, + peek at some of the values, which are decimals ranging from 0 -> 1

```python
first_image_as_list = list(x_train[0])
first_image_as_list[200: 210]
"""
[0.0,
 0.0,
 0.0,
 0.19140625,
 0.9296875,
 0.98828125,
 0.98828125,
 0.98828125,
 0.98828125,
 0.98828125]
"""

len(first_image_as_list)
# >>> 784 because its 28 x 28
```

To help convert 1 long list into a list of lists.
Defining a chunking function, generator, will "yield" only a few values at a time, and can handle un-even lengths e.g. I have 11 items, but want to see them in batches of 5

About `yield` and `generators`: [https://realpython.com/introduction-to-python-generators/](https://realpython.com/introduction-to-python-generators/)

```python
def chunks(x, sz):
    for i in range(0, len(x), sz):
        yield x[i: i + sz]

sample_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
list(chunks(sample_vals, 5))

"""
[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11]]
"""
```

So applying this chunking function to the `784` vectors:

```python
# there's no color channels yet
mpl.rcParams['image.cmap'] = 'gray'

# print the image
plt.imshow(
    chunks(first_image_as_list, 28)
)
```

--image--

### Another way if implementing iterator

There's another helpful standard lib + function called `islice` which is helpful

```python
sample_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
list(itertools.islice(sample_vals, 5))
# [1, 2, 3, 4, 5]

list(itertools.islice(sample_vals, 5))
# [6, 7, 8, 9, 10]
```

So what's happening with the image data, is that it will call the full list of `768` items, and will return a row of `28` each time its called. These will stack for 28 rows which will result in `28 x 28`

```python
image_iterator = iter(first_image_as_list)
img_as_list_of_lists = list(iter(lambda: list(itertools.islice(image_iterator, 28)), []))
plt.imshow(img_as_list_of_lists);
```
