# Practice multiplying matrices

**Goal:** practice tensor manipulation, matrix changes + multiplication are the BASICS. All algorithms build on these addition, multiplication + broadcasting ideas

## Clustering + Mean Shift Algorithms

What is cluster analysis? It is unsupervised analysis, "are there groups of similar things"? Cluster analysis can be over or mis-used. General guidance:

- same things (pixels in images)
- same scales (same 1 - 5 range for surveys)

### First create synthetic data with intentional clusters 

For now, will generate 6 centroids with 250 samples each

```python
import math, matplotlib.pyplot as plt, operator, torch
from functools import partial

from torch.distributions.multivariate_normal import MultivariateNormal
from torch import tensor

torch.manual_seed(42)
torch.set_printoptions(precision=3, linewidth=140, sci_mode=False)

n_clusters=6
n_samples =250

# generates 6 points in 2 dimensions (x, y) for plotting
# the 70 will create a spread, and the -35 is the offset
centroids = torch.rand(n_clusters, 2) * 70 - 35
```

```python
def sample(m):
    """
    m = (x, y)

    the covariance matrix looks like:
        tensor([[5., 0.],
                [0., 5.]])
    """
    covar_mtx = torch.diag(tensor([5.,5.]))
    return MultivariateNormal(m, covar_mtx).sample((n_samples,))

# adding groups of points
slices = [sample(c) for c in centroids]
data = torch.cat(slices)
```

Writing a plotting function

```python
def plot_data(centroids, data, n_samples, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    for i, centroid in enumerate(centroids):
        samples = data[i*n_samples:(i+1)*n_samples]
        ax.scatter(samples[:,0], samples[:,1], s=1)
        ax.plot(*centroid, markersize=10, marker="x", color='k', mew=5)
        ax.plot(*centroid, markersize=5, marker="x", color='m', mew=2)
```

```
plot_data(centroids, data, n_samples)
```

Now even though clusters are intentionally there, there are no labels! It's simply a collection of
scatterplot data.

## Mean Shift

`mean-shift` does not require the number of clusters to be known before hand, like `k-means`.

### Algorithm

- Start with a point `x` out of dataset `X` (big X)
- Find the distance between `x` and all other remaining points
- Take a `weighted` average determined by closeness to `X`
    - will use `gaussian kernel`
    - the farther points should have lower contribution, the rate of this decay is the `bandwidth` or the `standard deviation` of the `gaussian`
    - the closer points should have larger contribution
- Update `x` as a weighted average or all other points `X`
    - so all points `x` will be calculated on last step's positions


### Writing it in pytorch

Will push points closer and closer together, similar to gravity

```python
def gaussian(d, bw):
    """also known as the normal distribution"""
    numer = torch.exp(-0.5*((d / bw))**2)
    denom = (bw * math.sqrt(2*math.pi))
    return  numer / denom 
```

This is a high-level plotting function that actually accepts another function

```python
from functools import partial

def plot_func(f):
    """
    f - some function to be plotting from 0, 10, with 100 steps
    """
    x = torch.linspace(0,10,100)
    plt.plot(x, f(x))

# pass the gaussian with a bandwidth of 2.5
plot_func(partial(gaussian, bw=2.5))

# or 
plot_func(lambda x: gaussian(x, bw=2.5))
```

**Note**: Choose a `bandwidth` covers about 1/3rd of the data


```python
# keep the orignal data
# torch.Size(1500, 2)
Big_X = data.clone()

# isolate one point
# torch.Size(2)
little_x = data[0]
```

Sample subtraction

```python
(little_x - big_X)[:8]
>>> tensor([[ 0.000,  0.000],
            [ 0.513, -3.865],
            [-4.227, -2.345],
            [ 0.557, -3.685],
            [-5.033, -3.745],
            [-4.073, -0.638],
            [-3.415, -5.601],
            [-1.920, -5.686]])
```

Calculate the Euclidean distance

```python
euclid_distance = ((little_x - Big_X)**2).sum(1).sqrt()

# pass weights to gaussian function, torch.Size(1500)
weight = gaussian(euclid_distance, 2.5)

# then we want to weight the original, since this is matrix multiplication, the weight needs
# an additional dimension
# [1500, 1] x [1500, 2]
weighted_coords = weight[:, None] * big_X

# then perform average as normal
new_location = weighted_coords.sum(dim=0) / weight.sum()
```

Lets write this in a convenience function

```python
def one_update(X):
    """
    updates all points, note that this updating in place,
    the first point will be updated + will affect future updates
    """
    # iterating through each point
    for i, x in enumerate(X):
        # calculate distance from 1 point vs. all others
        dist = torch.sqrt(((x-X)**2).sum(1))
        
        # calculate gaussian weighting
        weight = gaussian(dist, 2.5)

        # update the point with the new average
        X[i] = (weight[:, None] * X).sum(0) / weight.sum()


def meanshift(data):
    X = data.clone()
    # go through 5 iterations
    for it in range(5):
        one_update(X)
    return X
```

A quick time benchmark

```python
%time X=meanshift(data)
>>> CPU times: user 642 ms, sys: 0 ns, total: 642 ms
```

```
plot_data(centroids+2, X, n_samples)
```

### Write the algo for the GPU

Looking again at the main update function

```python linenums="1"
def one_update(X):
    """
    updates all points, note that this updating in place,
    the first point will be updated + will affect future updates
    """
    # iterating through each point
    for i, x in enumerate(X):
        # calculate distance from 1 point vs. all others
        dist = torch.sqrt(((x-X)**2).sum(1))
        
        # calculate gaussian weighting
        weight = gaussian(dist, 2.5)

        # update the point with the new average
        X[i] = (weight[:, None] * X).sum(0) / weight.sum()
```

The strategy to reduce the overhead of row looping (line 7) will be broadcasting. To avoid memory allocation issues, this will be mini-batched, meaning a few records will be broadcast at a time.

```python
# batch size
bs = 5

# 1500 x 2
X = data.clone()

# now represents a subset of points instead of a single
# 5 x 2
x = X[:bs]
```

Now the distance calculation becomes more difficult:

- [5 x 2]
- [1500 x 2]
- The output needs to be [5 x 1500 x 2]

So whats the strategy?

- [5 x 1 x 2]
- [1 x 1500 x 2]
- The output should be [5 x 1500 x 2] 
- this is happening because the 1's added to the dimensions result in broadcasting

Walking through it slowly with shapes

```python
# torch.Size([5, 1500, 2])
delta = X[None, :] - x[:, None]

# summing on the LAST dimension
# torch.Size([5, 1500])
sq_dist =(delta**2).sum(dim=2)

# squares in place, still: torch.Size([5, 1500])
dist = sq_dist.sqrt()
```

Now wrap in a function

```python
def dist_b(a, b):
    delta = a[None, :] - b[:, None]
    sq_dist = (delta**2).sum(2)
    return sq_dist.sqrt()

# note that its important which slot big X and little x go into
dist_b(X, x)

>>> tensor([[ 0.000,  3.899,  4.834,  ..., 17.628, 22.610, 21.617],
            [ 3.899,  0.000,  4.978,  ..., 21.499, 26.508, 25.500],
            [ 4.834,  4.978,  0.000,  ..., 19.373, 24.757, 23.396],
            [ 3.726,  0.185,  4.969,  ..., 21.335, 26.336, 25.333],
            [ 6.273,  5.547,  1.615,  ..., 20.775, 26.201, 24.785]])
```

Once this is gathered, applied the gaussian norm

```python
weight = gaussian(dist_b(X, x), 2)
```

Again the weight calculation 

- weight is [5 x 1500]
- X is [1500 x 2]

So the same strategy will be employed:

- weight to be [5, 1500, 1]
- X is [1, 1500, 2]


Walking through slowly

```python
# torch.Size([5, 1500, 2])
weighted_coordinates = weight[...,None] * X[None]

# calculate the average in two parts 
# torch.Size([5, 2])
numer = weighted_coordinates.sum(1)

# torch.Size(5, 1)
denom = weight.sum(1, keepdim=True)

updated_points = numer / denom
```

Now writing in a gpu function:

```python linenum="1"
def meanshift(data, bs: int = 500):
    n = len(data)
    X = data.clone()

    # 5 iterations as before
    for it in range(5):

        # going through the dataset, by batch size
        for i in range(0, n, bs):
            # generates start + end indices
            s = slice(i, min(i + bs, n))

            # does the recalculation here
            weight = gaussian(dist_b(X, X[s]), 2.5)
            div = weight.sum(1, keepdim=True)

            # finally the update for the same batch slice
            X[s] = weight @ X / div
    return X
```

Timing

```python
%timeit -n 5 _=meanshift(data, 1250).cpu()
>>> 2.52 ms ± 272 µs per loop (mean ± std. dev. of 7 runs, 5 loops each)
```

## Homework

**Homework:** implement k-means clustering, dbscan, locality sensitive hashing, or some other clustering, fast nearest neighbors, or similar algorithm of your choice, on the GPU. Check if your version is faster than a pure python or CPU version.

"LHS" comes up a lot `== locality sensitive hashing`, for an optimized nearest neighbor

Bonus: Implement it in APL too!

Super bonus: Invent a new meanshift algorithm which picks only the closest points, to avoid quadratic time.

Super super bonus: Publish a paper that describes it :D


