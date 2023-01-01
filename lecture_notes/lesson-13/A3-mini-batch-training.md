# Minibatch training


Loading the following into memory

```python
import pickle, gzip, math, os, time, shutil,torch
from pathlib import Path

import matplotlib as mpl,
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import tensor, nn

from fastcore.test import test_close

torch.set_printoptions(precision=2, linewidth=140, sci_mode=False)
torch.manual_seed(1)
mpl.rcParams['image.cmap'] = 'gray'


# loading MNIST again
path_data = Path('data')
path_gz = path_data/'mnist.pkl.gz'
with gzip.open(path_gz, 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    x_train, y_train, x_valid, y_valid = map(tensor, [x_train, y_train, x_valid, y_valid])
```

### Setting up data + Model

```python
class Model(nn.Module):
    def __init__(self, n_in, nh, n_out):
        super().__init__()
        self.layers = [nn.Linear(n_in,nh), nn.ReLU(), nn.Linear(nh,n_out)]
        
    def __call__(self, x):
        for l in self.layers: x = l(x)
        return x

n, m = x_train.shape
c = y_train.max() + 1
nh = 50

model = Model(m, nh, 10)
pred = model(x_train)
pred.shape
```


## improving cross-entropy loss

Reminder, our goal is to output predictions as follows:

```
pred = [0.99, 0.0, ....]  # 10 values
truth = [1, 0, 0, ....]  # 10 values
```

### A quick example of cross-entropy loss

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "labels": ["cat", "dog", "plane", "fish", "building"],
    "output": [-4.89, 2.60, 0.59, -2.07, -4.57],
})
df["exp"] = np.exp(df["output"])
df["softmax"] = df["exp"] / df["exp"].sum()
print(df.to_markdown())
```

Below is a sample example table (shown in excel in class) of cross entropy loss being calculated

```
|    | labels   |   output |         exp |    softmax |
|---:|:---------|---------:|------------:|-----------:|
|  0 | cat      |    -4.89 |  0.00752142 | 0.00048803 |
|  1 | dog      |     2.6  | 13.4637     | 0.8736     |
|  2 | plane    |     0.59 |  1.80399    | 0.117052   |
|  3 | fish     |    -2.07 |  0.126186   | 0.00818761 |
|  4 | building |    -4.57 |  0.010358   | 0.00067208 |
```
 
 First, we will need to compute the softmax of our activations. This is defined by:

$$\hbox{softmax(x)}_{i} = \frac{e^{x_{i}}}{e^{x_{0}} + e^{x_{1}} + \cdots + e^{x_{n-1}}}$$

or more concisely:

$$\hbox{softmax(x)}_{i} = \frac{e^{x_{i}}}{\sum\limits_{0 \leq j \lt n} e^{x_{j}}}$$ 

In practice, we will need the log of the softmax when we calculate the loss.

Note: in general log will work better because of extremely small decimal sizes

```python
def log_softmax(x):
    numer = x.exp()
    denom = (x.exp().sum(-1,keepdim=True))
    return (numer / denom).log()
```

Another alternate way of writing this formula is the following:

Note that the formula 

$$\log \left ( \frac{a}{b} \right ) = \log(a) - \log(b)$$ 

gives a simplification when we compute the log softmax:

```python
def log_softmax(x):
    # x.exp().log() - x.exp().sum(-1,keepdim=True).log()
    # the first term reduces to `x`
    return x - x.exp().sum(-1,keepdim=True).log()
```

The one issue with the version above, is the `x.exp()` can lead to very large numbers and the loss of precision, so a 3rd trick will be used:

Note that there's another way of writing this:

Then, there is a way to compute the log of the sum of exponentials in a more stable way, called the [LogSumExp trick](https://en.wikipedia.org/wiki/LogSumExp). The idea is to use the following formula:

$$\log \left ( \sum_{j=1}^{n} e^{x_{j}} \right ) = \log \left ( e^{a} \sum_{j=1}^{n} e^{x_{j}-a} \right ) = a + \log \left ( \sum_{j=1}^{n} e^{x_{j}-a} \right )$$

where `a` is the maximum of the $x_{j}$.

**Explanation of the log sum exp trick**:

```python
x.exp() => (x - a).exp() * a.exp()
```

And applying the aggregates + reducing

```python
log(sum(x.exp())) => log(sum((x - a).exp()) * a.exp())
log(sum(x.exp())) => a + log(sum((x - a).exp())
```

This is adding complexity, but at the benefit of helping accuracy and precision. The implemented version can be found below

```python
# equivalent function with the trick applied
def logsumexp(x):
    m = x.max(-1)[0]
    return m + (x-m[:,None]).exp().sum(-1).log()
```

And now the pytorch version

```python
#pytorch equivalent
def log_softmax(x):
    return x - x.logsumexp(-1,keepdim=True)
```

Testing the above functions to ensure the output is close

```python
test_close(logsumexp(pred), pred.logsumexp(-1))
sm_pred = log_softmax(pred)
sm_pred
```

### Calculating cross entropy

The cross entropy loss for some target $x$ and some prediction $p(x)$ is given by:

$$ -\sum x\, \log p(x) $$

But since our $x$s are 1-hot encoded (actually, they're just the integer indices), this can be rewritten as $-\log(p_{i})$ where i is the index of the desired target.

This can be done using numpy-style [integer array indexing](https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#integer-array-indexing). Note that PyTorch supports all the tricks in the advanced indexing methods discussed in that link.

#### What needs to happen:

```python
preds = [
   [0.8, 0.15, 0.05],
   [0.07, 0.23, 0.7],
]

labels = [1, 2]

# these are the retrieval positions in the probability array
onehotlabels = [
    [0, 1, 0],
    [0, 0, 1],
]

# essentially the log probs for the indicies related to the target labels
cross_entropy_loss = log(0.15) + log(0.7)
```

How to access a different column per row?

Given:

```python
y_train[:3]
>>> tensor([5, 0, 4])
```

and the desired target values are the following:

```python
sm_pred[0,5], sm_pred[1,0], sm_pred[2,4]
>>> (tensor(-2.20, grad_fn=<SelectBackward0>),
    tensor(-2.37, grad_fn=<SelectBackward0>),
    tensor(-2.36, grad_fn=<SelectBackward0>)
```

Can pass 2 different arrays, and retrive the same values

```python
row_indices = [0,1,2]
col_indices = y_train[:3]
sm_pred[row_indices, col_indices]
>>> tensor([-2.20, -2.37, -2.36], grad_fn=<IndexBackward0>)
```

Now, ignoring the toy case, 

```python
# nll = negative log likelihood
def nll(input, target):
    return -input[range(target.shape[0]), target].mean()
```

And now can calculate loss:

```python
loss = nll(sm_pred, y_train)
loss
>>> tensor(2.30, grad_fn=<NegBackward0>)
```

Will compare our NLL vs. the pytorch NLL:

```python
test_close(F.nll_loss(F.log_softmax(pred, -1), y_train), loss, 1e-3)
```

The pytorch implementation

```python
test_close(F.cross_entropy(pred, y_train), loss, 1e-3)
```


## Basic Training Loop

```python
loss_func = F.cross_entropy

bs=50                  # batch size
xb = x_train[0:bs]     # a mini-batch from x
preds = model(xb)      # predictions
```

Take a peek at a single batch of training data:

```
yb = y_train[0:bs]
yb
>>> tensor([3, 9, 3, 8, 5, 9, 3, 9, 3, 9, 5, 3, 9, 9, 3, 9, 9, 5, 8, 7, 9, 5, 3, 8, 9, 5, 9, 5, 5, 9, 3, 5, 9, 7, 5, 7, 9, 9, 3, 9, 3, 5, 3, 8,
        3, 5, 9, 5, 9, 5])
```

Define our accuracy function:

```python
def accuracy(out, yb):
    return (out.argmax(dim=1)==yb).float().mean()

accuracy(preds, yb)
>>> tensor(0.08)
```

The accuracy is about 1/10, which is essentially as good as a random guess (given there are 10 classes). This is not surprising given this is the first step.

```python
def report(loss, preds, yb):
    """simple reporting function to dump out current accuracy"""
    print(f'{loss:.2f}, {accuracy(preds, yb):.2f}')

xb,yb = x_train[:bs],y_train[:bs]
preds = model(xb)
report(loss_func(preds, yb), preds, yb)
>>> 2.30, 0.08 # the loss / the accuracy
```

```python
lr = 0.5   # learning rate
epochs = 3 # how many epochs to train for

for epoch in range(epochs):
    for i in range(0, n, bs):
        s = slice(i, min(n,i+bs))
        xb,yb = x_train[s],y_train[s]
        preds = model(xb)
        loss = loss_func(preds, yb)
        loss.backward()
        with torch.no_grad():
            for l in model.layers:
                if hasattr(l, 'weight'):
                    l.weight -= l.weight.grad * lr
                    l.bias   -= l.bias.grad   * lr
                    l.weight.grad.zero_()
                    l.bias  .grad.zero_()
    report(loss, preds, yb)
```

```
0.02, 1.00
0.03, 1.00
0.03, 0.98
```