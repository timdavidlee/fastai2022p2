# Random Numbers

Building a generator from scratch. There is no way to make a random number. Computers can track time, add + subtract numbers.  Most programs will generate a number that LOOKS random.

Generally:

- `rand()` should be different from call to call
- `rand()` should be uniformly distributed (all numbers have an equal chance of being selected)


`Wichmann - Hill: algorithm`


```python
random_state = None

def seed(a):
    global random_state
    a, x = divmod(a, 30268)
    a, y = divmod(a, 30306)
    a, z = divmod(a, 30322)

    # store these 3 vals in random state
    random_state = int(x) + 1, int(y) + 1, int(z) + 1


def rand():
    global random_state
    x, y, z = random_state
    x = (171 * x) % 30269
    y = (172 * y) % 30307
    z = (170 * z) % 30323
    random_state = x, y, z
    return (x / 30269 + y / 30307 + z / 30323) % 1.0  # pulls out the decimal part
```

Number generates rely on this `state`. `rand()` will keep generating random numbers as long as the random state is passed to the next function

```python
if os.fork():
    print(f"in parent {rand()}")
else:
    print(f"In child: {rand()}")
    os._exit(os.EX_OK)
"""
in parent 0.030043691943175688
In child: 0.030043691943175688
"""
```

Oops. they are the same, this is because during the `fork` the `random_state` was also copied along with copying the processes, meaning that the 2nd execution will result in the same generated number. This might be intentional, but in other cases it might be the opposite of what is desired:

- using parallel processes to augment images differently (should not repeat!)
- to do this the random number generator must be initialized in each process individually!
- even pytorch out of the box has this same issue

#### torch

```python
import torch

if os.fork():
    print(f"in parent {torch.rand(1)}")
else:
    print(f"In child: {torch.rand(1)}")
    os._exit(os.EX_OK)

"""
in parent tensor([0.3805])
In child: tensor([0.3805])

Also the same!
"""
```

#### numpy

```python
import numpy as np

if os.fork():
    print(f"in parent {np.random.rand(1)}")
else:
    print(f"In child: {np.random.rand(1)}")
    os._exit(os.EX_OK)

"""
in parent tensor([0.3805])
In child: tensor([0.3805])

Also the same!
"""
```

#### base python

```python
from random import random

if os.fork():
    print(f"in parent {random(1)}")
else:
    print(f"In child: {random(1)}")
    os._exit(os.EX_OK)

"""
in parent 0.9554070280183096
In child: 0.7616488033957549

Not the same
"""
```

### What about speed


Our method

```python
# from before
%%timeit -n 10
def chunks(x, sz):
    for i in range(0, len(x), sz):
        yield x[i: i + sz]

list(chunks([rand() for _ in range(7840)], 10))
# 4.46 ms ± 66.3 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

The torch method

```python
%%timeit -n 10
torch.randn(784, 10)
"""
The slowest run took 4.16 times longer than the fastest. This could mean that an intermediate result is being cached.
61.6 µs ± 45 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
"""
```