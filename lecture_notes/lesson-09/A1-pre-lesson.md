# Before you get started


### 1. get a hugging face account

Be sure to create a huggingface account, and visit this webpage:

### 2. get a hugging face API token

Go to your profile and look under `settings` and then go to `Access Tokens`.


### 3. read + accept license (or you will be blocked)

Here is the `model card`, the description + developer: `https://huggingface.co/CompVis/stable-diffusion-v1-4` [link](https://huggingface.co/CompVis/stable-diffusion-v1-4)

You need to: read, accept, and acknowledge the LICENSE. if done correctly this will allow the following to be run:

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16
).to("cuda")
```

The above will pull the stable diffusion from the centralized huggingface modewl repository. This will also save the following information to a caching directory. This is passed during `run_docker` as an environmental variable.


### 4. Check your pytorch version

```python
import torch

torch.__version__
```

The following means its pytorch intended for CUDA 10.2 which is old
```
>>> 1.12.1+cu102
```

### 5. Common driver errors:

- you have a `gpu` but there's no NVIDIA drivers
- the container has `CUDA==10.2`, but `pytorch==cu11.3`, meaning the versions don't match

### 6. Can pytorch see the gpu?

```python
import torch

print(torch.cuda.is_available())
>>> True
```

Note: if the answer is `False` there is often a warning that is logged to give a hint at what is missing


### 7. is the GPU even available?

```shell
$ nvidia-smi
```