# Before you get started

Be sure to create a huggingface account, and visit this webpage:

`https://huggingface.co/CompVis/stable-diffusion-v1-4`

You need to: read, accept, and acknowledge the LICENSE. if done correctly this will allow the following to be run:

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16
).to("cuda")
```

The above will pull the stable diffusion from the centralized huggingface modewl repository. This will also save the following information to a caching directory.