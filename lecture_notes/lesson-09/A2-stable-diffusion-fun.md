# Stable Diffusion Introduction

This is a combination of verbal notes + summarizations from this notebook walkthrough:

- [github link: https://github.com/fastai/diffusion-nbs](https://github.com/fastai/diffusion-nbs)

## 1. About Hugging Face Pipelines, Generate your first image

Can pass `pipe()` prompts and we can get some images out of it.

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16
).to("cuda")

# pass one prompt
prompt = "a photograph of an astronaut riding a horse"
pipe(prompt).images[0]

# can change the randomized seed, will generate same image dif
torch.manual_seed(1024)
pipe(prompt).images[0]
```

`--image--`

### 1.1 What's happening step by step during image generation?

The above images are generated around ~50 steps. 

- starts with random noise
- slowly makes it less noisy to form it into an image

And at a high level, thats how these models are created

### 1.2 Why not just 1 step?

These models are not smart enough to do this in one-go. [**Caveat**] this is a very cutting-edge field with a lot of research, many of the new papers coming out see to shorten this step requirement to say 2-3 steps (instead of 51)

### 1.3 What happens with fewer steps?

The images are not as clear, and appear to be more "cloudy" can "convoluted"

## 2. Concept: `Guidance Scale`

`Guidance Scale`: What degree / intensity should focus be on the caption (words) vs. creating an image

First, a little helper function to display multiple images, will use to show images along different steps

``` python
def image_grid(imgs, rows, cols):
    """For displaying a grid of images"""
    w,h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs): grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

# first generate 4 images
guidance_scale = [
    1.1 3, 7, 14,
]
num_rows,num_cols = 4, 4
prompts = [prompt] * num_cols
images = concat(pipe(prompts, guidance_scale=g).images for g in guidance_scale)

# then show 4 images
image_grid(images, rows=num_rows, cols=num_cols)
```

`--image--`

### 2.1 What's happening behind the scenes?

The algorithm is doing generating 2 images:

1. image with the prompt
2. image with no prompt

Then it will average the two images together, and `Guidance scale` is a parameter that "weights" the average in a way

#### 2.1.2 Subtraction instead of averaging

Can `subtract` the two images instead of `average` and this can be accomplished with an additional function param:

```python
# a dog
torch.manual_seed(1000)
prompt = "Labrador in the style of Vermeer"
pipe(prompt).images[0]

# remove a prompt for the provided caption
torch.manual_seed(1000)
pipe(prompt, negative_prompt="blue").images[0]
```

## 3. Forget the captions, pass an image

This can be thought of jump-starting right to the noisy image first. 

- start with an image
- add some noise
- pre

```python
from diffusers import StableDiffusionImg2ImgPipeline
from fastdownload import FastDownload

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16,
).to("cuda")

# download an image to start with 
# VERY BASIC
p = FastDownload().download(
    'https://s3.amazonaws.com/moonup/production/uploads/1664665907257-noauth.png'
)
init_image = Image.open(p).convert("RGB")
init_image
```

`--image--`

Then the following code will use the random sketch as a starting point. The consideration factor is defined as `strength=0.8`, meaning the algo should stick the original closely vs. exploring / altering the image

```python
torch.manual_seed(1000)
prompt = "Wolf howling at the moon, photorealistic 4K"
images = pipe(
    prompt=prompt,
    num_images_per_prompt=3,
    init_image=init_image,
    strength=0.8,
    num_inference_steps=50
).images
image_grid(images, rows=1, cols=3)
```

`--image--`

### 3.1 Some advanced tricks: sketch -> generated image -> ANOTHER generated image

1. start with the same sketch
2. generate a decent image from the sketch starting point
3. then use the output as the starting point again, with a different prompt

```python
# download an image to start with 
# VERY BASIC
p = FastDownload().download(
    'https://s3.amazonaws.com/moonup/production/uploads/1664665907257-noauth.png'
)

# generate an image
torch.manual_seed(1000)
prompt = "Wolf howling at the moon, photorealistic 4K"
images = pipe(
    prompt=prompt,
    num_images_per_prompt=3,
    init_image=init_image,
    strength=0.8,
    num_inference_steps=50
).images
image_grid(images, rows=1, cols=3)

# take the last image of generated set
init_image = images[2]

# and now apply a different prompt
torch.manual_seed(1000)
prompt = "Oil painting of wolf howling at the moon by Van Gogh"
images = pipe(
    prompt=prompt,
    num_images_per_prompt=3,
    init_image=init_image,
    strength=1,
    num_inference_steps=70
).images
image_grid(images, rows=1, cols=3)
```

#### 3.1.1 Lambda Labs using the technique to make a text-2-pokemon model

[Lambda Labs Blog - Making the Pokemon Model](https://lambdalabs.com/blog/how-to-fine-tune-stable-diffusion-how-we-made-the-text-to-pokemon-model-at-lambda/)

In a nutshell:

- started with a database of pokemon images: [huggingface link of pokemon images + captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions)
-  then fine tuned one of the `stable-diffusion` models
- then it can take prompts to make pokemon!: 
    - `Girl with a pearl earring`
    - `Cute Obama creature`
    - `Donald Trump`
    - `Boris Johnson`
    - `Totoro`
    - `Hello Kitty`

`--image--`