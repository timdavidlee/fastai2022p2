# Looking inside the Pipeline of Hugging Face

How do we get to the following?

```python
from diffusers import StableDiffusionPipeline
```

To get started at a high level, let's download some the key components needed for assembling everything

- `CLIPTextModel`: turn captions into vectors
- `CLIPTokenizer`: 

- `AutoencoderKL`: how to squish our image down to a manageable size
- `UNet2DConditionModel`: the unet model

```python
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import LMSDiscreteScheduler
```

## 1. Get the different components

Similar to the huggingface pattern, will download the different parts

```python
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-ema",
    torch_dtype=torch.float16
).to("cuda")

unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="unet",
    torch_dtype=torch.float16
).to("cuda")
```

Next we need a noise generator, or level of noise intensity generator. Will use the following relationship

The following is a scheduler is from `Katherine Crowson` [https://github.com/crowsonkb](https://github.com/crowsonkb) a AI/Generative artist

```python
beta_start, beta_end = 0.00085, 0.012
plt.plot(torch.linspace(beta_start**0.5, beta_end**0.5, 1000) ** 2)
plt.xlabel('Timestep')
plt.ylabel('β');
```

`--image--`

```python
scheduler = LMSDiscreteScheduler(
    beta_start=beta_start,
    beta_end=beta_end,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000
)
```

## 2. Lets generate the astronaut on a horse again

Here's the settings that we will use

```python
prompt = ["a photograph of an astronaut riding a horse"]

height = 512
width = 512
num_inference_steps = 70
guidance_scale = 7.5
batch_size = 1
```

```python
text_input = tokenizer(
    prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt"
)
```

Looking at the tokenized text: note that the `49407` represents padding to ensure all inputs are the same length.

```python
text_input['input_ids'] # torch.Size([1, 77])

# "a photograph of an astronaut riding a horse"
>>> tensor([[49406,   320,  8853,   539,   550, 18376,  6765,   320,  4558, 49407,
             49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
             49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
             49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
             49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
             49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
             49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
             49407, 49407, 49407, 49407, 49407, 49407, 49407]])

tokenizer.decode(49407)
>>> "<|endoftext|>"
```

In addition, the `text_input` also has an attention mask to ignore these "EOT" tokens:

```python
text_input["attention_mask"]  # torch.Size([1, 77])

# "a photograph of an astronaut riding a horse"
>>> tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0]])
```

Now one the text is run through the encoder, the appropriate embeddings are mapped over

```python
text_embeddings = text_encoder(text_input.input_ids.to("cuda"))[0].half()
text_embeddings.shape

# >>> torch.Size([1, 77, 768])
```

From our discussion before, we need to generate a **un-guided** photo, but as a result, still need to pass empty inputs to the encoder

Note that `.half()` means floating point 16, or `fp16` or `float16` which trades off decimal precision for size

```python
# ensure the input is the same length
max_length = text_input.input_ids.shape[-1]

# the token will be the empty string
uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)

# will get the same size, but the embeddings will the the same
uncond_embeddings = text_encoder(uncond_input.input_ids.to("cuda"))[0].half()
uncond_embeddings.shape

# >>> torch.Size([1, 77, 768]) 

# combine both inputs together for processing purposes
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
```

normal gaussian noise is generated, and note that the noise generated is at the compressed level, not at the original size:

- original size: `512px x 512px x 3`
- compressed size: `64px x 64px x 4`

```python

# lets create some noise
torch.manual_seed(100)
latents = torch.randn((batch_size, unet.in_channels, height // 8, width // 8))
latents = latents.to("cuda").half()
latents.shape

"""
>>> torch.Size([1, 4, 64, 64])
tensor([[[[ 0.1268,  1.3564,  0.5630,  ..., -0.8042, -0.6245, -0.5884],
          [ 1.6699, -0.9272, -0.9761,  ...,  2.3848, -0.0355, -0.3179],
          [ 0.3579, -1.7842, -0.3052,  ...,  0.4880, -2.5781,  1.0010],
          ...
"""

# set the scheduler - currently 70 steps
scheduler.set_timesteps(num_inference_steps)

# tensor = tensor * tensor(14.6146) as an example
latents = latents * scheduler.init_noise_sigma

"""
Normally would be a full 1000 steps, but our methods will be skipping a bit
To look at the approximations. Its not really a "timestep" it is just means how much noise

scheduler.timesteps

>>> tensor([999.0000, 984.5217, 970.0435, 955.5652, 941.0870, 926.6087, 912.1304,
            897.6522, 883.1739, 868.6957, 854.2174, 839.7391, 825.2609, 810.7826,
            796.3043, 781.8261, 767.3478, 752.8696, 738.3913, 723.9130, 709.4348,
            ... for 70 steps ... 0.0000]

scheduler.sigmas
>>> tensor([14.6146, 13.3974, 12.3033, 11.3184, 10.4301,  9.6279,  8.9020,  8.2443,
            7.6472,  7.1044,  6.6102,  6.1594,  5.7477,  5.3709,  5.0258,  4.7090,
            4.4178,  4.1497,  3.9026,  3.6744,  3.4634,  3.2680,  3.0867,  2.9183,
            ... for 70 steps .... 0.000]

Reminder that sigma is the 'variance' or the amount of noise, that is decreasing, starting
with lots of noise and decreasing it
"""
``` 


```python
from tqdm.auto import tqdm

# loop through decreasing amounts of noise
for i, t in enumerate(tqdm(scheduler.timesteps)):
    input = torch.cat([latents] * 2)
    input = scheduler.scale_model_input(input, t)

    # predict the noise residual
    with torch.no_grad():
        # remember the text_embeddings is [empty embeddings, prompt embeddings]
        # - passes some initial noise (input),
        # - a noise constant t, and our text ebmeddings input
        pred = unet(input, t, encoder_hidden_states=text_embeddings).sample
        # pred is torch.Size([1, 4, 64, 64])

    # perform guidance
    pred_uncond, pred_text = pred.chunk(2)

    # this takes the starting picture + adds some small part of the guidance
    pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)

    # compute the "previous" noisy sample
    # the latents are updated and then will be fed in again
    latents = scheduler.step(pred, t, latents).prev_sample

"""
Once this is complete, decompress the image with the VAE
to get generated + adjusted photo, then feed it through PIL to take a look
"""
with torch.no_grad():
    image = vae.decode(1 / 0.18215 * latents).sample
    # 0.18215 - is a constant determined by the paper
    # image.shape -> torch.Size([1, 3, 512, 512])

image = (image / 2 + 0.5).clamp(0, 1)
image = image[0].detach().cpu().permute(1, 2, 0).numpy()
image = (image * 255).round().astype("uint8")
Image.fromarray(image)
```

## 3. Refactoring for programming agility

The following code is the same as above, but will allow for reusability

```python
def text_enc(prompts: List[str], maxlen: int = None) -> torch.TensorType:
    """Encodes a list of strings
    Returns:
        torch.TensorType[torch.Size([1, 77, 768]), float16]
    """
    if maxlen is None:
        maxlen = tokenizer.model_max_length
    inp = tokenizer(
        prompts, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt"
    )
    return text_encoder(inp.input_ids.to("cuda"))[0].half()


def mk_img(t):
    """transforms raw decompressed image into PIL-compatible ranges + sizes"""
    image = (t / 2 + 0.5).clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((image*255).round().astype("uint8"))


def mk_samples(prompts: List[str], g=7.5, seed=100, steps=70):
    """infers based on Unet, and returns a VAE decoded image of torch.Size([1, 3, 512, 512])"""
    bs = len(prompts)
    text = text_enc(prompts)
    uncond = text_enc([""] * bs, text.shape[1])
    emb = torch.cat([uncond, text])
    if seed: torch.manual_seed(seed)

    latents = torch.randn((bs, unet.in_channels, height//8, width//8))
    scheduler.set_timesteps(steps)
    latents = latents.to("cuda").half() * scheduler.init_noise_sigma

    for i,ts in enumerate(tqdm(scheduler.timesteps)):
        inp = scheduler.scale_model_input(torch.cat([latents] * 2), ts)
        with torch.no_grad(): u,t = unet(inp, ts, encoder_hidden_states=emb).sample.chunk(2)
        pred = u + g*(t-u)
        latents = scheduler.step(pred, ts, latents).prev_sample

    with torch.no_grad():
        return vae.decode(1 / 0.18215 * latents).sample
```

And now can run some samples in only a few lines:

```python
prompts = [
    "a photograph of an astronaut riding a horse",
    "an oil painting of an astronaut riding a horse in the style of grant wood"
]

# will run the exact process above
images = mk_samples(prompts)
```

## Homework

- try and condense Image2Image, Negative Prompts, to really understand the code