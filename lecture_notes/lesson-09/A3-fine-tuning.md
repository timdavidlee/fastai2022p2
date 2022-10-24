## 4. Fine Tuning

There are many ways of fine-tuning, but can take a lot of data + time. One is called `Textual inversion`


### 4.1 Fine Tuning: Textual Inversion

![Diagram](https://textual-inversion.github.io/static/images/training/training.JPG)

Lets say we want to create a new `style` for indian watercolor portraits. Art samples can be found here: [https://huggingface.co/sd-concepts-library/indian-watercolor-portraits](https://huggingface.co/sd-concepts-library/indian-watercolor-portraits). Let's call it `<watercolor-properties>`

#### 4.1.1 Download a 1 new word embedding for our new style name

```python
from diffusers import StableDiffusionPipeline
from fastdownload import FastDownload

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16
) 
pipe = pipe.to("cuda")

# will download a single single embedding
embeds_url = "https://huggingface.co/sd-concepts-library/indian-watercolor-portraits/resolve/main/learned_embeds.bin"
embeds_path = FastDownload().download(embeds_url)
embeds_dict = torch.load(str(embeds_path), map_location="cpu")

"""
embeds_dict 
{
    "<watercolor-portrait">: torch.tensor[float32, size=768]
}
"""
```

#### 4.1.2 Add the new token + insert the downloaded embedding

```python
# from the pipe, access the tokenizer
tokenizer = pipe.tokenizer

# from the pipe access the text_encoder
text_encoder = pipe.text_encoder

# a str, and a torch tensor
new_token, embeds = next(iter(embeds_dict.items()))
embeds = embeds.to(text_encoder.dtype)

# this was the key in the embed_dict
new_token
# >>> "<watercolor-portrait">

# add new token
text_encoder.resize_token_embeddings(len(tokenizer))
new_token_id = tokenizer.convert_tokens_to_ids(new_token)

# insert the embedding for our new token
text_encoder.get_input_embeddings().weight.data[new_token_id] = embeds
```

`--image--`

### 4.2 Fine Tuning: DreamBooth

Take a rare word in the existing vocabulary, then provide it images and fine tune the model. In this case a model was fine tuned with Jeremy Howard (the instructor) associated to the term `sks`

```python
from diffusers import StableDiffusionPipeline
from fastdownload import FastDownload

# downloading a model someone else finetuned
pipe = StableDiffusionPipeline.from_pretrained(
    "pcuenq/jh_dreambooth_1000",
    revision="fp16",
    torch_dtype=torch.float16
) 
pipe = pipe.to("cuda")

torch.manual_seed(1000)
prompt = "Painting of sks person in the style of Paul Signac"
images = pipe(prompt, num_images_per_prompt=4).images
image_grid(images, 1, 4)
```

`--image--`

