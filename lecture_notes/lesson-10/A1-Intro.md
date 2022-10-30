# Lesson 10

`Puru` - a bunch of images of (spherical) linear interpolation between two different starting points

Student examples of making different pictures

- Going from an old car prompt to a new car prompt
- dinosaur -> to a bird
- dog -> unicorn
- classic optimizer efforts, couple of hours

### Recap

- With Handwritten digits, 
- start with a 7, add some noise and get a "noisy 7"
- and feed this into Unet, and try to predict the noise
- we can also pass in the actual number 7 as "guidance" to help it along
- Turn captions into embeddings, via using existing images + captions (alt tab)
- Then train a text encoder + image encoder
- compare the dot product
- contrastive loss
- take the text encoder and feed into 

`--image--`

### Doing inference

- put a prompt + some noise
- Unet will predict noise
- will subtract gradually, and then resubmit the updated photo
- used to take 1000 steps, now takes 60 (maybe less!)

`--image--`

### Papers

A lot of interest + work still being focused on this work.

[Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512)
[On Distillation of guided Diffusion Models](https://arxiv.org/abs/2210.03142)
[Imagic: Text-Based Real Image Editing with Diffusion Models](https://arxiv.org/abs/2210.09276)

### Overview of "Progressive Distillation for Fast Sampling of Diffusion Models"

Hand-written Notes

