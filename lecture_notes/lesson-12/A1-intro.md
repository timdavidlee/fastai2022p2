# Introduction Remarks

[https://huggingface.co/spaces/pharma/CLIP-Interrogator](https://huggingface.co/spaces/pharma/CLIP-Interrogator)

Put an image, and it will generate a CLIP prompt. The CLIP prompt is intended to be supplied into  

-- hand written notes --

## Looking at the Hugging Face App code:

[https://huggingface.co/spaces/pharma/CLIP-Interrogator/blob/main/app.py](https://huggingface.co/spaces/pharma/CLIP-Interrogator/blob/main/app.py)

Note the reference data: [https://huggingface.co/spaces/pharma/CLIP-Interrogator/tree/main/data](https://huggingface.co/spaces/pharma/CLIP-Interrogator/tree/main/data)

The app will mix and match the various phrases to generate captions based on the uploaded image

`https://huggingface.co/spaces/pharma/CLIP-Interrogator/blob/main/data/artists.txt`

```
A. B. Jackson
A. J. Casson
A. R. Middleton Todd
A.B. Frost
A.D.M. Cooper
Aaron Bohrod
Aaron Douglas
Aaron Jasinski
Aaron Miller
Aaron Nagel
```

## BLIP Language Model

Bootstrapping Language Image Pre-training for Unified Vision-Language understanding

[arxiv paper](https://arxiv.org/abs/2201.12086)

This is specifically designed to give an ok caption from an image, but it is NOT the inverse of the CLIP encoder