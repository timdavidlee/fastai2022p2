# Reading a Paper together: Diffedit

**DiffEdit: Diffusion-based semantic image editing with mask guidance**
[Arxiv link](https://arxiv.org/pdf/2210.11427.pdf)

`Zotero` - chrome menu bar. The paper will appear in the software (`zotero`)

- can have the url
- has metadata
- can also read the paper
- annotate, edit + organize to papers

<img src="https://i.ibb.co/wJDH5sL/Screenshot-2022-10-30-at-4-55-08-PM.jpg" alt="Screenshot-2022-10-30-at-4-55-08-PM" border="3">

### Observations

- First sentence == "wow, how great is current advances!"
- Second sentence = "here is DiffEdit"
- Third sentence = "when an image and prompt are given, the generated image should retain as much of the original as possible"

Then from the pictures, it looks like the text prompt is closely aligned to the original image, so the generated image should only change what is requested.

**Main contribution**
Previous techniques usually require a "mask" to be manually supplied, but this papers' main contribution is to dynamically find the mask itself


### Introductions to papers

Talks about what its trying to do, tries to describe the problem + provide an overview to current methods.

<img src="https://i.ibb.co/59VKCwx/Screenshot-2022-10-30-at-5-02-46-PM.jpg" alt="Screenshot-2022-10-30-at-5-02-46-PM" border="0">

Looking at **related work** is a good place to look into current papers.

### Background

<img src="https://i.ibb.co/PcZNVWy/Screenshot-2022-10-30-at-5-06-23-PM.jpg" alt="Screenshot-2022-10-30-at-5-06-23-PM" border="0">

Scary time. A lot of equations. Background is often written last and intended to look smart to the reviewers.

Just a note: **No one in the world is going to look at this paragraph and immediately know.**

Let's walk through the math part of the papers. 

- Super helpful tip: learn the greek alphabet to interpret the equations

Equation 1 can be read as follows:

```
L = "loss"
epsilon = "true noise"
epsilon_theta = "noise estimator"
    - X_t = is the image at step `t`
    - t = the step number

|| epsilon - epsilon_theta(X_t, t)|| is the rank 2 norm
```

- What does double pipe mean: [Quora link](https://www.quora.com/What-does-mean-in-mathematics-2)
- What does the super + sub script mean: [Quora link](https://stats.stackexchange.com/questions/181620/what-is-the-meaning-of-super-script-2-subscript-2-within-the-context-of-norms)
- means the root sum of squares

What does the capital E mean?

- `Expected value` operator. In statistics the expected value is often the weighted average
- Say you have 50% chance of willing $10, the `E(situation) = 0.5 x $10 = $5` which means "in-general" people will win $5

### Looking at the "picture" algorithm diagram

<img src="https://i.ibb.co/brD5y3D/Screenshot-2022-10-30-at-5-28-18-PM.jpg" alt="Screenshot-2022-10-30-at-5-28-18-PM" border="0">

With walking through the talk step by step, the big idea is: 

1. run inference on the original image for the truth `horse` vs. `zebra` and then do a diff.
2. for the zebra inference, it will highlight the pixels relating to the animal and then leave the background alone
3. after inferring on both, doing the diff, will show that the background is the same on both images, this will be our MASK
4. then when going through the normal diffusion process, at every step, will replace the MASKED area (background) with the original, to ensure that the pixels remain the same

### Comment on Appendices:

- often contain experiments or lessons learned while developing the process