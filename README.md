# clip-text-directions

[post](https://www.notion.so/Traversing-through-CLIP-Space-PCA-and-Latent-Directions-b898932e13684d58957405b4a2747a79?pvs=4#64fd32f7aaf14b3985751a94d184b3a4)

some experiments with extracting directions in CLIP text space, similar to what was done [here](https://github.com/ethansmith2000/clip-decomposition) with image embeddings

Simply creting opposing pairs, extracting the CLS token representation, calculating mean difference between pairs, and then adding this difference to the CLS token to prompt embeddings used for text2image generation with stable diffusion gives some neat ways to finely guide the outputs of your model.

I've also tried training a sparse-autoencoder to isolate concepts.

for some more interesitng ways to modify text embeddings in text2image models, we can use gradient-based methods I tried [here](https://github.com/ethansmith2000/SGDImagePrompt)

## Requirements
pytorch, diffusers, transformers, pandas

## How To Use
inference.ipynb - shows some examples of extracting directions and applying them for image generation
notebook.ipynb - does some cleaning of dataframes to isolate the text component and rename columns to be fed into get_embeds.py
get_embeds.py - script for extracting clip CLS text embeddings
train_sae.py - once embeddings are obtained, train a sparse autoencoder on them
