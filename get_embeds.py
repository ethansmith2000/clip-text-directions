# !wget https://storage.googleapis.com/conceptual_12m/cc12m.tsv
# !wget https://huggingface.co/datasets/wanng/midjourney-v5-202304-clean/resolve/main/data/ori_prompts_df.parquet

import transformers
import torch
import glob
import pandas as pd
from tqdm import tqdm

df = pd.read_parquet("/home/ubuntu/clip-text-directions/full.parquet")
texts = df["text"].apply(lambda x: str(x))
texts = texts.to_list()

clip = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda").to(torch.float16)
tokenizer = transformers.AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

clip_forward = torch.compile(clip)

def get_embeds(batch):
    input_ids = tokenizer(batch, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(clip.device)
    with torch.no_grad():
        return clip_forward(input_ids).pooler_output

batch_size = 4096
cpu_steps = 64
all_embeds = []

embed_batch = []
for i in tqdm(range(0, len(df), batch_size)):
    batch = texts[i:i+batch_size]
    embeds = get_embeds(batch)
    embed_batch.append(embeds)
    if len(embed_batch) % cpu_steps == 0:
        embed_batch = torch.cat(embed_batch)
        embed_batch = embed_batch.to("cpu")
        all_embeds.append(embed_batch)
        embed_batch = []

if len(embed_batch) > 0:
    embed_batch = torch.cat(embed_batch)
    embed_batch = embed_batch.to("cpu")
    all_embeds.append(embed_batch)


torch.save(all_embeds, "all_embeds.pt")
