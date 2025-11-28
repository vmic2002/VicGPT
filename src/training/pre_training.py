from datasets import load_dataset, Dataset
from src.model.vic_gpt import VicGPT
from transformers import AutoTokenizer
# TODO WRITE loss.py FIRST from .loss import Loss

#ds = load_dataset("wikitext", "wikitext-103-v1")
ds = load_dataset("roneneldan/TinyStories")
print("Dataset loaded")

i=0
for item in ds["train"]:
    print(f"entry {i}")
    print(item["text"])
    print("-"*40)
    i+=1
    if i>2:
        break

tokenizer_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token # gpt2 has no pad_token

vicGPT = VicGPT(
    vocab_size=tokenizer.vocab_size,
    num_layers=1,
    embed_dim=512,
    num_heads=4,
    max_seq_len=128,
    mlp_hidden_dim=256,
    dropout_prob=0.1
)



# TODO TOKENIZE ENTIRE DATASET WITH .MAP
# USE DATALOADER AND BATCHING
# TRAIN FOR EVERY EPOCH
# also every var, including batch size should be export in the pre_training.sh in the root directory
print("done")
