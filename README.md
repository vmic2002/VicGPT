# VicGPT

A from-scratch decoder-only GPT in PyTorch: custom multi-head causal attention, transformer blocks, next-token training, and autoregressive generation.

## Why this exists

This is a personal learning project. The goal is to build a GPT **line by line** so I understand how a transformer, a training pipeline, and inference actually work — not to ship a product or chase benchmarks.

Every line of the implementation was written **by hand in vim**, without AI writing the code for me. AI was only used for discussing ideas. If I had let AI write the code, I could probably finish the rest in a few days — but **that speed is not the point**. Going slowly and typing everything myself is how I learn.

*(Irony acknowledged: this README was drafted with AI help.)*

## What works today


| Area             | Status                                                                                                                                                                                                                                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Model**        | `[VicGPT](src/model/vic_gpt.py)`: token + learned positional embeddings, stacked `[TransformerBlock](src/model/transformer_block.py)`s, language-model head; `[generate](src/model/vic_gpt.py)` with temperature, top-p / top-k, sliding context window, EOS stop                                                                                      |
| **Attention**    | `[MultiHeadMaskedSelfAttention](src/model/attention.py)`: fused QKV linear, causal mask buffer, manual matmul attention                                                                                                                                                                                                                                |
| **Pre-training** | `[training.py](src/training/training.py)`: HuggingFace datasets (TinyStories by default), GPT-2 tokenizer, sequence packing to `max_seq_len`, shifted cross-entropy (`[loss.py](src/training/loss.py)`), AdamW, gradient clipping, checkpoint resume, Weights & Biases logging; saves `config.json` and checkpoints under `models/pre_trained/<name>/` |
| **SFT**          | CLI flag exists; `**train()` returns early** — not implemented yet                                                                                                                                                                                                                                                                                     |
| **Tokenizer**    | `[tokenizer.py](src/tokenizer/tokenizer.py)` is empty; training and inference use HuggingFace `AutoTokenizer`                                                                                                                                                                                                                                          |
| **Inference**    | `[testing_playground.py](testing_playground.py)` loads `best_model.pt` + `config.json` for interactive generation                                                                                                                                                                                                                                      |


Training and inference use **MPS if available, else CPU** (CUDA is not wired up yet).

**Rough project history** (from git):

- Initial commit → multi-head masked self-attention → transformer block → `VicGPT` forward + `generate` → pre-training script (packing, loss, W&B) → sliding-window generation + gradient clipping → SFT scaffolding (unfinished)

## Project layout

```
.
├── README.md
├── requirements.txt
├── pre_training.sh
├── testing_playground.py
└── src/
    ├── model/
    │   ├── attention.py          # MultiHeadMaskedSelfAttention
    │   ├── transformer_block.py  # pre-norm block + GELU MLP
    │   └── vic_gpt.py            # VicGPT + generate + top-p / top-k sampling
    ├── tokenizer/
    │   └── tokenizer.py          # placeholder — not implemented
    └── training/
        ├── loss.py               # next-token cross-entropy with padding mask
        └── training.py           # pre-training (+ SFT stub)
```

Runtime directories (gitignored, not shown above): `data/` (HuggingFace cache), `models/pre_trained/`, `models/sft/`, `wandb/`.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running things

### `pre_training.sh`

Shell wrapper for pre-training on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) with the GPT-2 tokenizer. It sets `HF_DATASETS_CACHE=./data` and runs:

```bash
export HF_DATASETS_CACHE=./data
python -m src.training.training \
    --pre_training \
    --dataset_name="roneneldan/TinyStories" \
    --tokenizer_name="gpt2" \
    --name_output_model="vicGPTv2" \
    --max_seq_len=2048 \
    --num_proc=8 \
    --train_batch_size=4 \
    --val_batch_size=4 \
    --num_epochs=10 \
    --lr=2e-4 \
    --weight_decay=0.01 \
    --num_layers=3 \
    --embed_dim=512 \
    --num_heads=4 \
    --mlp_hidden_dim=1024 \
    --dropout_prob=0.1
```

Or simply:

```bash
./pre_training.sh
```

Checkpoints and `config.json` are written to `models/pre_trained/vicGPTv2/`.

### `testing_playground.py`

Interactive inference after training:

```bash
python testing_playground.py
```

1. Prompts for a text prompt and `max_new_tokens`
2. Loads `models/pre_trained/vicGPTv2/config.json` and `best_model.pt`
3. Rebuilds `VicGPT` from the saved hyperparameters
4. Runs `generate()` with `temperature=0.1` and `top_p=0.9`
5. Prints the decoded output and how many tokens the checkpoint was trained on

You need a trained checkpoint at that path first (paths are hardcoded in the script).

## TODO

Consolidated from in-code comments. Pre-training works end-to-end; SFT and a custom tokenizer are the next big chunks; inference and training polish follow.

### Model — `[src/model/vic_gpt.py](src/model/vic_gpt.py)`

- Custom `Tokenizer` class with a HuggingFace-compatible interface (so both can be swapped easily)
- `generate()` with KV cache
- Unit tests for top-k and top-p sampling

### Training — `[src/training/training.py](src/training/training.py)`

- Implement SFT
- Cosine learning-rate decay + warmup steps
- Streaming for packing — avoid loading the full token stream into RAM

### Tokenizer — `[src/tokenizer/tokenizer.py](src/tokenizer/tokenizer.py)`

- Implement custom BPE / tokenizer here (file is currently empty)

