import math
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch

from model.config import GPTConfig
from model.gpt import GPT
from tokenizer.byte_bpe import ByteTokenizer

# ---- data / io ----
DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/ckpt.pt"
TOKENIZER_PATH = "tokenizer/vocab.json"

tokenizer = ByteTokenizer.load(TOKENIZER_PATH)

# ---- model ----
config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    d_model=384,
    n_layers=6,
    n_heads=6,
    n_kv_heads=2,
    d_ff=1536,
    max_seq_len=256,
)

# ---- optimization ----
MICRO_BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 4
MAX_ITERS = 600  # ~1h at ~6s/iter on M-series MPS; bump this up for a real run
WARMUP_ITERS = 60
MAX_LR = 3e-4
MIN_LR = 3e-5
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

# ---- eval / logging ----
EVAL_INTERVAL = 100
EVAL_ITERS = 20
LOG_INTERVAL = 10

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.bfloat16
print("Device:", device)


def load_split(split):
    path = f"{DATA_DIR}/{split}.bin"
    return np.memmap(path, dtype=np.uint16, mode="r")


train_data = load_split("train")
val_data = load_split("val")
print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")


def get_batch(split):
    data = train_data if split == "train" else val_data
    block_size = config.max_seq_len

    ix = torch.randint(len(data) - block_size - 1, (MICRO_BATCH_SIZE,))
    x = torch.stack([
        torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix
    ])
    return x.to(device), y.to(device)


def get_lr(it):
    if it < WARMUP_ITERS:
        return MAX_LR * (it + 1) / WARMUP_ITERS
    if it >= MAX_ITERS:
        return MIN_LR

    decay_ratio = (it - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)


@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}

    for split in ["train", "val"]:
        split_losses = torch.zeros(EVAL_ITERS)
        for i in range(EVAL_ITERS):
            x, y = get_batch(split)
            with torch.autocast(device_type=device, dtype=dtype):
                _, loss, _ = model(x, y)
            split_losses[i] = loss.item()
        losses[split] = split_losses.mean().item()

    model.train()
    return losses


model = GPT(config).to(device)
print(f"Params: {model.num_params():,}")

# no weight decay on norm gains (1D params); decay everything else
decay_params = [p for p in model.parameters() if p.dim() >= 2]
no_decay_params = [p for p in model.parameters() if p.dim() < 2]

optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=MAX_LR,
    betas=(0.9, 0.95),
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

best_val_loss = float("inf")
t0 = time.perf_counter()

for it in range(MAX_ITERS):
    lr = get_lr(it)
    for group in optimizer.param_groups:
        group["lr"] = lr

    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0

    for _ in range(GRAD_ACCUM_STEPS):
        x, y = get_batch("train")
        with torch.autocast(device_type=device, dtype=dtype):
            _, loss, _ = model(x, y)
            loss = loss / GRAD_ACCUM_STEPS
        loss.backward()
        accum_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()

    if it % LOG_INTERVAL == 0:
        dt = time.perf_counter() - t0
        t0 = time.perf_counter()
        print(f"iter {it}: loss {accum_loss:.4f}, lr {lr:.2e}, {dt:.2f}s")

    if (it % EVAL_INTERVAL == 0 and it > 0) or it == MAX_ITERS - 1:
        losses = estimate_loss()
        print(f"iter {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save({
                "model": model.state_dict(),
                "config": config,
                "iter": it,
                "val_loss": best_val_loss,
            }, CHECKPOINT_PATH)
            print(f"Saved checkpoint to {CHECKPOINT_PATH} (val loss {best_val_loss:.4f})")

print("Training complete.")
