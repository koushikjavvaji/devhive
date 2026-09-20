# devhive 🐝

an AI war room. throw bugs in, get answers out.

humans and AI teammates — debugging, testing, fixing. together.

AI teammates can be anything — openai, local models, or the model we built
from scratch right here.

## what's actually here

Two things:

1. **A GPT built from scratch** (`model/`, `tokenizer/`) — byte-level BPE
   tokenizer, RMSNorm, RoPE, grouped-query attention, SwiGLU, KV-cached
   inference. Trained comment → code on a
   [CodeSearchNet](https://huggingface.co/datasets/sentence-transformers/codesearchnet)
   subset.
2. **The war room app** (`app/`) — a FastAPI + WebSocket chat where you
   paste an error or stack trace into a room and multiple AI teammates
   respond concurrently. The from-scratch model above is one of them.

## setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## run the app

```bash
./.venv/bin/uvicorn app.main:app --reload
```

Open http://localhost:8000, start a room, paste an error.

Teammates that join a room:

| Teammate | Needs | What it does |
|---|---|---|
| `triage-bot` | nothing | Deterministic traceback parser — exception type, common-error hints, call chain. Always available. |
| `from-scratch-gpt` | a trained checkpoint (see below) | Our own model. Trained comment → code on a small corpus, so treat it as a rough first draft, not a diagnosis. |
| `gpt` | `OPENAI_API_KEY` | Real LLM diagnosis, only joins if the key is set. |

Copy `.env.example` to `.env` and fill in what you want; unset optional
vars just mean that teammate/override is skipped.

## training the model from scratch

```bash
./.venv/bin/python -m scripts.train_tokenizer   # -> tokenizer/vocab.json
./.venv/bin/python -m scripts.prepare_data       # -> data/train.bin, data/val.bin
./.venv/bin/python -m scripts.train              # -> checkpoints/ckpt.pt
./.venv/bin/python -m scripts.generate           # quick CLI sanity check
```

Each script has its scale (examples, merges, iterations) as constants at
the top — the checked-in defaults are scoped down to finish in
minutes/~1h on a laptop; bump them up for a real run. `scripts/train.py`
prints device + throughput so you can gauge how long a bigger run will
actually take before committing to it.

## project layout

```
tokenizer/    byte-level BPE (train/save/load/encode/decode)
model/        the transformer (embeddings, RoPE, GQA, SwiGLU, GPT)
inference/    shared generation logic (KV-cached), used by the CLI and the app
scripts/      train_tokenizer / prepare_data / train / generate
app/          the war room: FastAPI server, rooms, teammates, static chat UI
```
