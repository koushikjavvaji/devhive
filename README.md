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
2. **The war room app** (`app/`) — a FastAPI + WebSocket chat where you paste an error or
   stack trace into a room and multiple AI teammates respond, then argue with each other
   over a couple of rounds until they converge on one answer. The from-scratch model above
   is one of the teammates.

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
| `triage-bot` | nothing | Deterministic traceback parser — exception type, common-error hints, call chain. Always available, never reacts (nothing to debate). |
| `from-scratch-gpt` | a trained checkpoint (see below) | Our own model. Trained comment → code on a small corpus, so treat it as a rough first draft, not a diagnosis. Doesn't react either — it's a completion model, not a conversational one. |
| `gpt` | `OPENAI_API_KEY` | A single default OpenAI teammate, if you just want the one. |
| (any name you give it) | `DEVHIVE_LLM_PROVIDERS` | Any number of OpenAI-compatible providers — DeepSeek, Groq, OpenRouter, Together, etc. This is how you get more than one real LLM in the room at once. |

Copy `.env.example` to `.env` and fill in what you want; unset optional
vars just mean that teammate/override is skipped.

### how the "war" actually works

One message triggers two kinds of round, run by `Room` in `app/rooms.py`:

1. **Round 1** — every teammate answers the human independently and concurrently. Fast,
   no cross-talk yet.
2. **Reaction rounds** (`Room.REACTION_ROUNDS`, default 2) — any teammate with
   `can_react = True` (the real LLMs; `triage-bot` and `from-scratch-gpt` opt out, since a
   regex parser and a code-completion model can't hold an opinion) sees the *entire* room,
   including every earlier reaction round, and is explicitly told to agree, disagree, or
   call out something another teammate got wrong — not just restate its own answer.

With two independent LLMs configured, this produces a real debate that converges instead
of two isolated takes. From an actual run pasting a `'dict' object has no attribute
'expired'` traceback: after round 1, `nemotron` and `groq-oss` each proposed a fix; over
the next two rounds they cross-checked each other's reasoning, and the final message
stated the fix both had converged on — not scripted, an actual emergent agreement.

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

## tests

```bash
./.venv/bin/pip install -r requirements-dev.txt
./.venv/bin/python -m pytest
```

App tests use an isolated db + a fixed teammate list (`app.main.create_app`),
so they don't touch `data/devhive.db` or need a trained checkpoint on disk.

## project layout

```
tokenizer/    byte-level BPE (train/save/load/encode/decode)
model/        the transformer (embeddings, RoPE, GQA, SwiGLU, GPT)
inference/    shared generation logic (KV-cached), used by the CLI and the app
scripts/      train_tokenizer / prepare_data / train / generate
app/          the war room: FastAPI server, rooms, teammates, static chat UI
tests/        pytest suite for all of the above
```
