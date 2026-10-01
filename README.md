# devhive 🐝

[![tests](https://github.com/koushikjavvaji/devhive/actions/workflows/tests.yml/badge.svg)](https://github.com/koushikjavvaji/devhive/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**[Live demo](https://devhive-277r.onrender.com)** — free tier, spins down after 15 min idle
(first request after that takes ~30-60s to wake up). Runs `triage-bot` + real LLM
teammates; `from-scratch-gpt` is [local-only](#run-the-app) — the deploy deliberately
skips the torch/training stack to keep the free-tier build small and fast.

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
./.venv/bin/uvicorn app.main:create_app --factory --reload
```

Open http://localhost:8000, start a room, paste an error.

Teammates that join a room:

| Teammate | Needs | What it does |
|---|---|---|
| `triage-bot` | nothing | Deterministic error parser for Python, JS/Node, Java, Go and Rust — exception type (Java: the root `Caused by:`), common-error hints, call chain. Always available, never reacts (nothing to debate). |
| `from-scratch-gpt` | a trained checkpoint (see below) | Our own model. Trained comment → code on a small corpus, so treat it as a rough first draft, not a diagnosis. Doesn't react either — it's a completion model, not a conversational one. |
| `gpt` | `OPENAI_API_KEY` | A single default OpenAI teammate, if you just want the one. |
| (any name you give it) | `DEVHIVE_LLM_PROVIDERS` | Any number of OpenAI-compatible providers — DeepSeek, Groq, OpenRouter, Together, etc. This is how you get more than one real LLM in the room at once. |

Copy `.env.example` to `.env` and fill in what you want; unset optional
vars just mean that teammate/override is skipped.

### how the "war" actually works

One message triggers two kinds of round, run by `Room` in `app/rooms.py`:

1. **Round 1** — every teammate answers the human independently and concurrently. Fast,
   no cross-talk yet. LLM teammates also get the earlier room history (trimmed to a
   budget), so a follow-up like "that didn't work" makes sense to them.
2. **Reaction rounds** (`Room.REACTION_ROUNDS`, default up to 2) — any teammate with
   `can_react = True` (the real LLMs; `triage-bot` and `from-scratch-gpt` opt out, since a
   regex parser and a code-completion model can't hold an opinion) sees the room,
   including every earlier reaction round, and is explicitly told to agree, disagree, or
   call out something another teammate got wrong — not just restate its own answer. Each
   reaction ends with a `VERDICT: AGREE/DISAGREE` line (stripped before it's shown); once
   every reactor agrees, the debate stops early instead of burning another round.

Replies stream into the room token by token. Messages sent while a debate is still running
queue up and each gets its own full debate, in order.
Each teammate's turn has a timeout, so one hung provider can't stall the room.

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

Training writes two checkpoints: `checkpoints/ckpt.pt` (best val loss — what the app
loads) and `checkpoints/last.pt` (latest state including the optimizer). To keep going
after a run finishes or gets interrupted, raise `MAX_ITERS` if needed and run
`./.venv/bin/python -m scripts.train --resume`.

## tests

```bash
./.venv/bin/pip install -r requirements-dev.txt
./.venv/bin/python -m pytest
./.venv/bin/ruff check .
```

App tests use an isolated db + a fixed teammate list (`app.main.create_app`),
so they don't touch `data/devhive.db` or need a trained checkpoint on disk. The storage
tests also run against Postgres when `DEVHIVE_TEST_DATABASE_URL` points at one (CI
starts a Postgres service for this); otherwise those cases are skipped.

## deploying

`render.yaml` + `requirements-render.txt` deploy the app on [Render](https://render.com)'s
free tier, deliberately without `from-scratch-gpt`: `requirements-render.txt` skips
torch/numpy/datasets entirely (verified locally with no checkpoint present and no torch
installed at all — the app starts fine and just doesn't offer that teammate), which keeps
the free-tier build fast and well under the RAM limit.

To go live: push to GitHub, then in the Render dashboard, "New" → "Blueprint" → pick this
repo (it reads `render.yaml` automatically) → set `DEVHIVE_LLM_PROVIDERS` in the
Environment tab to your own provider JSON (same format as `.env.example`) → Deploy.

Room history: the free tier's disk is wiped on every deploy/restart, so with the default
SQLite file history doesn't survive. To keep it, create a free Postgres database
([Neon](https://neon.tech) or [Supabase](https://supabase.com)) and set its connection
string as `DATABASE_URL` in the Environment tab — the app creates its tables on startup.
(Locally, `data/devhive.db` persists across restarts as you'd expect.)

`render.yaml` also sets a health check on `/api/health`, so a deploy that fails to start
(e.g. a malformed `DEVHIVE_LLM_PROVIDERS`) never replaces the version that's running.

### limits

Every human message fans out to several paid LLM calls, so a public deploy is protected
by a few in-memory limits (all overridable via env, see `app/limits.py`):

| Env var | Default | |
|---|---|---|
| `DEVHIVE_MAX_MESSAGE_CHARS` | 8000 | longest message accepted |
| `DEVHIVE_MESSAGES_PER_IP` / `DEVHIVE_MESSAGES_WINDOW_SECONDS` | 10 / 300 | per-IP message rate |
| `DEVHIVE_ROOMS_PER_IP_PER_HOUR` | 20 | per-IP room creation |
| `DEVHIVE_DAILY_MESSAGE_LIMIT` | 500 | global cap across everyone |
| `DEVHIVE_LLM_TIMEOUT` | 45 | seconds before an LLM call gives up |
| `DEVHIVE_TRUSTED_PROXY_HOPS` | 1 | proxies in front of the app appending to `X-Forwarded-For` (0 = none) |

## project layout

```
tokenizer/    byte-level BPE (train/save/load/encode/decode)
model/        the transformer (embeddings, RoPE, GQA, SwiGLU, GPT)
inference/    shared generation logic (KV-cached), used by the CLI and the app
scripts/      train_tokenizer / prepare_data / train / generate
app/          the war room: FastAPI server, rooms, rate limits, teammates, static chat UI
tests/        pytest suite for all of the above
```
