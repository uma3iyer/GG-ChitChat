# GG ChitChat

A character-style chat bot. Pick a Gilmore Girls character and chat with a bot
that generates **new** dialogue in that character's style — never reproducing
lines verbatim.

## How it works

1. **Load** (`src/load.py`) — parse transcripts into per-character line tables.
2. **Style** (`src/style.py`) — build per-character style cards + an embedded
   exemplar index; `retrieve(character, query, k)` pulls relevant exemplars.
3. **Generate** (`src/generate.py`) — assemble a persona prompt with retrieved
   exemplars and call the Anthropic API.
4. **Evaluate** (`src/evaluate.py`) — style-fidelity eval: can a judge tell the
   characters apart?
5. **Chat** (`app.py`) — Streamlit UI with a character selector.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your ANTHROPIC_API_KEY
```

## Commands

```bash
pytest                 # run tests
streamlit run app.py   # launch the chat UI
```

## Run the API

A FastAPI service (`api/main.py`) wraps the bot:

```bash
uvicorn api.main:app --reload          # http://127.0.0.1:8000
# GET  /health  -> {"status": "ok"}
# POST /chat    {"character": "Lorelai", "message": "hi"} -> {"reply": "..."}
```

Startup loads the embedder and all seven characters' style cards + embeddings
once. It **requires** the cached artifacts (`<char>_card.md`, `<char>_emb.npy`,
`<char>_lines.json` in `data/processed/`) — a missing artifact fails the boot by
design. Build them first via `src/style.py` (`build_style_card` + `build_index`).

## Data & copyright

Gilmore Girls scripts are copyrighted. Transcripts and any derived lines are
**never** committed — both `data/raw/` and `data/processed/` are gitignored. The
bot generates new dialogue and must never reproduce lines verbatim. This repo
shows code, eval results, and only short illustrative quotes.
