## Tech stack
- Python 3.11+, local venv
- Data: pandas
- Embeddings + retrieval: sentence-transformers, numpy cosine similarity
- LLM generation: Anthropic API
- Chat UI: Streamlit
- Tests: pytest
## Project layout
- `src/load.py`: load transcripts, parse into per-character line tables
- `src/style.py`: build per-character style cards + embedded exemplar index, retrieve(character, query, k)
- `src/generate.py`: assemble persona prompt + retrieved exemplars, call the LLM
- `src/evaluate.py`: style-fidelity eval (can a judge tell the characters apart)
- `app.py`: Streamlit chat UI with a character selector
- `data/raw/`: raw transcripts (gitignored, never committed)
- `data/processed/`: derived per-character lines, cards, embeddings (gitignored)
- `tests/`: pytest tests
## Commands
- `source .venv/bin/activate`
- `pip install -r requirements.txt`
- `pytest`
- `streamlit run app.py`
## Code style
- Small, single-purpose functions; type hints on anything public
- Use pandas for tabular work
- ANTHROPIC_API_KEY lives in .env, loaded with python-dotenv: never hardcode, never commit
- Named constants over magic numbers
## Workflow
- Plan before non-trivial changes; keep each plan to ~30 minutes of work
- End every milestone with a test or a printed expected output
- Commit per milestone with a clear message
## Data and copyright (important)
- Gilmore Girls scripts are copyrighted: never commit transcripts or derived
  lines. Both data/raw/ and data/processed/ are gitignored.
- The bot generates NEW dialogue in a character's style; it must never reproduce
  lines verbatim.
- The public repo shows code, the eval results, and only short illustrative quotes.
 - gitignore the CLAUDE.md and Roadmap.md files as well.