# Gilmore Girls Character Chatbot — Build Roadmap

Pick a character (Lorelai, Rory, Luke, Kirk, Sookie, Emily, Paris), send a
message, and the bot replies in their voice. Under the hood it's a retrieval-
grounded style-transfer system with a fidelity evaluation — fun on the surface,
real engineering underneath.

## The approach
For each character you build two things: a **style card** (an LLM-written
summary of how they talk) and an **embedded index of their real lines**. At chat
time you embed the user's message, retrieve that character's most relevant real
lines as few-shot examples, and assemble: style card + example lines + message →
a reply generated *in character*. That's a RAG system pointed at style instead
of facts — it reuses embeddings and retrieval directly. The evaluation then
measures whether each bot actually sounds like its character.

## How to run each session in Claude Code
Same loop every milestone: **Explore → Plan → Code → Commit.**
- Use **plan mode** (Shift+Tab twice, or `/plan`) for anything non-trivial. Read
  the plan it proposes, fix anything off, then approve.
- Every milestone ends with something you can **verify** — a test or a printed
  output. If Claude Code can check itself, it self-corrects.
- **Commit** at the end of each milestone. Keep `data/` out of every commit.
- Use `/compact` between milestones to keep context clean.

## Data and copyright
Gilmore Girls scripts are copyrighted. Use them locally; never commit the
transcripts or any derived lines (`data/raw/` and `data/processed/` are both
gitignored). The bot generates new dialogue in a character's style — it must not
reproduce lines verbatim. The public repo shows code, eval results, and only
short illustrative quotes.

---

## M0 — Scaffold + CLAUDE.md
**Goal:** a clean repo, environment, and skeleton with no real logic yet.

Drop the provided `CLAUDE.md` in your repo root first. This task is simple, so
skip plan mode and just type:

```
Set up the project skeleton described in CLAUDE.md. Create requirements.txt with
pandas, sentence-transformers, numpy, anthropic, streamlit, python-dotenv, and
pytest. Create a .gitignore that ignores .venv, .env, data/raw/, and
data/processed/. Create the src/, tests/, data/raw/, and data/processed/ folders
with placeholder files matching the layout in CLAUDE.md, a .env.example listing
ANTHROPIC_API_KEY, a stub README, and one passing placeholder test. Initialize a
git repo. Skeleton only — no real logic.
```

**Verify:** `pip install -r requirements.txt` installs cleanly; `pytest` runs and
the placeholder passes; the folder layout matches CLAUDE.md.
**Commit:** `scaffold project`.

---

## M1 — Load and parse the transcripts
**Goal:** a clean dict mapping each of the seven characters to a list of their lines.

By hand first: download a tidy Gilmore Girls lines dataset (a pre-parsed,
speaker-attributed CSV — search GitHub/Kaggle for "Gilmore Girls lines") and put
the file in `data/raw/`.

Use plan mode:

```
In plan mode. Write src/load.py to load the Gilmore Girls transcript dataset from
data/raw/ into a clean per-character structure. Read CLAUDE.md first. First
inspect the file's columns, since dataset formats vary. Then: load it with
pandas, normalize the speaker column (uppercase, strip whitespace, fix obvious
misspellings of the main names), keep only single-speaker lines for the seven
target characters (Lorelai, Rory, Luke, Kirk, Sookie, Emily, Paris), and drop
empty or voice-over lines. Expose a function returning a dict of character -> list
of lines. Print a line count per character at the end. Propose a plan first.
```

**Verify:** the per-character counts print and look right — Lorelai and Rory large,
Kirk and Sookie smaller.
**Commit:** code only (data stays gitignored).

---

## M2 — Style cards + embedded exemplars
**Goal:** for each character, a cached style card and a searchable index of their lines.

Use plan mode:

```
In plan mode. Write src/style.py with three capabilities. Read CLAUDE.md first.
(1) build_style_card(character): sample a spread of the character's lines, call
the Anthropic API to produce a short style card describing their voice
(vocabulary, rhythm, attitude, recurring references), and cache it to
data/processed/ so it's built once. (2) build_index(character): embed all of the
character's lines with sentence-transformers and save the vectors (a numpy array)
plus the line texts to data/processed/. (3) retrieve(character, query, k): embed
the query and return the k most cosine-similar real lines. No vector database
needed at this scale. Propose a plan first.
```

**Verify:** print Lorelai's style card and eyeball it; `retrieve("Lorelai", "I need
coffee", 5)` returns plausible lines.
**Commit.**

---

## M3 — In-character generation
**Goal:** `reply(character, message)` returns a new line in that character's voice.

Use plan mode:

```
In plan mode. Write src/generate.py with reply(character, message). Read CLAUDE.md
first. It should: load the character's cached style card, retrieve the top-k most
relevant real lines via src/style.retrieve, and assemble a prompt instructing the
model to answer AS the character — using the style card and example lines for
voice but generating NEW dialogue, never copying lines verbatim — then call the
Anthropic API and return the reply. Keep the system prompt and the few-shot
assembly as clearly separated, named pieces so they're easy to tune. Propose a
plan first.
```

**Verify:** `reply("Luke", "Can I get a refill?")` reads gruff and terse;
`reply("Emily", "Can I get a refill?")` reads formal and cutting. Spot-check a few
across characters.
**Commit.**

---

## M4 — Style-fidelity evaluation (the centerpiece)
**Goal:** measure whether each bot actually sounds like its character — the result
that turns this from a toy into a portfolio piece. Give this milestone real time.

Use plan mode:

```
In plan mode. Write src/evaluate.py for a style-fidelity evaluation. Read CLAUDE.md
first. Steps: define ~15 fixed neutral prompts; for each of the seven characters,
generate a reply with src/generate.reply; then for each reply, ask an Anthropic
LLM judge — given the seven character names and short descriptions but NOT the true
label — to guess which character produced it. Compute overall accuracy and a 7x7
confusion matrix of true vs predicted character. Save the matrix to data/processed/
and print a readable summary. Propose a plan first.
```

**Verify:** accuracy and the confusion matrix print; you can see which characters get
confused (does Rory get mistaken for Lorelai?). Then tune M3 — the prompt wording,
the number of retrieved examples — re-run, and watch the number move. That tuning
loop is the heart of the project.
**Commit.**

---

## M5 — The chat UI
**Goal:** a Streamlit chat with a character selector.

Use plan mode:

```
In plan mode. Write app.py: a Streamlit chat interface with a sidebar dropdown to
pick one of the seven characters, a chat input, and a running conversation that
calls src/generate.reply(character, message) each turn. Label each reply with the
character's name. Read CLAUDE.md first. Keep it simple — no auth, no database,
conversation state in st.session_state. Propose a plan first.
```

**Verify:** `streamlit run app.py`; chat with Lorelai, switch to Emily, and watch the
voice change.
**Commit.**

---

## M6 — Polish + portfolio writeup
**Goal:** make it legible to a stranger (and an interviewer).

- README with a screenshot or short GIF of the chat, a "how it works" section
  (the retrieval-grounding flow), and the eval results — accuracy and the confusion
  matrix — front and center.
- Confirm no raw or processed data is committed; double-check `.gitignore`.
- Lead the writeup with the real sentence: *"I built a retrieval-grounded character
  style-transfer system and measured its fidelity with an LLM-judge evaluation."*

**Commit and push.**

---

## Realistic shape
M0–M1 is one sitting. M2–M3 is the core build. M4 is the meaty one — spend time
there. M5 is the fun payoff. M6 wraps it. One milestone at a time.