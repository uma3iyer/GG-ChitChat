"""Build per-character style cards + an embedded exemplar index, and retrieve exemplars.

Turns each character's cleaned lines (from ``load.py``) into:
1. a short style card describing their voice (written once by the LLM, cached), and
2. an embedded exemplar index for ``retrieve(character, query, k)``.

Lines shorter than ``MIN_WORDS`` words are dropped everywhere — one- and two-word
lines ("Mom", "Yeah", "Sure") carry little personality.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from .load import lines_by_character, CHARACTERS

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
EMBED_MODEL = "all-MiniLM-L6-v2"   # small, standard sentence-transformer (384-dim)
CARD_MODEL = "claude-opus-4-8"
MIN_WORDS = 3                      # drop low-personality lines ("Mom", "Yeah", "Sure")
CARD_SAMPLE_SIZE = 80              # exemplar lines shown to the LLM when writing a card
CARD_MAX_TOKENS = 1024
CONTRAST_POOL = 30                 # candidate pool by query similarity before reranking
CONTRAST_WEIGHT = 0.5             # weight on distinctiveness vs query similarity
MOTIF_CAP = 1                     # <=1 coffee line and <=1 book/read line in the final k
DUP_SIM = 0.9                     # cosine >= this -> treat as a near-duplicate

_CARD_PROMPT = """You are a dialogue-style analyst for the show Gilmore Girls.
Below are sample lines spoken by {character}. Write a short STYLE CARD (a few
tight bullet points, ~150 words) capturing how {character} talks:
- vocabulary and references they reach for
- sentence rhythm and pacing
- attitude / emotional default

Describe the style in your own words. Do NOT quote or reproduce the sample lines
verbatim — this card must not copy copyrighted dialogue.

Sample lines:
{exemplars}
"""


@lru_cache(maxsize=1)
def _all_lines() -> dict[str, list[str]]:
    return lines_by_character()            # load + clean the CSV once per process


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _client():
    import anthropic

    load_dotenv()                          # pull ANTHROPIC_API_KEY from .env
    return anthropic.Anthropic()


def _personality_lines(character: str) -> list[str]:
    """A character's lines with at least MIN_WORDS words (more voice, less filler)."""
    return [ln for ln in _all_lines()[character] if len(ln.split()) >= MIN_WORDS]


def _even_sample(items: list[str], n: int) -> list[str]:
    """Evenly-spaced sample across the list to get a spread (deterministic)."""
    if len(items) <= n:
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def build_style_card(character: str, rebuild: bool = False) -> str:
    """Return a short style card for the character, building + caching it once."""
    path = PROCESSED_DIR / f"{character.lower()}_card.md"
    if path.exists() and not rebuild:
        return path.read_text(encoding="utf-8")

    sample = _even_sample(_personality_lines(character), CARD_SAMPLE_SIZE)
    prompt = _CARD_PROMPT.format(
        character=character,
        exemplars="\n".join(f"- {ln}" for ln in sample),
    )
    msg = _client().messages.create(
        model=CARD_MODEL,
        max_tokens=CARD_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    card = "".join(b.text for b in msg.content if b.type == "text").strip()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(card, encoding="utf-8")
    return card


def build_index(character: str, rebuild: bool = False) -> None:
    """Embed the character's personality lines and save vectors + texts to disk."""
    emb_path = PROCESSED_DIR / f"{character.lower()}_emb.npy"
    txt_path = PROCESSED_DIR / f"{character.lower()}_lines.json"
    if emb_path.exists() and txt_path.exists() and not rebuild:
        return

    lines = _personality_lines(character)
    vecs = np.asarray(
        _embedder().encode(lines, normalize_embeddings=True, show_progress_bar=True),
        dtype="float32",
    )                                  # unit vectors → cosine == dot product

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, vecs)
    txt_path.write_text(json.dumps(lines, ensure_ascii=False), encoding="utf-8")


@lru_cache(maxsize=None)
def _load_index(character: str) -> tuple[np.ndarray, tuple[str, ...]]:
    emb_path = PROCESSED_DIR / f"{character.lower()}_emb.npy"
    txt_path = PROCESSED_DIR / f"{character.lower()}_lines.json"
    if not (emb_path.exists() and txt_path.exists()):
        build_index(character)
    vecs = np.load(emb_path)
    lines = tuple(json.loads(txt_path.read_text(encoding="utf-8")))  # hashable for cache
    return vecs, lines


def build_centroid(character: str, rebuild: bool = False) -> np.ndarray:
    """Unit-normalized mean of the character's line vectors (cached as <char>_centroid.npy)."""
    path = PROCESSED_DIR / f"{character.lower()}_centroid.npy"
    if path.exists() and not rebuild:
        return np.load(path)
    vecs, _ = _load_index(character)
    c = vecs.mean(axis=0)
    norm = np.linalg.norm(c)
    c = (c / norm if norm else c).astype("float32")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, c)
    return c


@lru_cache(maxsize=1)
def _all_centroids() -> dict[str, np.ndarray]:
    """Per-character voice centroids, used to score how distinctive a line is."""
    return {c: build_centroid(c) for c in CHARACTERS}


def _select(lines: list[str], vecs: np.ndarray, k: int) -> list[str]:
    """Take up to k lines: motif caps + near-dup drop, backfilling so small chars aren't starved."""
    sel_lines: list[str] = []
    sel_vecs: list[np.ndarray] = []
    held: list[tuple[str, np.ndarray]] = []
    coffee = book = 0

    def is_dup(v: np.ndarray) -> bool:
        return any(float(v @ sv) >= DUP_SIM for sv in sel_vecs)

    for line, v in zip(lines, vecs):
        if is_dup(v):
            continue
        low = line.lower()
        ic = "coffee" in low
        ib = "book" in low or "read" in low          # 'read' covers reading
        if (ic and coffee >= MOTIF_CAP) or (ib and book >= MOTIF_CAP):
            held.append((line, v))                   # capped now; may backfill if starved
            continue
        sel_lines.append(line)
        sel_vecs.append(v)
        coffee += ic
        book += ib
        if len(sel_lines) == k:
            return sel_lines
    for line, v in held:                             # relax caps only if we couldn't fill k
        if is_dup(v):
            continue
        sel_lines.append(line)
        sel_vecs.append(v)
        if len(sel_lines) == k:
            break
    return sel_lines


def retrieve(character: str, query: str, k: int) -> list[str]:
    """Top-k lines: query-relevant AND distinctive to the character (contrastive rerank)."""
    vecs, lines = _load_index(character)
    q = np.asarray(
        _embedder().encode([query], normalize_embeddings=True), dtype="float32"
    )[0]
    sims = vecs @ q                              # both unit-normalized → cosine similarity
    pool = np.argsort(-sims)[:CONTRAST_POOL]     # candidate pool by query similarity

    cents = _all_centroids()
    own = cents[character]
    others = [c for name, c in cents.items() if name != character]
    cand = vecs[pool]
    own_sim = cand @ own
    other_sim = (
        (cand @ np.stack(others).T).max(axis=1) if others else np.zeros(len(pool))
    )
    distinct = own_sim - other_sim               # distinctiveness vs other characters
    final = sims[pool] + CONTRAST_WEIGHT * distinct
    ranked = pool[np.argsort(-final)]
    return _select([lines[i] for i in ranked], vecs[ranked], k)


def main() -> None:
    character = "Lorelai"
    print(f"=== {character} style card ===")
    print(build_style_card(character))
    build_index(character)
    print(f"\n=== retrieve('{character}', 'I need coffee', 2) ===")
    for line in retrieve(character, "I need coffee", 2):
        print("-", line)


if __name__ == "__main__":
    main()
