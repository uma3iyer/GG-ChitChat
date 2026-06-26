"""Offline unit tests for src/style.py.

These tests never hit the network or the Anthropic API and never read the real
(gitignored) transcript CSV. The embedder, the API client, the line source, and
the output directory are all replaced with deterministic fakes, so the
retrieval/caching logic is covered without a key or a model download.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import style


# --- deterministic fakes -------------------------------------------------------

# A tiny 3-axis "embedding": each line maps to counts of these keywords. This
# makes cosine similarity predictable — a "coffee" query lands on coffee lines.
_KEYS = ("coffee", "book", "dog")


class FakeEmbedder:
    """Stand-in for SentenceTransformer with a keyword-count embedding."""

    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(self, texts, normalize_embeddings=False, show_progress_bar=False):
        self.encode_calls += 1
        vecs = np.array(
            [[float(t.lower().count(k)) for k in _KEYS] for t in texts],
            dtype="float32",
        )
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.clip(norms, 1e-12, None)
        return vecs


class _FakeBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.content = [_FakeBlock(text)]


class FakeClient:
    """Stand-in for anthropic.Anthropic() that records call count."""

    CARD_TEXT = "STYLE CARD: talks fast, lots of pop-culture references."

    def __init__(self) -> None:
        self.calls = 0
        outer = self

        class _Messages:
            def create(self, **_kwargs):
                outer.calls += 1
                return _FakeMessage(outer.CARD_TEXT)

        self.messages = _Messages()


_LINES = {
    "Test": [
        "I love coffee so much",        # coffee
        "Coffee is the best thing",     # coffee
        "Reading my favorite book now",  # book
        "The big dog ran away",         # dog
        "Hi",                            # too short — dropped
        "no way",                        # 2 words — dropped
        "Yes I am",                      # 3 words — kept (no keyword)
    ]
}


@pytest.fixture(autouse=True)
def _clear_caches():
    """Each test gets a clean retrieval cache (it's an lru_cache on the module)."""
    style._load_index.cache_clear()
    yield
    style._load_index.cache_clear()


@pytest.fixture
def patched(tmp_path, monkeypatch):
    """Point style at a tmp output dir, fake lines, a fake embedder + client."""
    embedder = FakeEmbedder()
    client = FakeClient()
    monkeypatch.setattr(style, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(style, "_all_lines", lambda: _LINES)
    monkeypatch.setattr(style, "_embedder", lambda: embedder)
    monkeypatch.setattr(style, "_client", lambda: client)
    return embedder, client, tmp_path


# --- pure-logic helpers --------------------------------------------------------

def test_personality_lines_drops_short(patched):
    lines = style._personality_lines("Test")
    assert "Hi" not in lines           # 1 word
    assert "no way" not in lines       # 2 words
    assert "Yes I am" in lines         # 3 words — boundary kept
    assert "I love coffee so much" in lines
    assert all(len(ln.split()) >= style.MIN_WORDS for ln in lines)


def test_even_sample_returns_all_when_small():
    assert style._even_sample(["a", "b", "c"], 5) == ["a", "b", "c"]


def test_even_sample_spreads_across_list():
    sample = style._even_sample(["a", "b", "c", "d"], 2)
    assert sample == ["a", "c"]        # step 2.0 → indices 0, 2
    assert len(style._even_sample([str(i) for i in range(100)], 10)) == 10


# --- build_index ---------------------------------------------------------------

def test_build_index_writes_aligned_files(patched):
    _embedder, _client, tmp_path = patched
    style.build_index("Test")

    emb = np.load(tmp_path / "test_emb.npy")
    import json

    lines = json.loads((tmp_path / "test_lines.json").read_text())

    kept = style._personality_lines("Test")
    assert emb.shape == (len(kept), len(_KEYS))   # one row per kept line
    assert lines == kept                          # texts align with vectors
    # vectors with a keyword are unit-normalized
    coffee_row = emb[lines.index("I love coffee so much")]
    assert np.isclose(np.linalg.norm(coffee_row), 1.0)


def test_build_index_is_cached(patched):
    embedder, _client, _tmp = patched
    style.build_index("Test")
    style.build_index("Test")                     # files exist → must skip
    assert embedder.encode_calls == 1


def test_build_index_rebuild_forces_reencode(patched):
    embedder, _client, _tmp = patched
    style.build_index("Test")
    style.build_index("Test", rebuild=True)
    assert embedder.encode_calls == 2


# --- retrieve ------------------------------------------------------------------

def test_retrieve_returns_k_lines(patched):
    out = style.retrieve("Test", "coffee please", 3)
    assert len(out) == 3
    assert all(isinstance(s, str) for s in out)


def test_retrieve_ranks_by_cosine(patched):
    coffee_lines = {"I love coffee so much", "Coffee is the best thing"}
    top2 = style.retrieve("Test", "coffee", 2)
    assert set(top2) == coffee_lines              # both coffee lines outrank the rest

    assert style.retrieve("Test", "book", 1)[0] == "Reading my favorite book now"
    assert style.retrieve("Test", "dog", 1)[0] == "The big dog ran away"


def test_retrieve_builds_index_on_demand(patched):
    _embedder, _client, tmp_path = patched
    assert not (tmp_path / "test_emb.npy").exists()
    style.retrieve("Test", "coffee", 1)           # triggers build_index
    assert (tmp_path / "test_emb.npy").exists()


# --- build_style_card ----------------------------------------------------------

def test_build_style_card_writes_and_caches(patched):
    _embedder, client, tmp_path = patched

    card = style.build_style_card("Test")
    assert card == FakeClient.CARD_TEXT
    assert (tmp_path / "test_card.md").exists()
    assert client.calls == 1

    again = style.build_style_card("Test")        # cached on disk → no second call
    assert again == FakeClient.CARD_TEXT
    assert client.calls == 1


def test_build_style_card_rebuild_recalls_client(patched):
    _embedder, client, _tmp = patched
    style.build_style_card("Test")
    style.build_style_card("Test", rebuild=True)
    assert client.calls == 2
