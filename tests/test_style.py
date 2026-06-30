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
    """Each test gets clean retrieval + centroid caches (lru_caches on the module)."""
    style._load_index.cache_clear()
    style._all_centroids.cache_clear()
    yield
    style._load_index.cache_clear()
    style._all_centroids.cache_clear()


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


# A geometry-controlled world for contrastive retrieval: exact text -> vector, so
# query similarity, centroids, and distinctiveness are all predictable.
_WORLD_MAP = {
    "the distinctive alpha thing": [1.0, 0.0, 0.0],   # far from B's voice
    "the shared topic here": [0.0, 1.0, 0.0],         # sits on B's centroid
    "bee line number one": [0.0, 1.0, 0.0],
    "bee line number two": [0.0, 1.0, 0.0],
    "bee line number three": [0.0, 1.0, 0.0],
    "query about topics here": [1.0, 1.0, 0.0],        # equally near both A lines
}
_WORLD_LINES = {
    "A": ["the distinctive alpha thing", "the shared topic here"],
    "B": ["bee line number one", "bee line number two", "bee line number three"],
}


class MapEmbedder:
    """Embeds known strings to fixed vectors (unknown -> zeros), then normalizes."""

    def encode(self, texts, normalize_embeddings=False, show_progress_bar=False):
        vecs = np.array([_WORLD_MAP.get(t, [0.0, 0.0, 0.0]) for t in texts], dtype="float32")
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.clip(norms, 1e-12, None)
        return vecs


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Two-character world (A, B) with controlled geometry for contrastive tests."""
    monkeypatch.setattr(style, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(style, "CHARACTERS", ("A", "B"))
    monkeypatch.setattr(style, "_all_lines", lambda: _WORLD_LINES)
    monkeypatch.setattr(style, "_embedder", lambda: MapEmbedder())
    return tmp_path


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


# --- _select: motif caps, near-dup drop, backfill ------------------------------

def test_select_caps_one_coffee_one_book():
    lines = ["coffee one cup", "coffee two cup", "a good book here", "plain line here"]
    vecs = np.eye(4, dtype="float32")             # all orthogonal -> no dups
    out = style._select(lines, vecs, 3)
    assert len(out) == 3
    assert sum("coffee" in ln for ln in out) == 1   # capped (non-coffee available)
    assert sum(("book" in ln or "read" in ln) for ln in out) == 1


def test_select_backfills_when_starved():
    lines = ["coffee one cup", "coffee two cup", "coffee three cup"]
    vecs = np.eye(3, dtype="float32")
    out = style._select(lines, vecs, 3)
    assert len(out) == 3                            # cap relaxed — nothing else to use
    assert sum("coffee" in ln for ln in out) == 3


def test_select_drops_near_duplicates():
    lines = ["unique line one", "unique line one too", "another distinct line"]
    vecs = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="float32")  # first two identical
    out = style._select(lines, vecs, 3)
    assert out == ["unique line one", "another distinct line"]


# --- centroid + contrastive retrieve -------------------------------------------

def test_build_centroid_is_unit_mean(world):
    c = style.build_centroid("A")                  # mean of [1,0,0] and [0,1,0]
    assert np.allclose(c, [0.7071, 0.7071, 0.0], atol=1e-3)
    assert np.isclose(np.linalg.norm(c), 1.0)


def test_retrieve_prefers_distinctive_over_shared(world):
    # both A lines are equally query-relevant; the one far from B's voice should win
    out = style.retrieve("A", "query about topics here", 2)
    assert out[0] == "the distinctive alpha thing"
    assert out[1] == "the shared topic here"


def test_retrieve_builds_index_on_demand(world):
    tmp_path = world
    assert not (tmp_path / "a_emb.npy").exists()
    style.retrieve("A", "query about topics here", 1)   # triggers build_index
    assert (tmp_path / "a_emb.npy").exists()


# --- preload -------------------------------------------------------------------

def test_preload_raises_on_missing_artifacts(world):
    with pytest.raises(FileNotFoundError) as exc:
        style.preload(("A", "B"))
    assert "a_card.md" in str(exc.value)            # lists what's missing


def test_preload_warms_when_present(world):
    tmp_path = world
    for c in ("A", "B"):
        style.build_index(c)                         # writes <c>_emb.npy + <c>_lines.json
        (tmp_path / f"{c.lower()}_card.md").write_text("card", encoding="utf-8")
    style.preload(("A", "B"))                        # all artifacts present -> no raise
    assert (tmp_path / "a_centroid.npy").exists()    # centroids warmed


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
