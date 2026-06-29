"""Offline unit tests for src/evaluate.py.

Collaborators (generation, the judge API client, randomness) are faked, and the
output dir is redirected to a tmp path, so the three steps and the metrics math
are verified without a key, network, or the real CSV.
"""

from __future__ import annotations

import json

import pytest

from src import evaluate
from src.load import CHARACTERS


class _FakeBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.content = [_FakeBlock(text)]


class FakeClient:
    """Returns canned judge responses from a queue; records call kwargs."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        outer = self

        class _Messages:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                text = outer._responses.pop(0) if outer._responses else ""
                return _FakeMessage(text)

        self.messages = _Messages()


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Redirect all eval artifact paths into a tmp dir for every test."""
    monkeypatch.setattr(evaluate, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(evaluate, "REPLIES_PATH", tmp_path / "eval_replies.json")
    monkeypatch.setattr(evaluate, "JUDGMENTS_PATH", tmp_path / "eval_judgments.json")
    monkeypatch.setattr(evaluate, "MATRIX_CSV", tmp_path / "confusion_matrix.csv")
    monkeypatch.setattr(evaluate, "HEATMAP_PNG", tmp_path / "confusion_matrix.png")
    return tmp_path


# --- step 1: generate ----------------------------------------------------------

def test_generate_per_character_counts_and_cost_params(monkeypatch):
    monkeypatch.setattr(evaluate, "EVAL_PROMPTS", ["p1", "p2"])
    seen = []

    def fake_reply(ch, msg, **kwargs):
        seen.append((ch, kwargs))
        return f"{ch} says hi to {msg}"

    monkeypatch.setattr(evaluate.generate, "reply", fake_reply)

    rows = evaluate.generate_replies()
    expected = sum(evaluate.GEN_PER_CHARACTER[c] for c in CHARACTERS) * 2
    assert len(rows) == expected
    assert set(rows[0]) == {"true_character", "prompt", "gen_index", "reply"}
    # per-character row counts follow GEN_PER_CHARACTER × prompts (not a fixed number)
    for c in CHARACTERS:
        n = sum(1 for r in rows if r["true_character"] == c)
        assert n == evaluate.GEN_PER_CHARACTER[c] * 2
    # every generation call bounds cost
    assert all(
        kw.get("effort") == evaluate.EVAL_EFFORT
        and kw.get("max_tokens") == evaluate.EVAL_MAX_TOKENS
        for _, kw in seen
    )


def test_generate_uses_cache_then_regenerate(monkeypatch):
    monkeypatch.setattr(evaluate, "EVAL_PROMPTS", ["p1"])
    calls = {"n": 0}

    def fake_reply(ch, msg, **kwargs):
        calls["n"] += 1
        return "x"

    monkeypatch.setattr(evaluate.generate, "reply", fake_reply)

    evaluate.generate_replies()
    first = calls["n"]
    assert first == sum(evaluate.GEN_PER_CHARACTER[c] for c in CHARACTERS)
    evaluate.generate_replies()                 # cache hit -> no new calls
    assert calls["n"] == first
    evaluate.generate_replies(regenerate=True)  # forced rebuild
    assert calls["n"] == 2 * first


# --- _parse_name ---------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("Lorelai", "Lorelai"),
        ("  Rory.\n", "Rory"),
        ("I think it's Kirk, honestly", "Kirk"),
        ("Lorelai or Rory", None),     # ambiguous
        ("no idea", None),             # none
    ],
)
def test_parse_name(text, expected):
    assert evaluate._parse_name(text) == expected


# --- step 2: judge -------------------------------------------------------------

def test_judge_retries_once_then_succeeds(monkeypatch):
    client = FakeClient(["gibberish", "Lorelai"])   # malformed, then valid
    monkeypatch.setattr(evaluate, "_client", lambda: client)
    shuffles = {"n": 0}
    monkeypatch.setattr(evaluate.random, "shuffle", lambda seq: shuffles.__setitem__("n", shuffles["n"] + 1))

    assert evaluate._judge("some reply") == "Lorelai"
    assert len(client.calls) == 2          # one retry
    assert shuffles["n"] == 2              # shuffled fresh each call


def test_judge_gives_up_after_retry(monkeypatch):
    client = FakeClient(["nope", "still nope"])
    monkeypatch.setattr(evaluate, "_client", lambda: client)
    monkeypatch.setattr(evaluate.random, "shuffle", lambda seq: None)
    assert evaluate._judge("reply") is None
    assert len(client.calls) == evaluate.JUDGE_RETRIES + 1


def test_judge_is_blind_to_prompt_and_lists_all_names(monkeypatch):
    client = FakeClient(["Luke"])
    monkeypatch.setattr(evaluate, "_client", lambda: client)
    monkeypatch.setattr(evaluate.random, "shuffle", lambda seq: None)

    evaluate._judge("THE_REPLY_TEXT")
    kwargs = client.calls[0]
    user = kwargs["messages"][0]["content"]
    assert "THE_REPLY_TEXT" in user            # judge sees the reply
    assert "SECRET_PROMPT" not in user         # never receives the prompt
    for c in CHARACTERS:                        # system lists all seven names
        assert c in kwargs["system"]


def test_judge_replies_caches_and_requires_replies(monkeypatch):
    # no replies cache yet -> clear error
    with pytest.raises(FileNotFoundError):
        evaluate.judge_replies()

    evaluate.REPLIES_PATH.write_text(
        json.dumps([{"true_character": "Kirk", "prompt": "p", "gen_index": 0, "reply": "hi"}]),
        encoding="utf-8",
    )
    calls = {"n": 0}

    def fake_judge(text):
        calls["n"] += 1
        return "Kirk"

    monkeypatch.setattr(evaluate, "_judge", fake_judge)

    out = evaluate.judge_replies()
    assert out[0]["predicted"] == "Kirk"
    assert calls["n"] == 1
    evaluate.judge_replies()                    # cache hit -> no new judge calls
    assert calls["n"] == 1


# --- step 3: metrics -----------------------------------------------------------

def _write_judgments(rows):
    evaluate.JUDGMENTS_PATH.write_text(json.dumps(rows), encoding="utf-8")


def test_metrics_perfect_judge():
    rows = [{"true_character": c, "predicted": c, "reply": "x"} for c in CHARACTERS]
    _write_judgments(rows)
    m = evaluate.compute_metrics()
    assert m["accuracy"] == 1.0
    assert m["confusion"].shape == (len(CHARACTERS), len(CHARACTERS))
    assert int(m["confusion"].trace()) == len(CHARACTERS)
    assert evaluate.MATRIX_CSV.exists()


def test_metrics_indexing_and_specific_cells():
    # one Lorelai reply judged as Rory, one Rory reply judged as Lorelai
    _write_judgments([
        {"true_character": "Lorelai", "predicted": "Rory", "reply": "x"},
        {"true_character": "Rory", "predicted": "Lorelai", "reply": "x"},
        {"true_character": "Kirk", "predicted": "Kirk", "reply": "x"},
    ])
    m = evaluate.compute_metrics()
    assert m["lorelai_as_rory"] == 1
    assert m["rory_as_lorelai"] == 1
    assert m["most_confused_pair"][:2] == ("Lorelai", "Rory")
    assert m["most_confused_pair"][2] == 2          # symmetric swaps
    assert m["recall"]["Kirk"] == 1.0


def test_metrics_skips_unparsed():
    _write_judgments([
        {"true_character": "Kirk", "predicted": None, "reply": "x"},
        {"true_character": "Kirk", "predicted": "Kirk", "reply": "x"},
    ])
    m = evaluate.compute_metrics()
    assert m["unparsed"] == 1
    assert int(m["confusion"].sum()) == 1          # the None row is excluded


def test_save_heatmap_writes_png():
    rows = [{"true_character": c, "predicted": c, "reply": "x"} for c in CHARACTERS]
    _write_judgments(rows)
    evaluate.save_heatmap(evaluate.compute_metrics())
    assert evaluate.HEATMAP_PNG.exists()
    assert evaluate.HEATMAP_PNG.stat().st_size > 0


# --- single-character refresh --------------------------------------------------

def test_regenerate_character_only_touches_that_character(monkeypatch):
    evaluate.REPLIES_PATH.write_text(json.dumps([
        {"true_character": "Rory", "prompt": "p", "gen_index": 0, "reply": "old rory"},
        {"true_character": "Luke", "prompt": "p", "gen_index": 0, "reply": "old luke"},
    ]), encoding="utf-8")
    seen = []

    def fake_reply(ch, msg, **kwargs):
        seen.append(kwargs)
        return f"new {ch}"

    monkeypatch.setattr(evaluate.generate, "reply", fake_reply)

    out = {r["true_character"]: r["reply"] for r in evaluate.regenerate_character("Rory")}
    assert out["Rory"] == "new Rory"
    assert out["Luke"] == "old luke"          # untouched
    # generation cost params are threaded through
    assert seen == [{"effort": evaluate.EVAL_EFFORT, "max_tokens": evaluate.EVAL_MAX_TOKENS}]


def test_rejudge_character_reuses_other_predictions(monkeypatch):
    evaluate.REPLIES_PATH.write_text(json.dumps([
        {"true_character": "Rory", "prompt": "p", "gen_index": 0, "reply": "rory reply"},
        {"true_character": "Luke", "prompt": "p", "gen_index": 0, "reply": "luke reply"},
    ]), encoding="utf-8")
    evaluate.JUDGMENTS_PATH.write_text(json.dumps([
        {"true_character": "Rory", "prompt": "p", "gen_index": 0, "reply": "rory reply", "predicted": "Lorelai"},
        {"true_character": "Luke", "prompt": "p", "gen_index": 0, "reply": "luke reply", "predicted": "Luke"},
    ]), encoding="utf-8")
    judged = []

    def fake_judge(text):
        judged.append(text)
        return "Rory"

    monkeypatch.setattr(evaluate, "_judge", fake_judge)

    out = {r["true_character"]: r["predicted"] for r in evaluate.rejudge_character("Rory")}
    assert out["Rory"] == "Rory"              # re-judged fresh
    assert out["Luke"] == "Luke"              # reused cached prediction
    assert judged == ["rory reply"]           # judge ran only on Rory's reply
