"""Offline unit tests for src/generate.py.

The collaborators (style card, retrieval, API client) are replaced with fakes,
so prompt assembly is verified deterministically without a key or network.
"""

from __future__ import annotations

import pytest

from src import generate


class _FakeBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.content = [_FakeBlock(text)]


class FakeClient:
    """Records the system + messages it was called with; returns canned text."""

    REPLY_TEXT = "Coffee? Always. Let's go before the world ends."

    def __init__(self) -> None:
        self.kwargs = None
        outer = self

        class _Messages:
            def create(self, **kwargs):
                outer.kwargs = kwargs
                return _FakeMessage(outer.REPLY_TEXT)

        self.messages = _Messages()


_CARD = "CARD: talks fast, pop-culture references, mock-melodrama."
_EXEMPLARS = ["I would die without coffee", "Did you hear about the town meeting"]


@pytest.fixture
def patched(monkeypatch):
    client = FakeClient()
    calls = {}

    def fake_card(character):
        calls["card_character"] = character
        return _CARD

    def fake_retrieve(character, query, k):
        calls["retrieve"] = (character, query, k)
        return list(_EXEMPLARS)

    monkeypatch.setattr(generate.style, "build_style_card", fake_card)
    monkeypatch.setattr(generate.style, "retrieve", fake_retrieve)
    monkeypatch.setattr(generate, "_client", lambda: client)
    return client, calls


def test_reply_returns_client_text(patched):
    client, _calls = patched
    assert generate.reply("Lorelai", "hi") == FakeClient.REPLY_TEXT


def test_system_prompt_carries_card_and_rules(patched):
    client, _calls = patched
    generate.reply("Lorelai", "hi")
    system = client.kwargs["system"]
    assert _CARD in system
    assert "Lorelai" in system
    assert "never copy" in system.lower()          # the verbatim-copy rule


def test_user_message_carries_exemplars_and_message(patched):
    client, _calls = patched
    generate.reply("Lorelai", "Want some coffee?")
    user = client.kwargs["messages"][0]["content"]
    for ex in _EXEMPLARS:
        assert ex in user
    assert "Want some coffee?" in user


def test_collaborators_called_with_expected_args(patched):
    _client, calls = patched
    generate.reply("Kirk", "where is Lulu")
    assert calls["card_character"] == "Kirk"
    assert calls["retrieve"] == ("Kirk", "where is Lulu", generate.RETRIEVE_K)


def test_model_and_token_cap_passed(patched):
    client, _calls = patched
    generate.reply("Lorelai", "hi")
    assert client.kwargs["model"] == generate.REPLY_MODEL
    assert client.kwargs["max_tokens"] == generate.REPLY_MAX_TOKENS
