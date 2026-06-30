"""Offline tests for the FastAPI service (preload + generation are stubbed)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import main


@pytest.fixture
def client(monkeypatch):
    """A TestClient with startup warming and reply generation stubbed out."""
    monkeypatch.setattr(main.style, "preload", lambda: None)   # skip heavy boot load
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_chat_returns_reply(client, monkeypatch):
    monkeypatch.setattr(main.generate, "reply", lambda ch, msg: f"{ch} says {msg}")
    r = client.post("/chat", json={"character": "Lorelai", "message": "hi"})
    assert r.status_code == 200
    assert r.json() == {"reply": "Lorelai says hi"}


def test_chat_bad_character_400(client):
    r = client.post("/chat", json={"character": "Spongebob", "message": "hi"})
    assert r.status_code == 400
    assert "Spongebob" in r.json()["detail"]


def test_chat_generation_failure_500(client, monkeypatch):
    def boom(ch, msg):
        raise RuntimeError("anthropic exploded")

    monkeypatch.setattr(main.generate, "reply", boom)
    r = client.post("/chat", json={"character": "Kirk", "message": "hi"})
    assert r.status_code == 500
    assert r.json()["detail"] == "Failed to generate a reply."   # real error not leaked


def test_chat_missing_field_422(client):
    r = client.post("/chat", json={"character": "Kirk"})          # no message
    assert r.status_code == 422
