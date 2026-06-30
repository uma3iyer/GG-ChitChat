"""FastAPI service wrapping the GG ChitChat bot (reuses src.generate / src.style)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src import generate, style
from src.load import CHARACTERS

load_dotenv()                                    # ANTHROPIC_API_KEY from the environment
logger = logging.getLogger("gg_chitchat.api")

ALLOWED_ORIGINS = ["https://uiyer.com", "http://localhost:8000"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    style.preload()                              # startup: embedder + all cards/embeddings, once; loud on missing
    yield


app = FastAPI(title="GG ChitChat API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    character: str
    message: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    if req.character not in CHARACTERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown character {req.character!r}; choose one of {list(CHARACTERS)}.",
        )
    try:
        return {"reply": generate.reply(req.character, req.message)}
    except Exception:
        logger.exception("reply generation failed")   # keep the real error server-side
        raise HTTPException(status_code=500, detail="Failed to generate a reply.")
