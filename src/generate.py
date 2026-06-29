"""Assemble the persona prompt + retrieved exemplars and call the LLM.

``reply(character, message)`` loads the character's cached style card, retrieves
the most relevant real lines, and asks the model to answer *as* the character —
using the card and examples for voice but generating NEW dialogue, never copying
transcript lines verbatim.
"""

from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv

from . import style

REPLY_MODEL = "claude-opus-4-8"
REPLY_MAX_TOKENS = 512
RETRIEVE_K = 8                    # example lines pulled per message

# --- tunable prompt pieces (kept separate on purpose) -------------------------

SYSTEM_PROMPT = """You are {character} from the TV show Gilmore Girls. Reply to \
the user *as* {character}, staying fully in character.

Here is a style card describing {character}'s voice:

{style_card}

Rules:
- Speak in {character}'s voice, matching the style card.
- Generate NEW dialogue. The example lines below are only there to show the
  voice — never copy, quote, or lightly reword them.
- Reply with only what {character} would say out loud: no narration, no stage
  directions, no surrounding quotation marks.
"""

FEW_SHOT_HEADER = (
    "Some real lines {character} has said (for voice reference only — do not reuse them):"
)

USER_TEMPLATE = """{few_shot}

The user says: "{message}"

Reply as {character} with new dialogue:"""


def _assemble_few_shot(character: str, exemplars: list[str]) -> str:
    """Format retrieved exemplars into a labeled reference block."""
    lines = "\n".join(f"- {ex}" for ex in exemplars)
    return f"{FEW_SHOT_HEADER.format(character=character)}\n{lines}"


@lru_cache(maxsize=1)
def _client():
    import anthropic

    load_dotenv()                 # pull ANTHROPIC_API_KEY from .env
    return anthropic.Anthropic()


def reply(
    character: str,
    message: str,
    k: int = RETRIEVE_K,
    *,
    max_tokens: int = REPLY_MAX_TOKENS,
    effort: str | None = None,
) -> str:
    """Generate a NEW in-character reply to ``message`` (never a verbatim line).

    ``max_tokens`` and ``effort`` (e.g. "low") let callers bound cost; ``effort``
    is only sent when provided, so default behavior is unchanged.
    """
    card = style.build_style_card(character)            # cached style card
    exemplars = style.retrieve(character, message, k)   # top-k relevant real lines

    system = SYSTEM_PROMPT.format(character=character, style_card=card)
    user = USER_TEMPLATE.format(
        few_shot=_assemble_few_shot(character, exemplars),
        message=message,
        character=character,
    )
    kwargs: dict = {
        "model": REPLY_MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if effort is not None:
        kwargs["output_config"] = {"effort": effort}
    msg = _client().messages.create(**kwargs)
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def main() -> None:
    character, message = "Lorelai", "Want to grab some coffee?"
    print(f"{character} <- {message!r}\n")
    print(reply(character, message))


if __name__ == "__main__":
    main()
