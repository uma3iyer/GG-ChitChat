"""Build per-character style cards + an embedded exemplar index.

Skeleton only — no real logic yet.
"""

from __future__ import annotations

import pandas as pd


def build_style_cards(lines: pd.DataFrame) -> pd.DataFrame:
    """Summarize each character's voice into a style card.

    Args:
        lines: Per-character line table from ``load.load_lines``.

    Returns:
        A DataFrame indexed by character with style-card fields.
    """
    raise NotImplementedError


def retrieve(character: str, query: str, k: int) -> list[str]:
    """Retrieve the ``k`` exemplar lines most relevant to ``query``.

    Args:
        character: Which character's exemplars to search.
        query: The user's message to match against.
        k: Number of exemplars to return.

    Returns:
        A list of exemplar lines, most relevant first.
    """
    raise NotImplementedError
