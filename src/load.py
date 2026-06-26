"""Load transcripts and parse them into per-character line tables.

Skeleton only — no real logic yet.
"""

from __future__ import annotations

import pandas as pd


def load_lines(raw_dir: str) -> pd.DataFrame:
    """Load raw transcripts and return a table of (character, line) rows.

    Args:
        raw_dir: Directory containing raw transcript files.

    Returns:
        A DataFrame with at least 'character' and 'line' columns.
    """
    raise NotImplementedError
