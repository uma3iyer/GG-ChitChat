"""Load the Gilmore Girls transcript dataset into a clean per-character structure.

The raw data is a single CSV (``data/raw/Gilmore_Girls_Lines.csv``) with columns
``Character``, ``Line``, ``Season``. We keep only single-speaker lines for the
seven target characters and drop empty lines. (The dataset contains no
voice-over lines, so no voice-over filter is needed.)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / "Gilmore_Girls_Lines.csv"
CHARACTERS = ("Lorelai", "Rory", "Luke", "Kirk", "Sookie", "Emily", "Paris")


def load_lines(csv_path: Path | str = RAW_CSV) -> pd.DataFrame:
    """Load the raw CSV; return cleaned single-speaker rows for the target characters.

    Keeps only rows whose ``Character`` is exactly one of ``CHARACTERS`` (an
    exact match excludes combined-speaker rows like ``Lorelai/rory``), trims
    surrounding whitespace, and drops empty/NaN lines.
    """
    df = pd.read_csv(csv_path, index_col=0)
    df = df[df["Character"].isin(CHARACTERS)].copy()
    df["Line"] = df["Line"].astype("string").str.strip()
    df = df[df["Line"].notna() & (df["Line"] != "")]
    return df.reset_index(drop=True)


def lines_by_character(df: pd.DataFrame | None = None) -> dict[str, list[str]]:
    """Return a dict of character -> list of their cleaned lines (in transcript order)."""
    if df is None:
        df = load_lines()
    return {c: df.loc[df["Character"] == c, "Line"].tolist() for c in CHARACTERS}


def main() -> None:
    """Print a line count per character (and total)."""
    by_char = lines_by_character()
    total = 0
    for c in CHARACTERS:
        n = len(by_char[c])
        total += n
        print(f"{c:10s} {n:6d}")
    print(f"{'TOTAL':10s} {total:6d}")


if __name__ == "__main__":
    main()
