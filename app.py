"""Streamlit chat UI with a character selector.

Skeleton only — no real logic yet.
"""

from __future__ import annotations

import streamlit as st


def main() -> None:
    st.title("GG ChitChat")
    st.caption("Chat with a Gilmore Girls character — new dialogue, in their style.")
    st.selectbox("Character", ["(coming soon)"])
    st.info("Skeleton UI — generation not wired up yet.")


if __name__ == "__main__":
    main()
