"""Style-fidelity eval (3 steps): generate replies -> blind judge -> metrics report.

Each step is independently re-runnable and reads the previous step's cache from
``data/processed/`` (gitignored), so the expensive generation pass is paid once
and judging / metrics can be re-run for free.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from . import generate
from .load import CHARACTERS

# Eval artifacts live in a top-level, committed folder (outside gitignored data/)
# — they are the bot's NEW generated dialogue + metrics, meant for the public repo.
EVAL_DIR = Path(__file__).resolve().parent.parent / "evaluation"
REPLIES_PATH = EVAL_DIR / "eval_replies.json"
JUDGMENTS_PATH = EVAL_DIR / "eval_judgments.json"
MATRIX_CSV = EVAL_DIR / "confusion_matrix.csv"
HEATMAP_PNG = EVAL_DIR / "confusion_matrix.png"

# Replies per (character, prompt): extra resolution on the Lorelai/Rory pair we
# most want to tell apart; one each for the rest. Total = 9 * 30 = 270 replies.
GEN_PER_CHARACTER = {
    "Lorelai": 2,
    "Rory": 2,
    "Luke": 1,
    "Kirk": 1,
    "Sookie": 1,
    "Emily": 1,
    "Paris": 1,
}
EVAL_EFFORT = "low"                 # bound cost/latency on the generation calls
EVAL_MAX_TOKENS = 180
JUDGE_MODEL = "claude-haiku-4-5"   # cheaper than generation; runs 270x
JUDGE_MAX_TOKENS = 32
JUDGE_RETRIES = 1                  # one retry on a malformed answer
RANDOM_BASELINE = 1 / len(CHARACTERS)   # 1/7 ~ 14.3%

EVAL_PROMPTS = [
    # everyday / small talk
    "How's your day going?",
    "What did you get up to this weekend?",
    "Anything interesting happen today?",
    "What are your plans for tonight?",
    "How was your morning?",
    # opinions / universal
    "What do you think about Mondays?",
    "Do you have any pet peeves?",
    "What's your idea of a perfect day?",
    "Do you believe in luck?",
    "Are you a morning person or a night owl?",
    "What's something people get wrong about you?",
    # requests / interaction
    "Can I ask you a favor?",
    "I need some advice.",
    "Can you settle an argument for me?",
    "Tell me a story.",
    "What would you do with a free afternoon?",
    # emotional / sincerity-inviting
    "I'm having a rough day.",
    "I just got some good news!",
    "I'm kind of nervous about something.",
    "I think I made a mistake.",
    "I could use some encouragement.",
    # situational / reactive
    "Someone just cut in front of me in line.",
    "I'm running really late.",
    "I can't decide what to do this weekend.",
    "My plans just fell through.",
    "I have to make a big decision.",
    # open-ended
    "What's the most ridiculous thing that's happened to you lately?",
    "What's been on your mind?",
    "Tell me something good.",
    "What's a small thing that made you happy recently?",
]   # 30 prompts; 270 labeled replies total (2x Lorelai/Rory, 1x the other five)

JUDGE_SYSTEM = (
    "You are identifying which Gilmore Girls character spoke a line of dialogue.\n"
    "Choose exactly one of these characters: {names}.\n"
    "Reply with ONLY the character's name — nothing else."
)
JUDGE_TEMPLATE = 'Line of dialogue:\n"""{reply}"""'


# --- step 1: generate ----------------------------------------------------------

def generate_replies(regenerate: bool = False) -> list[dict]:
    """Generate (or load) 420 labeled replies, cached to eval_replies.json."""
    if REPLIES_PATH.exists() and not regenerate:
        return json.loads(REPLIES_PATH.read_text(encoding="utf-8"))

    rows = [
        {
            "true_character": ch,
            "prompt": prompt,
            "gen_index": gi,
            "reply": generate.reply(
                ch, prompt, effort=EVAL_EFFORT, max_tokens=EVAL_MAX_TOKENS
            ),
        }
        for ch in CHARACTERS
        for prompt in EVAL_PROMPTS
        for gi in range(GEN_PER_CHARACTER[ch])
    ]
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    REPLIES_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


# --- step 2: judge -------------------------------------------------------------

@lru_cache(maxsize=1)
def _client():
    import anthropic

    load_dotenv()
    return anthropic.Anthropic()


def _parse_name(text: str) -> str | None:
    """Return the one character named in ``text``, else None (ambiguous/none)."""
    low = text.strip().lower()
    hits = [c for c in CHARACTERS if c.lower() in low]
    return hits[0] if len(hits) == 1 else None


def _judge(reply_text: str) -> str | None:
    """Blind judge: sees only the reply + the 7 names (freshly shuffled). Retries once."""
    for _ in range(JUDGE_RETRIES + 1):
        names = list(CHARACTERS)
        random.shuffle(names)                       # fresh order per call -> no position bias
        msg = _client().messages.create(
            model=JUDGE_MODEL,
            max_tokens=JUDGE_MAX_TOKENS,
            system=JUDGE_SYSTEM.format(names=", ".join(names)),
            messages=[{"role": "user", "content": JUDGE_TEMPLATE.format(reply=reply_text)}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        name = _parse_name(text)
        if name is not None:
            return name
    return None


def judge_replies(rejudge: bool = False) -> list[dict]:
    """Judge every cached reply, caching results to eval_judgments.json."""
    if JUDGMENTS_PATH.exists() and not rejudge:
        return json.loads(JUDGMENTS_PATH.read_text(encoding="utf-8"))
    if not REPLIES_PATH.exists():
        raise FileNotFoundError("No eval_replies.json — run the generate step first.")

    replies = json.loads(REPLIES_PATH.read_text(encoding="utf-8"))
    judged = [{**r, "predicted": _judge(r["reply"])} for r in replies]

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    JUDGMENTS_PATH.write_text(json.dumps(judged, indent=2, ensure_ascii=False), encoding="utf-8")
    return judged


# --- step 3: metrics + report --------------------------------------------------

def compute_metrics() -> dict:
    """Confusion matrix + accuracy + recall from eval_judgments.json; writes CSV."""
    if not JUDGMENTS_PATH.exists():
        raise FileNotFoundError("No eval_judgments.json — run the judge step first.")
    judged = json.loads(JUDGMENTS_PATH.read_text(encoding="utf-8"))

    idx = {c: i for i, c in enumerate(CHARACTERS)}
    n = len(CHARACTERS)
    conf = np.zeros((n, n), dtype=int)       # rows = true, cols = predicted
    unparsed = 0
    for j in judged:
        p = j["predicted"]
        if p in idx:
            conf[idx[j["true_character"]], idx[p]] += 1
        else:
            unparsed += 1

    total = int(conf.sum())
    accuracy = float(np.trace(conf) / total) if total else 0.0
    recall = {
        c: (float(conf[i, i] / conf[i].sum()) if conf[i].sum() else 0.0)
        for i, c in enumerate(CHARACTERS)
    }
    # most-confused unordered pair by symmetric off-diagonal mass
    pair = max(
        (
            (CHARACTERS[i], CHARACTERS[k], int(conf[i, k] + conf[k, i]))
            for i in range(n)
            for k in range(i + 1, n)
        ),
        key=lambda t: t[2],
    )

    _save_matrix_csv(conf)
    return {
        "characters": list(CHARACTERS),
        "confusion": conf,
        "accuracy": accuracy,
        "recall": recall,
        "unparsed": unparsed,
        "lorelai_as_rory": int(conf[idx["Lorelai"], idx["Rory"]]),
        "rory_as_lorelai": int(conf[idx["Rory"], idx["Lorelai"]]),
        "most_confused_pair": pair,
    }


def _save_matrix_csv(conf: np.ndarray) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with MATRIX_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred", *CHARACTERS])
        for c, row in zip(CHARACTERS, conf):
            w.writerow([c, *row.tolist()])


def save_heatmap(m: dict) -> None:
    """Render the confusion matrix as a heatmap PNG (color = recall, counts annotated)."""
    import matplotlib

    matplotlib.use("Agg")              # no display needed; just write a file
    import matplotlib.pyplot as plt

    chars, conf = m["characters"], m["confusion"]
    n = len(chars)
    row_sums = conf.sum(axis=1, keepdims=True)
    norm = np.divide(
        conf, row_sums, out=np.zeros(conf.shape, dtype=float), where=row_sums != 0
    )   # row-normalize so color = recall, comparable across uneven row counts

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), labels=chars, rotation=45, ha="right")
    ax.set_yticks(range(n), labels=chars)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        f"Style-fidelity confusion — accuracy {m['accuracy']:.0%}  (n={int(conf.sum())})"
    )
    for i in range(n):
        for j in range(n):
            count = int(conf[i, j])
            if count:
                ax.text(
                    j, i, str(count), ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9,
                )
    fig.colorbar(im, ax=ax, label="row-normalized (recall)")
    fig.tight_layout()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(HEATMAP_PNG, dpi=150)
    plt.close(fig)


def print_report(m: dict) -> None:
    chars, conf = m["characters"], m["confusion"]
    print(f"Overall accuracy: {m['accuracy']:.1%}   (random baseline {RANDOM_BASELINE:.1%})")
    if m["unparsed"]:
        print(f"Unparseable judgments excluded: {m['unparsed']}")
    print("\nPer-character recall:")
    for c in chars:
        print(f"  {c:8s} {m['recall'][c]:.1%}")
    print(f"\nLorelai judged as Rory: {m['lorelai_as_rory']}")
    print(f"Rory judged as Lorelai: {m['rory_as_lorelai']}")
    a, b, s = m["most_confused_pair"]
    print(f"Most-confused pair: {a} <-> {b} ({s} swaps)")
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(" " * 9 + "".join(f"{c[:4]:>6}" for c in chars))
    for i, c in enumerate(chars):
        print(f"{c:8s} " + "".join(f"{conf[i, j]:>6}" for j in range(len(chars))))


def evaluate_style_fidelity() -> dict:
    """Run all three steps using caches where present; return the metrics dict."""
    generate_replies()
    judge_replies()
    metrics = compute_metrics()
    save_heatmap(metrics)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Style-fidelity eval (generate / judge / report).")
    ap.add_argument(
        "step", nargs="?", default="all", choices=["generate", "judge", "report", "all"]
    )
    ap.add_argument("--regenerate", action="store_true", help="force-rebuild replies cache")
    ap.add_argument("--rejudge", action="store_true", help="force-rebuild judgments cache")
    args = ap.parse_args()

    if args.step in ("generate", "all"):
        print(f"replies: {len(generate_replies(regenerate=args.regenerate))}")
    if args.step in ("judge", "all"):
        print(f"judgments: {len(judge_replies(rejudge=args.rejudge))}")
    if args.step in ("report", "all"):
        metrics = compute_metrics()
        save_heatmap(metrics)
        print_report(metrics)
        print(f"\nheatmap: {HEATMAP_PNG}")


if __name__ == "__main__":
    main()
