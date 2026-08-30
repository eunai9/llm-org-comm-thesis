"""A 100-message packet for human reading, plus the cheap checks that precede it.

Statistics on generated text answer "how different?" but never "different how?",
and the judge -- itself an LLM -- cannot be the only witness to its own family's
output. So this builds the artefact a person actually reads: a fixed, stratified
sample of 100 stimulus / real-reply / generated-reply triples, in a stable
order, with an empty coding column per item.

**The sample is drawn once and frozen.** Re-sampling per run would mean each
review session codes different messages, and the codes could never be pooled or
re-checked. The seed and the stratification (by direction, so writing up, down
and sideways appear in proportion) are the whole sampling design.

**Automatic flags are screening, not coding.** Each item carries a handful of
mechanical checks -- length, greeting, sign-off, whether a number appears that
was not in the incoming message. They exist to tell the reader where to look
and to make the manual codes auditable against something, not to substitute for
reading. Every one of them is a proxy: "number not in the stimulus" is a
hallucination *suspicion*, and only a person can say whether the reply was
actually wrong.

Two files come out, deliberately: a CSV to code in (one row per item, empty
verdict columns) and a Markdown file to read from, since a spreadsheet cell is
a bad place to read three emails.

Run with ``python -m thesis.analysis.review_pack``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from thesis.analysis.pairs import PAIRS_PATH
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import TABLES_DIR, ensure_dirs

log = get_logger(__name__)

SEED = 20260830
SAMPLE_SIZE = 100

# Verdict columns the human fills in. Left empty on purpose: a pre-filled
# column is a suggestion, and a suggestion is the fastest way to turn manual
# coding into agreement with a heuristic.
CODING_COLUMNS: tuple[str, ...] = (
    "plausible_as_a_reply",
    "addresses_the_request",
    "role_consistent",
    "fabricates_detail",
    "failure_mode",
    "notes",
)

# The taxonomy the coder picks from, printed into the packet so the categories
# are fixed before coding rather than invented while reading.
FAILURE_MODES: tuple[tuple[str, str], ...] = (
    ("ok", "a plausible reply; nothing a colleague would flag"),
    ("too_thin", "an acknowledgment where the message needed substance"),
    ("fabricated_detail", "states a fact, number or commitment not in the incoming message"),
    ("ignores_request", "does not answer what was actually asked"),
    # Added after reading the first items rather than theorised in advance:
    # the model's most distinctive failure is answering a request by restating
    # it back to the person who made it, as though the roles were reversed.
    ("mirrors_request", "restates the sender's own request back at them"),
    ("generic", "could be a reply to almost any email; no purchase on this one"),
    ("wrong_role", "wrong seniority, wrong department, or answers as the wrong person"),
    ("format", "email conventions off: no greeting, no sign-off, wrong subject handling"),
    ("incoherent", "contradicts itself, is truncated mid-thought, or is unreadable"),
)

_GREETING = re.compile(r"^\s*(hi|hello|dear|hey|good\s+(morning|afternoon))\b|^\s*\w+[,:]\s", re.I)
_SIGNOFF = re.compile(
    r"\b(thanks|thank you|regards|best|sincerely|cheers|let me know)\b[\s,.!]*$", re.I
)
_SENTENCE_END = re.compile(r"[.!?]\s*$")
_NUMBER = re.compile(r"\b\d[\d,.]*\b")


def _numbers(text: str) -> set[str]:
    return {n.strip(".,") for n in _NUMBER.findall(text)}


def screen(stimulus: str, generated: str) -> dict[str, object]:
    """Mechanical flags for one generated reply. Proxies, not verdicts."""
    words = generated.split()
    return {
        "generated_words": len(words),
        "has_greeting": bool(_GREETING.search(generated)),
        "has_signoff": bool(_SIGNOFF.search(generated)),
        "ends_mid_sentence": not bool(_SENTENCE_END.search(generated.strip())),
        # A number the reply asserts that its stimulus never mentioned. Often
        # innocuous (a date the persona proposes), sometimes an invented term
        # presented as established fact -- which is exactly the distinction a
        # human coder has to make and this flag cannot.
        "new_numbers": sorted(_numbers(generated) - _numbers(stimulus)),
    }


def draw_sample(frame: pd.DataFrame, *, n: int = SAMPLE_SIZE, seed: int = SEED) -> pd.DataFrame:
    """Stratified sample of ``n`` pairs, proportional by writing direction."""
    if len(frame) <= n:
        return frame.reset_index(drop=True)
    sample, _ = train_test_split(
        frame, train_size=n, random_state=seed, stratify=frame["direction"]
    )
    return sample.sort_values("thread_id").reset_index(drop=True)


def build_packet(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per review item: the texts, the screening flags, empty verdicts."""
    rows: list[dict[str, object]] = []
    for position, item in enumerate(frame.itertuples(), start=1):
        flags = screen(item.stimulus_text, item.generated_reply)
        rows.append(
            {
                "item": position,
                "thread_id": item.thread_id,
                "persona_id": item.persona_id,
                "direction": item.direction,
                "decision": item.decision,
                "confidence": item.confidence,
                **flags,
                "real_words": len(str(item.real_reply_body_recleaned).split()),
                "stimulus_text": item.stimulus_text,
                "real_reply": item.real_reply_body_recleaned,
                "generated_subject": item.generated_subject,
                "generated_reply": item.generated_reply,
                "reasoning_brief": item.reasoning_brief,
                **{column: "" for column in CODING_COLUMNS},
            }
        )
    return pd.DataFrame.from_records(rows)


def _truncate(text: str, n_words: int) -> str:
    words = str(text).split()
    if len(words) <= n_words:
        return " ".join(words)
    return " ".join(words[:n_words]) + " […]"


def render_markdown(packet: pd.DataFrame, *, stimulus_words: int = 90) -> str:
    """The readable version: one section per item, ending in a blank verdict line."""
    lines: list[str] = [
        "# Manual review packet: 100 generated replies",
        "",
        f"Stratified sample of {len(packet)} of the matched real-vs-generated pairs, "
        f"drawn with seed {SEED}. Incoming messages are truncated to "
        f"{stimulus_words} words for reading; the full text is in the CSV.",
        "",
        "Code each item in `manual_review_sheet.csv`. Failure modes:",
        "",
        *[f"- `{name}` — {description}" for name, description in FAILURE_MODES],
        "",
        "---",
        "",
    ]
    for item in packet.itertuples():
        flags = []
        if not item.has_greeting:
            flags.append("no greeting")
        if not item.has_signoff:
            flags.append("no sign-off")
        if item.ends_mid_sentence:
            flags.append("ends mid-sentence")
        if item.new_numbers:
            flags.append(f"numbers not in stimulus: {', '.join(item.new_numbers)}")
        lines += [
            f"## Item {item.item} — {item.thread_id}",
            "",
            f"*{item.persona_id}, writing {item.direction}; declared decision: "
            f"{item.decision} ({item.confidence}). "
            f"{item.generated_words} words generated vs {item.real_words} real.*",
            "",
            f"**Screening flags:** {'; '.join(flags) if flags else 'none'}",
            "",
            "**Incoming message**",
            "",
            "> " + _truncate(item.stimulus_text, stimulus_words).replace("\n", " "),
            "",
            "**Real reply**",
            "",
            "> " + _truncate(item.real_reply, 120).replace("\n", " "),
            "",
            f"**Generated reply** (subject: {item.generated_subject})",
            "",
            "> " + str(item.generated_reply).replace("\n", " "),
            "",
            f"*Model's stated reasoning:* {item.reasoning_brief}",
            "",
            "**Verdict:** ______  **Failure mode:** ______",
            "",
            "---",
            "",
        ]
    return "\n".join(lines)


def summarize_flags(packet: pd.DataFrame) -> dict[str, object]:
    """Counts of the mechanical flags, for the write-up to quote."""
    return {
        "n_items": len(packet),
        "median_generated_words": int(packet["generated_words"].median()),
        "median_real_words": int(packet["real_words"].median()),
        "share_no_greeting": round(float((~packet["has_greeting"]).mean()), 3),
        "share_no_signoff": round(float((~packet["has_signoff"]).mean()), 3),
        "share_ends_mid_sentence": round(float(packet["ends_mid_sentence"].mean()), 3),
        "share_with_new_numbers": round(
            float(packet["new_numbers"].apply(lambda n: len(n) > 0).mean()), 3
        ),
        "by_direction": {
            str(k): int(v) for k, v in packet["direction"].value_counts().to_dict().items()
        },
    }


def write_packet(packet: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """Write the coding sheet and the readable packet; return both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "manual_review_sheet.csv"
    md_path = out_dir / "manual_review_packet.md"
    csv_frame = packet.copy()
    csv_frame["new_numbers"] = csv_frame["new_numbers"].apply(lambda n: " ".join(n))
    csv_frame.to_csv(csv_path, index=False)
    md_path.write_text(render_markdown(packet), encoding="utf-8")
    return csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=str(PAIRS_PATH))
    parser.add_argument("--n", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--out-dir", default=str(TABLES_DIR))
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    frame: pd.DataFrame = pd.read_parquet(args.pairs)
    packet = build_packet(draw_sample(frame, n=args.n))
    csv_path, md_path = write_packet(packet, Path(args.out_dir))
    log.info("wrote %s and %s", csv_path, md_path)
    log.info("screening summary: %s", json.dumps(summarize_flags(packet)))


if __name__ == "__main__":
    main()
