"""Turn a coded review sheet into the counts and figures the write-up quotes.

Separate from :mod:`thesis.analysis.review_pack` on purpose: that module builds
the blank sheet, this one reports what came back on it. Keeping them apart means
a second coder's sheet can be summarised by exactly the same code, which is the
precondition for reporting agreement between two coders rather than one
person's numbers.

Two figures, because the counts answer two different questions. Which failures
occur, and how often -- one bar per category. And whether the dominant failure
depends on who the persona is writing to, which is the one place the review
touches a research question rather than a validity check: if mirroring a
request were really about deference, it should be commonest writing upward.

Run with ``python -m thesis.analysis.review_summary``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from thesis.analysis.plots import plot_category_counts
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import DOCS_FIGURES_DIR, MANIFESTS_DIR, TABLES_DIR, ensure_dirs

log = get_logger(__name__)

CODED_PATH: Path = TABLES_DIR / "manual_review_coded_first_pass.csv"

DIRECTION_LABELS: dict[str, str] = {
    "down": "writing down",
    "lateral": "writing to a peer",
    "up": "writing up",
}


def failure_counts(coded: pd.DataFrame) -> pd.Series:
    """Counts per failure mode, "ok" first and the rest by descending frequency."""
    counts = coded["failure_mode"].value_counts()
    ordered = ["ok", *[mode for mode in counts.index if mode != "ok"]]
    return counts.reindex([mode for mode in ordered if mode in counts.index])


def mirrored_rate_by_direction(coded: pd.DataFrame) -> pd.DataFrame:
    """Share of items coded as mirroring the request, per writing direction."""
    frame = coded.assign(mirrored=coded["failure_mode"] == "mirrors_request")
    grouped = frame.groupby("direction")["mirrored"].agg(["mean", "size"])
    return grouped.rename(columns={"mean": "rate", "size": "n"}).sort_values("rate")


def summarize(coded: pd.DataFrame) -> dict[str, object]:
    """Everything the write-up quotes from the coded sheet."""
    counts = failure_counts(coded)
    by_direction = mirrored_rate_by_direction(coded)
    return {
        "n_items": len(coded),
        "n_distinct_generated_replies": int(coded["generated_reply"].nunique()),
        "failure_modes": {str(k): int(v) for k, v in counts.items()},
        "share_ok": round(float((coded["failure_mode"] == "ok").mean()), 3),
        "share_plausible": round(float((coded["plausible_as_a_reply"] == "y").mean()), 3),
        "share_addresses_request": round(float((coded["addresses_the_request"] == "y").mean()), 3),
        "share_fabricates_detail": round(float((coded["fabricates_detail"] == "y").mean()), 3),
        "mirrored_rate_by_direction": {
            str(index): {"rate": round(float(row.rate), 3), "n": int(row.n)}
            for index, row in by_direction.iterrows()
        },
    }


def plot(coded: pd.DataFrame) -> list[Path]:
    """Write both figures; return their paths."""
    counts = failure_counts(coded)
    paths = [
        plot_category_counts(
            [str(index) for index in counts.index],
            [int(value) for value in counts],
            DOCS_FIGURES_DIR / "review_failure_modes.png",
            title="What 100 generated replies get wrong, read against the message they answer",
            subtitle=(
                "One primary code per reply, assigned by reading. Mirroring the sender's "
                "own request is the dominant failure."
            ),
            x_label="replies",
            highlight=("mirrors_request",),
        )
    ]

    by_direction = mirrored_rate_by_direction(coded)
    paths.append(
        plot_category_counts(
            [
                f"{DIRECTION_LABELS.get(str(index), str(index))} (n={int(row.n)})"
                for index, row in by_direction.iterrows()
            ],
            [round(float(row.rate) * 100) for _, row in by_direction.iterrows()],
            DOCS_FIGURES_DIR / "review_mirroring_by_direction.png",
            title="Mirroring the request, by who the persona is writing to",
            subtitle=(
                "Share of replies coded as restating the sender's own request. "
                "Small cells; read as a direction to check, not a result."
            ),
            x_label="% of replies",
        )
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coded", default=str(CODED_PATH))
    parser.add_argument("--out", default=str(MANIFESTS_DIR / "manual_review.json"))
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    coded = pd.read_csv(args.coded)
    summary = summarize(coded)
    for path in plot(coded):
        log.info("wrote %s", path)
    Path(args.out).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    log.info("manual review summary: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
