"""A blind human-coding page for validation check 3 of the Q1 role-inference
redesign: can a person tell, from a reply alone, who it was written to?

This is the true ceiling the judge is measured against, not a secondary
check. If a person cannot tell direction from a reply, no judge can, and
the honest finding is that the replies don't encode direction -- a result
that would not depend on any model or judge choice.

The page follows blind_review.py's self-contained-HTML pattern (no server,
codes kept in the browser, exported as CSV), but asks a different question
over different data: one reply, not two, and a three-way "who was this
written to" choice rather than a failure-mode picklist. That difference in
shape is why this is its own module rather than an extension of
blind_review.py, whose page compares two models' replies for a different
purpose (Q2 mirroring review, not Q1 direction).

Run with ``python -m thesis.analysis.role_coding``.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import pandas as pd

from thesis.analysis.blinding import strip_identity
from thesis.analysis.role_inference import GENERATING_GRIDS
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import TABLES_DIR, ensure_dirs

log = get_logger(__name__)

SAMPLE_SIZE = 180
SEED = 20260923
CHOICES: tuple[str, ...] = ("up", "lateral", "down")
CHOICE_TEXT: dict[str, str] = {
    "up": "written to someone more senior than the writer",
    "lateral": "written to a peer",
    "down": "written to someone more junior than the writer",
}


def draw_stratified_sample(
    grids: dict[str, Path], *, n: int = SAMPLE_SIZE, seed: int = SEED
) -> pd.DataFrame:
    """``n`` replies at replicate 1, stratified by model and true direction
    as evenly as ``n`` allows, identity-stripped. Frozen by the seed, the
    same reason ``review_pack.draw_sample`` freezes its own sample -- codes
    from different sessions must be pooled against the same items.
    """
    frames = []
    for model, path in grids.items():
        frame = pd.read_parquet(path)
        frame = frame[frame["replicate"] == 1].assign(model=model)
        frames.append(frame[["cell_id", "direction", "body", "model"]])
    pool = pd.concat(frames, ignore_index=True)

    per_stratum = max(1, n // (pool["model"].nunique() * len(CHOICES)))
    parts = [
        group.sample(n=min(per_stratum, len(group)), random_state=seed)
        for _, group in pool.groupby(["model", "direction"])
    ]
    sample = pd.concat(parts, ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True).head(n)
    return pd.DataFrame(
        {
            "item": range(1, len(sample) + 1),
            "model": sample["model"],
            "true_direction": sample["direction"],
            "text": sample["body"].map(strip_identity),
        }
    )


def render_coding_page(sample: pd.DataFrame, *, title: str = "Q1 role-inference coding") -> str:
    """A self-contained HTML page. Model and true_direction are never
    embedded in the page data -- only ``item`` and ``text`` are, so the
    rendered page cannot leak the answer key any more than the judge's own
    prompt does. The three CHOICES words (up/lateral/down) do appear, since
    they are the multiple-choice options every item is judged against, not
    an answer -- the same way the judge's own schema enum is visible to the
    judge without leaking which one is correct for a given item.
    """
    items = [
        {"item": int(row.item), "text": str(row.text)} for row in sample.itertuples(index=False)
    ]
    payload = json.dumps(
        {"items": items, "choices": list(CHOICES), "choice_text": CHOICE_TEXT}
    ).replace("</", "<\\/")
    storage_key = "role-coding-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return (
        _PAGE.replace("__TITLE__", html.escape(title))
        .replace("__STORAGE_KEY__", storage_key)
        .replace("__DATA__", payload)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", default=str(TABLES_DIR))
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    sample = draw_stratified_sample(GENERATING_GRIDS, n=args.n, seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    page_path = out_dir / "role_coding_page.html"
    key_path = out_dir / "role_coding_key.csv"
    page_path.write_text(render_coding_page(sample), encoding="utf-8")
    sample[["item", "model", "true_direction"]].to_csv(key_path, index=False)
    log.info("wrote %s (%d items) and the key %s", page_path, len(sample), key_path)


_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --ink:#1f2328; --muted:#59636e; --line:#d1d9e0; --bg:#f6f8fa; --card:#fff; --done:#1a7f37; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }
  header { position: sticky; top: 0; z-index: 1; background: var(--card); border-bottom: 1px solid var(--line);
           padding: 10px 20px; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
  header h1 { font-size: 16px; margin: 0 12px 0 0; }
  main { max-width: 760px; margin: 0 auto; padding: 16px 20px 60px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; margin-bottom: 16px; }
  .text { white-space: pre-wrap; overflow-wrap: anywhere; background: var(--bg); border-radius: 6px; padding: 10px 12px; }
  .muted { color: var(--muted); }
  .done { color: var(--done); font-weight: 600; }
  fieldset { border: 1px solid var(--line); border-radius: 6px; margin: 10px 0; padding: 6px 12px 8px; }
  .choice { display: block; padding: 4px 0; cursor: pointer; }
  button, select, input[type=text] { font: inherit; padding: 5px 10px; }
  nav { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
</style>
</head>
<body>
<header>
  <h1>Who was this written to?</h1>
  <label>Your name <input type="text" id="coder" size="14"></label>
  <span id="progress" class="muted"></span>
  <button id="export">Export CSV</button>
</header>
<main>
  <details class="card" open>
    <summary><strong>How to code</strong></summary>
    <ol>
      <li>Read the reply. It is one side of a workplace email exchange; you are not shown what it answers.</li>
      <li>Guess who it was written to: someone more senior than the writer, a peer, or someone more junior.</li>
      <li>Judge only from how it is written. There is no right answer visible anywhere on this page.</li>
      <li>Your codes are saved in this browser as you go. Press Export CSV when you finish.</li>
    </ol>
  </details>
  <nav class="card">
    <button id="prev">&larr; Previous</button>
    <select id="jump"></select>
    <button id="next">Next &rarr;</button>
  </nav>
  <section class="card" id="item"></section>
</main>
<script>
const DATA = __DATA__;
const STORAGE_KEY = "__STORAGE_KEY__";
let codes = {};
try { codes = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); } catch (e) { codes = {}; }
let current = 0;

function save() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(codes)); } catch (e) { /* private window etc. */ }
}

function render() {
  const item = DATA.items[current];
  const chosen = codes[item.item] || {};
  const choices = DATA.choices.map(c => `
    <label class="choice">
      <input type="radio" name="answer" value="${c}" ${chosen.answer === c ? "checked" : ""}>
      ${DATA.choice_text[c]}
    </label>`).join("");
  document.getElementById("item").innerHTML = `
    <p class="muted">Item ${item.item} of ${DATA.items.length}</p>
    <div class="text">${item.text.replace(/</g, "&lt;")}</div>
    <fieldset><legend>Who was this written to?</legend>${choices}</fieldset>
  `;
  document.querySelectorAll('input[name="answer"]').forEach(el => {
    el.addEventListener("change", () => {
      codes[item.item] = { answer: el.value, coder: document.getElementById("coder").value };
      save();
      updateProgress();
    });
  });
  const jump = document.getElementById("jump");
  jump.innerHTML = DATA.items.map((it, i) => `<option value="${i}" ${i === current ? "selected" : ""}>${it.item}${codes[it.item] ? " done" : ""}</option>`).join("");
  updateProgress();
}

function updateProgress() {
  const done = Object.keys(codes).length;
  document.getElementById("progress").textContent = `${done} / ${DATA.items.length} coded`;
}

document.getElementById("prev").addEventListener("click", () => { current = Math.max(0, current - 1); render(); });
document.getElementById("next").addEventListener("click", () => { current = Math.min(DATA.items.length - 1, current + 1); render(); });
document.getElementById("jump").addEventListener("change", (e) => { current = Number(e.target.value); render(); });
document.getElementById("export").addEventListener("click", () => {
  const rows = [["item", "answer", "coder"]];
  for (const item of DATA.items) {
    const c = codes[item.item] || {};
    rows.push([item.item, c.answer || "", c.coder || ""]);
  }
  const csv = rows.map(r => r.map(v => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "role_coding_codes.csv";
  a.click();
});

render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
