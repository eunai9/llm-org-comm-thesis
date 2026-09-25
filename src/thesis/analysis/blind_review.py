"""A blind coding page: two models' replies to the same emails, model hidden.

The section 35 codes are Claude's first pass, not a person's. This builds the
material for a person to code. Each email appears once, with two replies to
it, one from each model. Which reply came from which model is random per email
and hidden. The key goes to a separate file, so the page cannot reveal it.

The page is one HTML file. It runs in a browser with no server and no network.
Codes are kept in the browser while coding and exported as a CSV at the end.

Both output files contain Enron text, so they are written to the tables
folder, which git ignores.

Run with ``python -m thesis.analysis.blind_review``.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from thesis.analysis.mirroring import pair_key
from thesis.analysis.review_pack import CODING_COLUMNS, FAILURE_MODES, SEED, draw_sample
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import INTERIM_DIR, TABLES_DIR, ensure_dirs

log = get_logger(__name__)

FIRST_PAIRS: Path = INTERIM_DIR / "real_vs_generated_pairs_act.parquet"
SECOND_PAIRS: Path = INTERIM_DIR / "pairs_deepseek.parquet"
SAMPLE_SIZE = 50
SLOTS: tuple[str, str] = ("A", "B")

# The rank labels of thesis.data.roles. Rank 1 is the most junior.
RANK_LABELS: dict[int, str] = {
    1: "Employee",
    2: "Manager",
    3: "Director",
    4: "Vice President",
    5: "Managing Director",
    6: "President/CEO",
}

# Direction is seen from the persona, as in thesis.sim.real_stimuli.
DIRECTION_TEXT: dict[str, str] = {
    "up": "writing up to someone more senior",
    "down": "writing down to someone more junior",
    "lateral": "writing to a peer of the same rank",
}

# The yes/no columns of the coding sheet, each asked as a plain question.
YES_NO_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("plausible_as_a_reply", "Could a real colleague have sent this reply?"),
    ("addresses_the_request", "Does it answer what the email actually asks?"),
    ("role_consistent", "Does it fit the writer's rank and department?"),
    ("fabricates_detail", "Does it state a fact, number or commitment that is not in the email?"),
)


def prompt_key(frame: pd.DataFrame) -> pd.Series:
    """Persona plus incoming email. Pairs with the same key got the same prompt."""
    return (
        frame["persona_id"] + "|" + frame["stimulus_text"].map(lambda t: " ".join(str(t).split()))
    )


def draw_blind_sample(
    first: pd.DataFrame, second: pd.DataFrame, *, n: int = SAMPLE_SIZE, seed: int = SEED
) -> pd.DataFrame:
    """One row per sampled email, with both models' replies to it.

    Pairs that share a prompt share one generated reply, so only the first of
    them is kept. Otherwise the coder would code the same reply twice.
    """
    first = first.assign(pair_key=pair_key(first["cell_id"]), prompt_key=prompt_key(first))
    first = first.drop_duplicates("prompt_key")
    second = second.assign(pair_key=pair_key(second["cell_id"]))
    sample = draw_sample(first, n=n, seed=seed)
    merged = sample.merge(
        second[["pair_key", "generated_reply"]],
        on="pair_key",
        how="inner",
        suffixes=("_first", "_second"),
    )
    if len(merged) != len(sample):
        msg = f"{len(sample) - len(merged)} sampled emails have no reply from the second model"
        raise ValueError(msg)
    return merged.assign(email=range(1, len(merged) + 1))


def assign_slots(
    sample: pd.DataFrame, names: tuple[str, str], *, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Put the two replies of each email in slots A and B, in random order.

    Exactly half of the emails have the second model in slot A. Returns the
    items to show and the key that says which model wrote each item.
    """
    rng = np.random.default_rng(seed)
    swapped = rng.permutation(len(sample)) < len(sample) // 2
    items: list[dict[str, object]] = []
    key: list[dict[str, object]] = []
    for row, swap in zip(sample.itertuples(index=False), swapped, strict=True):
        replies = [(names[0], row.generated_reply_first), (names[1], row.generated_reply_second)]
        if swap:
            replies.reverse()
        for slot, (model, reply) in zip(SLOTS, replies, strict=True):
            items.append({"email": row.email, "slot": slot, "reply": reply})
            key.append(
                {
                    "email": row.email,
                    "slot": slot,
                    "model": model,
                    "pair_key": row.pair_key,
                    "thread_id": row.thread_id,
                }
            )
    return pd.DataFrame(items), pd.DataFrame(key)


def _writer(rank: int, department: str) -> str:
    label = RANK_LABELS.get(rank, f"rank {rank}")
    article = "an" if label[0] in "AEIOU" else "a"
    return f"{article} {label} in {department} (seniority rank {rank} of 6)"


def build_page_data(sample: pd.DataFrame, items: pd.DataFrame) -> dict[str, object]:
    """Everything the page shows. It holds no model names and no cell ids."""
    replies = {(row.email, row.slot): str(row.reply) for row in items.itertuples(index=False)}
    emails = [
        {
            "email": int(row.email),
            "writer": _writer(int(row.seniority_rank), str(row.department)),
            "direction": DIRECTION_TEXT[str(row.direction)],
            "incoming": str(row.stimulus_text),
            "real_reply": str(row.real_reply_body_recleaned),
            "replies": {slot: replies[(row.email, slot)] for slot in SLOTS},
        }
        for row in sample.itertuples(index=False)
    ]
    return {
        "emails": emails,
        "slots": list(SLOTS),
        "questions": [list(question) for question in YES_NO_QUESTIONS],
        "codebook": [list(mode) for mode in FAILURE_MODES],
        "columns": ["email", "slot", *CODING_COLUMNS, "coder"],
    }


def render_page(data: dict[str, object], *, title: str = "Blind reply coding") -> str:
    """The self-contained HTML page, with the data embedded as JSON.

    ``</`` is escaped inside the JSON, so an email that contains
    ``</script>`` cannot end the data block early.
    """
    payload = json.dumps(data).replace("</", "<\\/")
    storage_key = "blind-review-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return (
        _PAGE.replace("__TITLE__", html.escape(title))
        .replace("__STORAGE_KEY__", storage_key)
        .replace("__DATA__", payload)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", default=str(FIRST_PAIRS))
    parser.add_argument("--second", default=str(SECOND_PAIRS))
    parser.add_argument("--n", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", default=str(TABLES_DIR))
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    first, second = pd.read_parquet(args.first), pd.read_parquet(args.second)
    sample = draw_blind_sample(first, second, n=args.n, seed=args.seed)
    names = (str(first["model"].iloc[0]), str(second["model"].iloc[0]))
    items, key = assign_slots(sample, names, seed=args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    page_path = out_dir / "blind_review_page.html"
    key_path = out_dir / "blind_review_key.csv"
    page_path.write_text(render_page(build_page_data(sample, items)), encoding="utf-8")
    key.to_csv(key_path, index=False)
    log.info(
        "wrote %s (%d emails, %d replies) and the key %s",
        page_path,
        len(sample),
        len(items),
        key_path,
    )


_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --ink:#1f2328; --muted:#59636e; --line:#d1d9e0; --bg:#f6f8fa; --card:#fff; --done:#1a7f37; --warn:#9a6700; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }
  header { position: sticky; top: 0; z-index: 1; background: var(--card); border-bottom: 1px solid var(--line);
           padding: 10px 20px; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
  header h1 { font-size: 16px; margin: 0 12px 0 0; }
  main { max-width: 1200px; margin: 0 auto; padding: 16px 20px 60px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; margin-bottom: 16px; }
  .text { white-space: pre-wrap; overflow-wrap: anywhere; background: var(--bg); border-radius: 6px; padding: 10px 12px; }
  .muted { color: var(--muted); }
  .done { color: var(--done); font-weight: 600; }
  .warn { color: var(--warn); }
  .replies { display: grid; grid-template-columns: 1fr; gap: 16px; }
  @media (min-width: 900px) { .replies { grid-template-columns: 1fr 1fr; } }
  fieldset { border: 1px solid var(--line); border-radius: 6px; margin: 10px 0; padding: 6px 12px 8px; }
  legend { font-weight: 600; padding: 0 4px; }
  .choice { display: block; padding: 2px 0; cursor: pointer; }
  .choice code { font-weight: 600; }
  textarea { width: 100%; min-height: 60px; font: inherit; }
  button, select, input[type=text] { font: inherit; padding: 5px 10px; }
  nav { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  dt { font-weight: 600; margin-top: 6px; }
  dd { margin: 0 0 4px 16px; }
</style>
</head>
<body>
<header>
  <h1>Blind reply coding</h1>
  <label>Your name <input type="text" id="coder" size="14"></label>
  <span id="progress" class="muted"></span>
  <button id="export">Export CSV</button>
  <span id="storage-warning" class="warn" hidden>This browser is not saving your codes. Export often.</span>
</header>
<main>
  <details class="card" open>
    <summary><strong>How to code</strong></summary>
    <ol>
      <li>Read the incoming email and who is replying to it.</li>
      <li>The real reply is there for reference. It is not the answer key.</li>
      <li>Each email has two replies, A and B. Two different AI models wrote them, in random order. Judge each reply on its own.</li>
      <li>For each reply, answer the four questions and pick one main failure mode. Pick <code>ok</code> if nothing is wrong.</li>
      <li>If more than one failure mode fits, pick the most serious one and name the others in the notes.</li>
      <li>Your codes are saved in this browser as you go. Press Export CSV when you finish, and now and then as a backup.</li>
    </ol>
  </details>
  <nav class="card">
    <button id="prev">&larr; Previous</button>
    <select id="jump"></select>
    <button id="next">Next &rarr;</button>
  </nav>
  <section class="card" id="email"></section>
  <div class="replies" id="replies"></div>
  <details class="card">
    <summary><strong>Codebook</strong></summary>
    <dl id="codebook"></dl>
  </details>
</main>
<script type="application/json" id="data">__DATA__</script>
<script>
(function () {
  const DATA = JSON.parse(document.getElementById("data").textContent);
  const KEY = "__STORAGE_KEY__";
  const emails = DATA.emails;
  const total = emails.length * DATA.slots.length;
  const state = load();
  if (!(state.at >= 0 && state.at < emails.length)) state.at = 0;

  function load() {
    try {
      const saved = JSON.parse(localStorage.getItem(KEY));
      if (saved && saved.codes) return saved;
    } catch (e) {}
    return { coder: "", at: 0, codes: {} };
  }
  function save() {
    try { localStorage.setItem(KEY, JSON.stringify(state)); }
    catch (e) { document.getElementById("storage-warning").hidden = false; }
  }
  function codesFor(email, slot) {
    const id = email + slot;
    if (!state.codes[id]) state.codes[id] = {};
    return state.codes[id];
  }
  function complete(email, slot) {
    const c = state.codes[email + slot] || {};
    return DATA.questions.every(q => c[q[0]]) && Boolean(c.failure_mode);
  }
  function countDone() {
    let n = 0;
    emails.forEach(e => DATA.slots.forEach(s => { if (complete(e.email, s)) n += 1; }));
    return n;
  }
  function el(tag, attrs, text) {
    const node = document.createElement(tag);
    Object.entries(attrs || {}).forEach(([k, v]) => node.setAttribute(k, v));
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function radio(name, value, checked, labelNode, onPick) {
    const label = el("label", { class: "choice" });
    const input = el("input", { type: "radio", name: name, value: value });
    input.checked = checked;
    input.addEventListener("change", () => onPick(value));
    label.append(input, " ", labelNode);
    return label;
  }
  function set(email, slot, column, value) {
    codesFor(email, slot)[column] = value;
    save();
    refreshProgress();
  }
  function refreshProgress() {
    const done = countDone();
    const progress = document.getElementById("progress");
    progress.textContent = "Coded " + done + " of " + total + " replies";
    progress.className = done === total ? "done" : "muted";
    const jump = document.getElementById("jump");
    jump.innerHTML = "";
    emails.forEach((e, i) => {
      const both = DATA.slots.every(s => complete(e.email, s));
      const option = el("option", { value: String(i) }, "Email " + e.email + " of " + emails.length + (both ? "  ✓" : ""));
      option.selected = i === state.at;
      jump.append(option);
    });
  }
  function renderEmail() {
    const e = emails[state.at];
    const box = document.getElementById("email");
    box.innerHTML = "";
    box.append(
      el("h2", {}, "Email " + e.email + " of " + emails.length),
      el("p", {}, "The replies are written by " + e.writer + ", " + e.direction + "."),
      el("h3", {}, "Incoming email"),
      el("div", { class: "text" }, e.incoming)
    );
    const real = el("details", {});
    real.append(el("summary", {}, "The real reply (for reference only)"), el("div", { class: "text" }, e.real_reply));
    box.append(real);

    const replies = document.getElementById("replies");
    replies.innerHTML = "";
    DATA.slots.forEach(slot => {
      const c = codesFor(e.email, slot);
      const card = el("section", { class: "card" });
      card.append(el("h3", {}, "Reply " + slot), el("div", { class: "text" }, e.replies[slot]));
      DATA.questions.forEach(([column, question]) => {
        const fieldset = el("fieldset");
        fieldset.append(el("legend", {}, question));
        ["yes", "no"].forEach(v => fieldset.append(
          radio(e.email + slot + column, v, c[column] === v, document.createTextNode(v),
                pick => set(e.email, slot, column, pick))
        ));
        card.append(fieldset);
      });
      const modes = el("fieldset");
      modes.append(el("legend", {}, "Main failure mode (pick one)"));
      DATA.codebook.forEach(([name, description]) => {
        const text = el("span");
        text.append(el("code", {}, name), ": " + description);
        modes.append(radio(e.email + slot + "mode", name, c.failure_mode === name, text,
                           pick => set(e.email, slot, "failure_mode", pick)));
      });
      card.append(modes);
      const notes = el("textarea", { placeholder: "Notes (optional)" });
      notes.value = c.notes || "";
      notes.addEventListener("input", () => set(e.email, slot, "notes", notes.value));
      card.append(el("strong", {}, "Notes"), notes);
      replies.append(card);
    });
    refreshProgress();
    window.scrollTo(0, 0);
  }
  function go(i) {
    state.at = Math.max(0, Math.min(emails.length - 1, i));
    save();
    renderEmail();
  }
  function csvCell(v) {
    return '"' + String(v === undefined || v === null ? "" : v).replace(/"/g, '""') + '"';
  }

  document.getElementById("prev").addEventListener("click", () => go(state.at - 1));
  document.getElementById("next").addEventListener("click", () => go(state.at + 1));
  document.getElementById("jump").addEventListener("change", ev => go(Number(ev.target.value)));
  const coder = document.getElementById("coder");
  coder.value = state.coder || "";
  coder.addEventListener("input", () => { state.coder = coder.value; save(); });
  const codebook = document.getElementById("codebook");
  DATA.codebook.forEach(([name, description]) => codebook.append(el("dt", {}, name), el("dd", {}, description)));

  document.getElementById("export").addEventListener("click", () => {
    if (!(state.coder || "").trim()) { alert("Please enter your name first."); coder.focus(); return; }
    const missing = total - countDone();
    if (missing > 0 && !confirm(missing + " replies are not fully coded yet. Export anyway?")) return;
    const lines = [DATA.columns.map(csvCell).join(",")];
    emails.forEach(e => DATA.slots.forEach(slot => {
      const row = Object.assign({ email: e.email, slot: slot, coder: state.coder }, state.codes[e.email + slot] || {});
      lines.push(DATA.columns.map(column => csvCell(row[column])).join(","));
    }));
    const blob = new Blob([lines.join("\r\n") + "\r\n"], { type: "text/csv" });
    const link = el("a", { href: URL.createObjectURL(blob), download: "blind_review_codes.csv" });
    document.body.append(link);
    link.click();
    link.remove();
  });

  renderEmail();
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
