# General handoff, 2026-09-23 (updated)

For a fresh agent picking up this thesis. Read this first, then `HANDOVER.md`.

This file is deliberately short. It records what changed recently and what to
do next. It does not repeat the thesis background, the dataset, the models, or
the judging criteria. Those live elsewhere.

| Where | What it holds |
|---|---|
| `HANDOVER.md` (repo root, gitignored, local only) | Full context: thesis, data, simulator, models, judging criteria, the three big problems, working conventions, operational traps. Start here after this file. |
| `PROGRESS.md` | Main log, 54 numbered sections in date order. Every result and every failed attempt. |
| `PROGRESS_llms.md` | Four-model comparison by topic, plus the newest Q1 run. |
| `PROGRESS_nvidia.md` | Free-tier build log. How the NVIDIA and Groq clients came about. |

This file replaces an earlier same-day version. That one is stale in two
places, corrected below: it said Q1's gap to real email was settled at
p≈.047, and it listed `grid_draw_reliability` as an open, unfixed blocker.
Both are resolved; see sections 1 and 2.

---

## 1. Q1 now reaches significance. The gap to real email is not settled.

A third draw was generated overnight on Sep 23. The writing-down effect on
Llama 3.2 3B (main run, local, full design) is now significant:

| Sample | Coefficient | p |
|---|---:|---:|
| 240 replies (`PROGRESS.md` s39) | +0.163 | .401 |
| 1,440 replies (s51) | +0.134 | .092 |
| 2,880 replies, 2 draws (s54) | +0.092 | .130 |
| **4,320 replies, 3 draws** | **+0.157** VB / **+0.127** clustered | **.002** / **.0006** |

The earlier nulls were **underpowered, not absent** — the estimate needed
about 4,028 replies for 80% power, and 4,320 clears it. That part is solid;
a second Claude session verified it independently.

**The "smaller than real email" claim is not settled, and was corrected
today.** An earlier write-up quoted p≈.047 for the gap to real email's own
effect, unqualified. That number came from one of three legitimate ways to
estimate it, and it happens to be the one whose standard error (10 persona
clusters) is known to run anti-conservative:

| Fit | p (gap vs. real email) |
|---|---:|
| Draw 1 only, VB | .209 |
| 3 draws pooled, VB | .182 |
| 3 draws pooled, persona-clustered | .047 |

Honest statement: the effect itself is real (all fits agree). Whether it is
smaller than real email's is not established — it depends on the method.
This has been corrected everywhere it appeared in this repo.

Full write-up: `PROGRESS_llms.md`, sections "The main Q1 run reaches
significance (Sep 23)" and the estimator-dependence table within it.
`HANDOVER.md` section 6.2, "Q1 shows the effect, but the gap to real email
is not settled," has already been rewritten to match — no action needed
there.

---

## 2. Two bugs found and fixed today, both about multi-draw grids

**`grid_draw_reliability`** (commit `a9ae1d2`) was hard-coded to expect
exactly draws 1 and 2 and crashed on the 3-draw grid. Generalized to any
number of draws: proper multi-rater ICC, mean pairwise correlation, Fleiss'
kappa for decision agreement (still exactly Cohen's kappa at two draws, so
no previously published two-draw number changed). `run_q1_analysis` now
runs on the 3-draw grid without a workaround.

**`grid_contrasts` and `contrast_estimates`** (commit `82cd1ae`) — a second,
more consequential bug, found by a second Claude session cross-checking
`PROGRESS_llms.md` against the grid's own printed output. Both always read
the draw-1-only model fits regardless of how many draws a grid had, so the
CLI's "Against real email" table and the saved manifest's own headline
number silently used a third of the 3-draw grid's data (+0.134) while the
progress log quoted the correctly pooled number (+0.157/+0.127) — two
different numbers for the same quantity, from the same code path. Both
functions now prefer the pooled fit
(`sentence_model_clustered`/`aggregated_reply_model`/`aggregated_hedge_model`)
when a grid has more than one draw. `outputs/manifests/q1_full_grid_3draws.json`
is regenerated with the fix.

Both fixes are TDD'd, tested, and pass the full suite plus black/ruff/mypy.

---

## 3. Open, in progress: is `is_imperative` even measuring the right thing?

Not yet written into any log. A second Claude session found, on the 3-draw
grid, and flagged as "worth knowing, not yet acted on":

1. Generated replies average 1.42 sentences and 93 characters; the real-email
   benchmark averages 4.75 sentences. 60.8% of generated replies are a
   single sentence, so `imperative_ratio` can only take about three values
   per reply — likely most of why its draw-to-draw ICC is only 0.351.
2. The imperative-detection rule only catches bare imperatives. 10.6% of
   replies carry a directive cue with no bare imperative and score zero;
   the miss rate is worst writing down (13.0% down, 8.7% lateral, 10.2%
   up). A crude recount moves the down-vs-lateral gap from 3.7 to about 8.0
   points.

Neither changes a number already published. Both are about whether the
measure itself is valid, not about the model fits. The user was, as of
this session, discussing a length-matching diagnostic with that other
session — check with them before duplicating that work.

---

## 4. What the previous sessions did today (all committed and pushed)

| Work | Commit(s) |
|---|---|
| Persona-clustered cross-check for the sentence-level p-value | `85e833a` |
| Prompt hash persisted into every saved Q1 grid | `5ee4b46` |
| `q1_models.py`'s grid loader reports real cache/generation counts | `36fbc3e` |
| Four-model comparison regenerated with the clustered fit | `9b922af` |
| Main Q1 run: third draw generated overnight, reaches significance | `3bf5fb3` |
| `grid_draw_reliability` generalized past two draws | `a9ae1d2`, `e058937` |
| `grid_contrasts`/`contrast_estimates` use the pooled fit for multi-draw grids | `82cd1ae`, `7402fbc` |
| `HANDOVER.md` section 6.2 rewritten to match | local only, not a commit (gitignored) |

---

## 5. What to do next

Ranked.

1. **Decide on the Q1 measurement-validity question in section 3** with the
   user, before building anything — it may change what "the effect" even
   means, which would touch every number in `PROGRESS_llms.md`'s Q1
   section.
2. **The length-matched Q1-vs-real comparison.** `borrowed_words` already
   has this rule; `is_imperative` needs it too. Directly related to item 1
   — do them together, not separately.
3. **Run Q1 at scale on the other three models.** Only Llama has had the
   overnight-sized run. DeepSeek, gpt-oss-20b and gpt-oss-120b are still on
   the original 240-cell pilot. gpt-oss-120b is Groq-rate-capped to a
   multi-day job, not an overnight one; the other two are not.
4. **Human coding.** `blind_review.py` and its test are untracked — commit
   them regardless of when coding starts. No person has coded anything yet,
   so every mirroring validation still rests on Claude's own first-pass
   codes.
5. **`q1_models.py`'s own output manifest** still doesn't copy a grid's
   `prompt_text_hash` into its per-model entries — small, not urgent.

Full "Next steps" list, most-valuable-first, with more items: `PROGRESS_llms.md`.

---

## 6. Traps that have actually cost time

- **The laptop must stay awake for an overnight run.** Sleep pauses it,
  shutdown kills it. Long runs belong in a detached `tmux` session.
- **A killed run is cheap to resume.** The response cache regenerates only
  what's missing. Never start from zero.
- **Other sessions work in this same repo, concurrently.** Stage explicit
  filenames, never `git add -A`, and keep the gap between `git add` and
  `git commit` short. Check `git status` for files that aren't yours
  before staging — `review_pack.py` (modified) and `blind_review.py` +
  its test (untracked) are not this session's; leave them alone unless
  told otherwise.
- **Coordinate with concurrent sessions before touching shared files.**
  `q1.py`, `PROGRESS_llms.md`, and `HANDOVER.md` all had two sessions
  working on them today; a quick cross-session message before a large
  edit avoided duplicate or conflicting work more than once.
- **Local generation costs no API usage.** It keeps running with the
  session closed. Worth knowing if a user is short on quota.

---

## 7. Suggested skills

Call these with the Skill tool, as the work in section 5 comes up:

| Skill | When |
|---|---|
| `superpowers:test-driven-development` | Any change to a model-fitting function. This repo's convention: a fit change ships with a test that recovers a known injected effect, and any bugfix ships with a golden-value regression test captured before the change. |
| `superpowers:systematic-debugging` | If the section 3 measurement question turns up an actual bug (not just a validity question) in `extract_q1_sentence_features` or the imperative-detection rule. |
| `superpowers:verification-before-completion` | Before any number goes into a progress log — this project has now twice found a wrong number already published because this step was skipped. |

Agents (via the Agent tool): `results-verifier` before a number goes into a
progress log or a chapter (Opus, costs a lot, so not for routine checks);
`experiment-runner` for the other three models' at-scale runs; spawn on
Sonnet, not Opus — the user is usage-constrained.

---

## 8. House rules that are easy to get wrong

- **Writing style is a rule, not a preference.** Short sentences, one idea
  each, plain words, result first. No rhetorical build-up, no long comma
  chains. Applies to chat, logs, commit messages and docstrings.
- **Never swap a number quietly.** If a number changes, say what changed
  and why, in writing, in the log. Section 1 above is an example of this
  rule in practice, not an exception to it.
- **Quality gates before every commit:** `black`, `ruff`, `mypy`, and the
  full test suite. All four, every time.
- **Push after each result.** The user reads results on GitHub, not
  locally.
