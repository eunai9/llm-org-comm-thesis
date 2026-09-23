# General handoff, 2026-09-23

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

---

## 1. Read this before trusting `HANDOVER.md` section 6.2

**Q1 now reaches significance. `HANDOVER.md` section 6.2 says it does not.
That section is stale as of Sep 23.**

A third draw was generated overnight on Sep 23. The writing-down effect on
Llama 3.2 3B is now significant:

| Sample | Coefficient | p |
|---|---:|---:|
| 240 replies (`PROGRESS.md` s39) | +0.163 | .401 |
| 1,440 replies (s51) | +0.134 | .092 |
| 2,880 replies, 2 draws (s54) | +0.092 | .130 |
| **4,320 replies, 3 draws** | **+0.157** VB / **+0.127** clustered | **.002** / **.0006** |

Two things follow.

1. The earlier nulls were **underpowered, not absent**. The estimate needed
   about 4,028 replies for 80% power. 4,320 clears it.
2. The simulator's effect is still **smaller than real email's** (+0.127
   against +0.253, difference p approx .047). That is the same conclusion
   `PROGRESS.md` section 54 reached at two draws, now on firmer ground.

Full write-up: `PROGRESS_llms.md`, section "The main Q1 run reaches
significance (Sep 23)". Fixing section 6.2 of `HANDOVER.md` is a small,
worthwhile first task.

---

## 2. Known open bug, blocking the 3-draw grid

`grid_draw_reliability` is hard-coded for exactly two draws and crashes on
three. So `run_q1_analysis` cannot be called on
`data/interim/q1_direction_grid_full_3draws.parquet` without a workaround.
The Sep 23 numbers above were produced by calling the fitting functions
directly and skipping the reliability step.

The grid itself is complete and correct. The blocker is a design decision, not
just a code fix: **what should reliability mean with three or more draws?**
Draw 1 against draw 2 only, every pair averaged, or something else. Decide
that first, then fix, then the whole analysis runs end to end again.

---

## 3. What the previous session did

All committed and pushed unless noted.

| Work | Where |
|---|---|
| Q1 at two draws, plus the aggregated/clustered analysis and a draw-reliability report | `PROGRESS.md` section 54 |
| Cache made crash-safe: a damaged entry is a miss, not a crash; writes `fsync` before rename | commit `7e99f40` |
| Figure path made draw-count aware, so a multi-draw run cannot overwrite a single-draw figure | in `q1.py`, `full_grid_figure_path` |
| `HANDOVER.md` extended with operational context and the three big problems | local only, gitignored |

A note on that last row: `HANDOVER.md` is in `.gitignore` on purpose, so it
exists only on this laptop and is not on GitHub.

---

## 4. What to do next

Ranked. The first two are cheap.

1. **Fix `HANDOVER.md` section 6.2** so it matches section 1 above. Five
   minutes, and it stops the next agent inheriting a wrong headline.
2. **Decide what reliability means with 3+ draws, then fix
   `grid_draw_reliability`.** Unblocks the 3-draw grid.
3. **Re-judge Q2.** It is the last result still resting on the pre-Sep-5
   prompt. Needs a reusable module built first, then about 366 judge calls,
   2 to 3 hours. Its headline, that role consistency fails equivalence, should
   not be trusted until then. See `HANDOVER.md` section 6.3(b).
4. **Run Q1 on the larger models.** Only Llama has the full treatment.
   DeepSeek showed the effect on the old 240-cell design. gpt-oss-20b and
   gpt-oss-120b have not been run on any Q1 grid.
5. **Human coding.** The blind coding page exists but `blind_review.py` and
   its test are **untracked**, so they live only on this laptop. Committing
   them is a good idea regardless. No person has coded anything yet, so every
   mirroring validation still rests on Claude's own first-pass codes.
6. **Not built, named in the research plan:** a contamination probe and an
   anonymized-stimulus arm. Both cheap, both close an obvious examiner
   objection.

Supervisor questions are listed in `HANDOVER.md` section 10. Note that item 5
there, whether Q1 is worth 14 more hours, has now been answered by running it.

---

## 5. Traps that have actually cost days

Do not rediscover these. Full detail in `HANDOVER.md` section 9.

- **The laptop must stay awake.** Sleep pauses a run, shutdown kills it. Long
  runs belong in a detached `tmux` session, not in a Claude session.
- **A killed run is cheap to resume.** The cache regenerates only what is
  missing. Never start from zero.
- **Another session works in this same repo.** Stage explicit filenames, never
  `git add -A`, and keep the gap between `git add` and `git commit` short.
  Staged work has been swept into another session's commit before.
- **Three things exist only on this laptop:** the 59 MB response cache
  (`runs/_cache`), `blind_review.py`, and `HANDOVER.md`. None is backed up.
- **Local generation costs no Claude usage.** It keeps running with the
  session closed. Say so if the user is short on quota.

---

## 6. Suggested skills and agents

Call these with the Skill tool:

| Skill | When |
|---|---|
| `superpowers:systematic-debugging` | The `grid_draw_reliability` crash, before proposing a fix. |
| `superpowers:test-driven-development` | Any fix to that function. This repo's convention is that a model-fitting change ships with a test that recovers a known injected effect. |
| `superpowers:verification-before-completion` | Before claiming any result or fix is done. This project has been burned repeatedly by stale or unverified numbers. |

Call these with the Agent tool:

| Agent | When |
|---|---|
| `results-verifier` | Before a number goes into a progress log, a chapter, or a supervisor email. Runs on Opus and costs a lot, so not for routine checks. The Sep 23 significance result is worth one pass. |
| `experiment-runner` | The Q2 re-judge, Q1 on the larger models, or any re-run from cache. It launches long runs and hands back rather than waiting. |
| `progress-writer` | Recording a verified result in the logs. |
| `lit-scout` | Related-work searches only. |

Spawn subagents on Sonnet, not Opus. The user is usage-constrained.

---

## 7. House rules that are easy to get wrong

- **Writing style is a rule, not a preference.** Short sentences, one idea
  each, plain words, result first. No rhetorical build-up, no long comma
  chains, no dashes bolted onto clauses. Applies to chat, logs, commit
  messages and docstrings.
- **Never swap a number quietly.** If a number changes, say what changed and
  why. The logs record their own corrections on purpose.
- **Quality gates before every commit:** `black`, `ruff`, `mypy`, and the full
  test suite. All four.
- **Push after each result.** The user reads results on GitHub, not locally.
