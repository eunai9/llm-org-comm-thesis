# Session handoff — 2026-11-23 17:32

Read this first, then `PROGRESS_llms.md` sections 25 and 26 on `main`.

## Where things live

- WSL repo `~/projects/thesis`: branch `layer1-consensus-tone`. Code and `.venv`. Run Python here.
- WSL repo `~/projects/thesis-main`: branch `main`. `PROGRESS_llms.md`, figures, manifests, handoffs. No venv.
- Convention: code is committed on the branch. The log, figures and manifests are committed on `main`.
- OneDrive folder (Windows side): the deck `Final Result for Thesis.pptx`, backups in `backups/`, `Results overview (2026-11).md`.

## Done this session (details are in the commits and sections)

- Deck: the stakes and tone slide (new slide 7) is now saved to OneDrive. The deck has 11 slides. md5 `3b42204a995ef7b1acfae3ddc912b063` on 2026-11-21. If the md5 differs, the user edited it by hand; work from their file.
- Section 25 (qwen): the direction gap inside each of the six scenarios (task types), with a plain-language description per scenario. Commit `971e44d` on `main`.
- Section 26 (all four models): the same analysis for gpt-oss-120b, gpt-oss-20b and Llama, plus a heatmap comparison. Commit `ce116d6` on `main`. The old sections 26 and 27 are now 27 ("Where the code lives") and 28 ("Next steps").
- Code: `src/thesis/analysis/scenario_direction.py`, commits `0d34880` and `6819ee8` on the branch. Flags: `--writer MODEL`, `--judge`, `--compare`.
- All four models' draw-1 main sets now have a gpt-oss-120b tone score for every reply (in the response cache). Local measures are cached at `data/interim/q1_{slug}_stakes_tone_scores.parquet`.
- Power analysis (answered in chat, not saved to a file): with qwen's observed gaps, large gaps need 20–40 replies per direction, medium gaps 60–110 (twice that after Holm), and decision shifts 100–320. More personas help more than more draws of the same 10.

## Open items (not started)

- Deck: the footer "Q1 — examples and results" is wrong on the Q2 and open-problems slides (slides 10 and 11). The title slide still says "Progress update" and "October 21, 2026". The old all-model tone slide and the qwen measures slide have similar titles.
- The deck has no slide for sections 25 or 26 yet. The user may want one.
- Add sections 24–26 and the per-item Q2 rubric numbers to `Results overview (2026-11).md`.
- results-verifier has not checked sections 24, 25 or 26.
- Possible follow-ups from section 26:
  - Llama could use all three draws.
  - gpt-oss-120b judges its own tone. A per-scenario check with the gpt-oss-20b judge is not done.
  - The keyword markers ("tells the other to act", "I'll do it") are rough and not validated.

## Gotchas

- Groq runs need `set -a && . ./.env && set +a` first. Run long jobs in tmux.
  - Nested quoting through `wsl.exe -e bash -c` loses `$vars`. Put loops in a script file instead. The last runner is `~/projects/thesis/logs/run_scenario_tone.sh`.
  - The tone judge runs at about 10 replies per minute. One model took about 2 hours, but the gpt-oss-20b run spread over a day (daily cap or laptop sleep).
- pandas string regex uses the pyarrow engine (RE2). `\u` escapes fail there. Also, the Edit/Write tools turn `’` into a literal `’`, which ruff flags (RUF001). Use `chr(0x2019)`.
- Windows-side writes can add CRLF. Strip with `sed -i 's/\r$//'` before committing.
- The branch has untracked `embedding_*` figures and a modified `cost_ledger.csv` that are not from this session. Leave them alone.
- The Windows-side `python` is not on PATH. Use WSL python.

## Suggested skills

- `anthropic-skills:pptx` for any deck work (for example a slide for sections 25 or 26).
- `dataviz` before making or changing a chart.
- `superpowers:verification-before-completion` before claiming a result holds or the deck is saved.
