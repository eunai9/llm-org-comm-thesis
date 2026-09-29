# Session handoff — 2026-09-25, 1536

For a fresh agent continuing this Claude Code session on the LLM-org-comm
thesis project. Read this first. It does not duplicate the spec, the plan,
or the repo's own logs — it points to them.

## Where things live

- This chat's working folder is the Windows OneDrive folder (`Expose.pdf`
  and docs only). The real repo is in WSL at `~/projects/thesis`, pushed to
  `https://github.com/eunai9/llm-org-comm-thesis`, branch `main`. Reach it
  via `wsl.exe -e bash -c "cd ~/projects/thesis && <command>"`.
- `HANDOVER.md` (repo root, gitignored, local only) has full project
  context. Read it before anything else in the repo.
- A peer Claude session may also be working in this same repo. This
  session was told explicitly: the `gpt-oss-120b` generation job (via
  Groq) belongs to that peer session. **Do not touch it** — do not launch,
  monitor, or "helpfully" fix anything about it, even with useful
  diagnostic context in hand. See the pinned lesson in
  `respect-task-ownership-in-shared-repo.md` (this session's memory).
- Convention in force: stage explicit filenames, never `git add -A`, keep
  the gap between `git add` and `git commit` short (a peer session's
  staged-but-uncommitted files can otherwise get swept into a commit).

## What this session did, in order

1. Resumed from `docs/general_handoff_2026-09-25_1244.md`. Found WSL had
   rebooted again just before this session started; no job was running.
2. Investigated the previous session's Groq 429 on `gpt-oss-120b`.
   Root-caused it precisely: Groq's **daily token cap** (200,000 TPD), not
   a per-minute limit as the prior handoff guessed — confirmed via a raw
   API probe whose error body read `tokens per day (TPD): Limit 200000,
   Used 199795`. This is a rolling window, not a midnight reset. Launched
   the job once to verify, then the user said it belongs to another
   session — stopped immediately, killed the tmux session and the
   monitor, touched nothing about it since.
3. User asked whether NVIDIA might be blocking their account (too much
   use). Checked: all NVIDIA failures were plain connection timeouts, zero
   429/403/401 responses, and the same key works fine for other calls —
   concluded it's NVIDIA's infrastructure being degraded, not an
   account-level block. A new account would hit the same degraded
   backend.
4. Ran the `gpt-oss-20b` full grid (1,440 cells) via NVIDIA. Fixed a real
   bug along the way: launching a long job through nested
   `wsl.exe -e bash -c "... tmux new-session ..."` quoting silently failed
   (empty log, no error) — the same class of bug the prior session's
   `watchdog.sh` had. Fixed by writing a real script file instead of an
   inline nested command: `scripts/run_q1_gptoss20b_full.sh` (currently
   **untracked**, not yet committed — it's a reusable launcher, worth
   committing if this job continues).
5. The job progressed but very slowly (NVIDIA timing out on most calls,
   self-healing via retries rather than crashing). After watching it for
   about 40 minutes with only 25/1440 cells done, the user chose to pause
   rather than let it keep grinding. **Killed, not running.** Progress is
   safe: 25 cells cached (11 from an earlier cache, 14 freshly generated
   this session), nothing lost, same command resumes from there. No
   output parquet exists yet (`data/interim/q1_direction_grid_gpt_oss_20b_full.parquet`
   is only written on a completed run) — progress lives in `runs/_cache`
   only.
6. Confirmed `deepseek-ai/deepseek-v4.1-flash` is genuinely broken on
   NVIDIA specifically right now (times out at 90s while `gpt-oss-20b` on
   the same endpoint responds in 1.5s, and the model is listed live in the
   catalog) — a real per-model deployment problem, not a general outage.
   Left untouched, matches the prior session's finding.
7. Read `PROGRESS_llms.md`'s "Next steps" and sorted every item into
   blocked-on-generation (items 2, 4, 6 — skip, per the user's steer to
   hold off on the three-draw work) vs. doable now vs. needs-the-user.
8. **Item 10, done and pushed** (commit `50bd3aa`): committed
   `blind_review.py` + `tests/test_blind_review.py` (untracked since Sep
   13, 9/9 tests passing) together with `review_pack.py`'s `wrong_register`
   codebook fix (already made, uncommitted, the test suite depends on it).
   This also satisfies Task 1 of
   `docs/superpowers/plans/2026-09-23-q1-role-inference-plan.md`, and
   unblocks item 3 (the mirroring hand-coding page is now ready to run:
   `python -m thesis.analysis.blind_review`).
9. **Item 1 (length-matched Q1-vs-real comparison), explored but not
   built.** Used `superpowers:brainstorming` (bounded path), designed an
   approach that mirrors `mirroring.py`'s existing `truncate_words`
   pattern at sentence grain instead of word grain (cut real email to
   each model's own mean sentence count, first-N sentences, reusing
   `fit_sentence_model`/`fit_email_model` unchanged). Before the user
   approved the design, they asked to see real numbers first. A throwaway
   spike script (not committed, not in `src/`) found something
   substantive: cutting real email down to a model's own short length
   **increases** real email's own per-sentence directive rate in 3 of 4
   models, the opposite of the assumed length-artifact direction. This
   would shrink `gpt-oss-120b`'s reported overshoot (+0.643 → roughly
   +0.459) and mostly erase DeepSeek's. **The user is rightly skeptical**
   — this is one uncross-checked script, not a verified result. Do not
   put these numbers in any log or slide. If this gets picked back up:
   the spike script's logic is worth re-deriving cleanly (it did catch and
   fix one real bug — `q1_real_emails.parquet` stacks both "strict" and
   "loose" samples under a `sample` column; filter to `sample == "strict"`
   or you double-count).
10. **Item 5 (embedding map + review pack for the newer models), half
    done.** Review packs built and written for all three models
    (`outputs/tables/review_deepseek/`, `review_gpt_oss/`,
    `review_gpt_oss_120b/` — gitignored, contain Enron text, not
    committed). One finding worth the user's attention: gpt-oss replies
    end mid-sentence 18–20% of the time (`share_ends_mid_sentence`:
    gpt-oss-20b 0.20, gpt-oss-120b 0.18), against 0% for DeepSeek — not
    previously reported anywhere in the logs.
11. **Embedding map blocked.** It needs Ollama (for `nomic-embed-text`
    embeddings). WSL cannot reach Ollama running on Windows:
    `127.0.0.1:11434` gives "connection refused" from WSL, even though
    Ollama responds fine to Windows-native requests (`200 OK` via
    PowerShell `Invoke-WebRequest`) and the Windows Firewall rule allows
    it. Diagnosed as WSL2 localhost-forwarding not working for this port
    right now. **The user tried `wsl --shutdown` + restart at my
    suggestion — did not fix it.** Not yet tried: setting
    `OLLAMA_HOST=0.0.0.0` on the Windows Ollama service (binds to all
    interfaces instead of loopback-only, sidesteps the forwarding
    mechanism) — proposed to the user, a persistent system change so
    better done by them, not yet actioned as of this handoff. Also worth
    knowing: even once connectivity works, `nomic-embed-text` is **not
    yet pulled** on this Windows Ollama instance (`ollama list` there
    shows only `llama3.2:3b` and `qwen2:7b`) — that will need
    `ollama pull nomic-embed-text` too.

## Current live state (as of this handoff)

- No tmux session, no `thesis.analysis.q1` process running. Everything is
  idle.
- Uncommitted, real (not noise): `scripts/run_q1_gptoss20b_full.sh`
  (untracked, the quoting-safe launcher — worth committing before reuse),
  `logs/` (untracked, gitignored probably worth adding to `.gitignore`
  explicitly if not already), `outputs/manifests/cost_ledger.csv`
  (modified, today's real API calls).
- Untracked and **not this session's work** — leave alone unless the user
  says otherwise: `docs/figures/mirroring_*.png` (8 files, present since
  before this session started, likely the peer session's).
- `outputs/tables/review_deepseek/`, `review_gpt_oss/`,
  `review_gpt_oss_120b/` exist locally, gitignored by design (Enron
  text), nothing to commit there.
- Ollama on Windows: process alive, responds to Windows-native requests,
  unreachable from WSL. `nomic-embed-text` not pulled there yet.

## Immediate next steps, in order

1. If picking the embedding map back up: check whether the user has set
   `OLLAMA_HOST=0.0.0.0` and restarted Ollama. If so, verify from WSL
   (`curl http://127.0.0.1:11434/api/tags`), pull `nomic-embed-text` if
   missing, then run
   `python -m thesis.analysis.embedding_map --pairs data/interim/pairs_<model>.parquet --figure-prefix embedding_<model>_ --out outputs/manifests/embedding_map_<model>.json`
   for `deepseek`, `gpt_oss`, `gpt_oss_120b`.
2. Decide with the user whether `scripts/run_q1_gptoss20b_full.sh` and
   `logs/` should be committed (the launcher is reusable and fixes a real
   bug class) or left local.
3. If resuming `gpt-oss-20b`: same command as before
   (`python -m thesis.analysis.q1 --nvidia openai/gpt-oss-20b@low --design full --out data/interim/q1_direction_grid_gpt_oss_20b_full.parquet --progress-every 25`),
   launched via `scripts/run_q1_gptoss20b_full.sh` in `tmux`. Check
   NVIDIA's reliability first (a quick single-call probe succeeding does
   **not** mean sustained calls will — this session learned that the hard
   way). Expect it to be slow; the user already chose to pause once under
   this exact condition, so check with them before re-launching rather
   than assuming they want it running again.
4. If picking item 1 (length-matched Q1-vs-real) back up: this needs a
   proper design approval first, not just a spike. See point 9 above —
   the spike's finding is interesting but unverified. Don't skip straight
   to implementation.
5. Still open from `PROGRESS_llms.md`'s Next steps and not touched this
   session: item 3 (now unblocked, needs the user personally to code —
   the page is ready), items 7, 8, 9 (small code cleanups + cache
   backup — item 9 needs the user to name a backup destination).
6. Do not touch `gpt-oss-120b` (Groq) — see "Where things live" above.

## Suggested skills

- **superpowers:brainstorming** — if item 1 (length-matched comparison)
  gets picked back up, this needs to go through proper design approval
  before implementation, since the spike surfaced a real methodological
  question (first-N vs. last-N sentence truncation) the user hasn't
  weighed in on yet.
- **results-verifier** agent — before any number from the length-matching
  spike, or from the widened model set once `gpt-oss-20b` finishes, goes
  into a progress log, a chapter, or anything supervisor-facing. This
  project has a documented history of an unverified number nearly
  reaching a supervisor slide (`PROGRESS_llms.md`, "The main Q1 run
  reaches significance" section) — treat that as the standing bar.
- **progress-writer** agent — for eventual `PROGRESS.md` /
  `PROGRESS_llms.md` entries once `gpt-oss-20b` finishes or the embedding
  map lands, per the project's append-don't-rewrite convention.
- **superpowers:systematic-debugging** — if the Ollama/WSL connectivity
  issue resurfaces after the user's fix attempt; this session already
  ruled out "account block" and "stale post-restart state" as causes for
  two different problems today (NVIDIA reliability, WSL networking) by
  checking rather than assuming, and that discipline is worth continuing
  rather than guessing again.

## Not included here (see the artifacts directly)

- Full reasoning for the Q1 redesign: `docs/superpowers/specs/2026-09-23-q1-authority-judge-design.md`.
- Full task-by-task implementation plan: `docs/superpowers/plans/2026-09-23-q1-role-inference-plan.md`
  (Task 1 now done, commit `50bd3aa`; Tasks 2–13 not started, plan's own
  header still asks the user to pick an execution method first).
- `PROGRESS_llms.md`'s "Next steps" section — the full, current list this
  session worked from.
- Recent commit history: `git log --oneline -5` in the repo.
