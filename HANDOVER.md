# Handover: context for a new session

Read this first. It carries the thesis topic, the data, the simulator, the
models, the judging criteria, and the open problems. It is written so a new
session can pick up work without reading the full logs.

The three logs, and what each is for:

| File | What it holds |
|---|---|
| `PROGRESS.md` | The main log. 54 numbered sections, in date order. Every result and every failed attempt. |
| `PROGRESS_llms.md` | The four-model comparison, organized by topic (mirroring, telling AI from real, Q1). |
| `PROGRESS_nvidia.md` | The free-tier build log, in date order. How the NVIDIA and Groq clients came about. |

When a number matters, check the log section it came from and the date. Some
numbers in older sections are stale. See "Instability" below.

---

## 1. The thesis

**Title.** LLMs for Simulating Organizational Communication and Decision-Making
with Email.

**The idea.** Give an LLM a persona built from a real employee's email habits,
show it a real incoming email, and have it write the reply. Then ask two
things: does the simulated reply behave like a real one, and can an LLM judge
score such replies reliably.

**Three research questions.**

| | Question | How it is answered |
|---|---|---|
| **Q1** | Does hierarchical role change what gets written? | Count orders and hedges in generated replies, split by whether the persona writes up, down, or to a peer. Mixed models, no judge involved. Compared against the same measurement on real Enron email. |
| **Q2** | Are generated replies diverse enough, and close to the real pattern? | An LLM judge scores a real reply and a generated reply to the same email on 6 rubric items. Equivalence testing (TOST), not just a difference test. Plus a model-free check: can a plain classifier tell them apart. |
| **Q3** | Can an LLM judge give consistent and calibrated scores? | Judge-swap design. Two model families each write replies and each judge replies, including their own. Self-preference is the interaction term. |

**Timeline.** Data work done Aug 2026. Model and evaluation work Sep to Dec
2026. Writing Jan 2027. Defence end of Mar 2027.

**Standing decision: no paid API, ever.** Confirmed with the supervisor. All
model calls are local (Ollama) or on free tiers (NVIDIA, Groq). `configs/models.yaml`
still lists paid placeholders (`claude-opus-5`, `gpt-4o`); they are unused.

---

## 2. The dataset

**Source.** The official CMU 2015 Enron archive. Not the Kaggle CSV, though
that is the same corpus. Downloaded and checksum-verified.

| | Number |
|---|---:|
| Raw files | 517,401 |
| **Unique emails after dedup** | **254,359** |
| Mailboxes | 150 |
| Emails with an identified sender | 44.8% of usable ones |

Always cite 254,359, not 517,401. The raw count double-counts because the
export saved one copy per folder.

**Hierarchy comes from job titles, not from a reporting chart.** The planned
source, Agarwal et al.'s (2012) supervisor-pair gold standard, has no live
public copy. The fallback is a 156-person employee list from Perry & Wolfe
(2011), with a hand-made title-to-rank ladder frozen before any result was
computed. Ranks: 1 Employee, 2 Manager, 3 Director, 4 Vice President,
5 Managing Director, 6 President/CEO. Provenance is in
`data/external/SOURCES.md`.

**This matters for Q1.** "Writing down" means writing to someone of lower
job-title rank, not to your own subordinate. Two people at the same rank in
different departments count as peers. Say this as a limitation in the thesis.

**Threads** are reconstructed by subject line plus shared participants plus a
30-day window. The export stripped the technical headers, so there is no other
way. The rules are deliberately strict: splitting one conversation in two is
better than merging two unrelated ones.

**The samples everything uses:**

| Set | What it is | Size |
|---|---|---:|
| `S_label` | Emails an AI labels for purpose and tone | 3,000 |
| `S_shots` | Real threads the simulator replies to | 200 |
| `S_real_eval` | Real replies inside those threads, for side-by-side judging | 313 drawn |
| **Matched pairs actually used** | Real reply and generated reply to the same email | **183** |

Real replies in those 183 pairs: median 44 words, mean 65. The mean is
inflated by signature and header lines the cleaner does not strip.

**The Q1 real-email arm** (section 48) is separate: 2,202 strict emails from
107 senders, where both sender and one recipient have a known rank.

**Data rules.** The corpus is real correspondence. Analysis only. Never commit
it. Never commit API keys. `data/`, `runs/` and `outputs/tables/` are
gitignored for that reason.

---

## 3. The simulator

**One reply per cell.** A cell is a persona plus a scenario. The model gets a
persona block, a memory block, and one incoming email, and returns JSON:
`subject`, `body`, `decision`, `confidence`, `reasoning_brief`. The decision is
one of accept, decline, defer, escalate, none.

**Personas.** 10, derived from corpus statistics per rank and department, not
written by hand. Each carries style numbers: mean tokens, imperative ratio,
hedge rate, deference rate, question ratio, mean recipients. These numbers are
rendered into the prompt text, so a change to how they are computed changes
every prompt.

**Memory.** A memory stream in the style of Park et al.'s generative agents,
frozen to a snapshot so it stays constant across runs.

**The scenario grid.** 6 task types times 3 directions times 2 stakes levels
times 4 tones = 144 scenarios.

- Task types: request information, approve or decline, report problem,
  schedule coordination, resolve disagreement, confirm details.
- Directions: up, lateral, down.
- Stakes: routine, high.
- Tones: deferential, warm, neutral, assertive. This is the tone of the
  **incoming** message, not an instruction to the persona.

With 10 personas the full grid is 1,440 replies. An older 24-scenario subset
(240 replies) is what sections up to 39 used; it sits inside the full grid.

**Prompt variants.** `default` and `decide_first`. The second makes the model
state its decision before writing the email. It did not fix mirroring
(section 49).

**The cache is the backbone.** `runs/_cache`, about 59 MB, keyed on the exact
rendered prompt plus model, provider and draw index. Consequences:

- Re-running any analysis costs nothing and returns identical replies.
- Any upstream change that alters prompt text silently invalidates every
  cached reply for that prompt. This has happened five times (persona stats,
  corpus rebuild, the Sep 5 prompt change).
- The cache exists only on this laptop and must never go into git. It is not
  backed up. That is an open risk.

---

## 4. The models used

| Model | Family | Where it runs | Role |
|---|---|---|---|
| llama3.2:3b | Meta | Local, Ollama on Windows | Main simulator model for all Q1/Q2 work |
| qwen2.5:3b | Alibaba | Local, Ollama | Second family for the Q3 judge-swap |
| DeepSeek V4 Flash | DeepSeek | NVIDIA free tier | Larger-model comparison |
| gpt-oss-20b | OpenAI | NVIDIA free tier, `@low` | Larger-model comparison |
| gpt-oss-120b | OpenAI | Groq free tier, `@low` | Larger-model comparison |

**Known limitation running through everything.** Every Q1, Q2 and Q3 result
comes from 3B local models. They prove the machinery works and have caught
several real bugs, but they cannot stand in a final results table. Whether the
thesis gets bigger models, or is reframed around open-weight models as the
object of study, is an open supervisor question.

**Practical notes.** Ollama runs on Windows, not inside WSL, so WSL reaches it
through a temporary second server on the WSL-facing address. The NVIDIA tier
stalls often and needs retries. The Groq tier is fast but capped at 8,000
tokens per minute, so calls are spaced 15 seconds apart. Long runs belong in
the user's own `tmux` window, not in a Claude session, and the laptop must
stay awake.

---

## 5. Judging criteria

**Q1 uses no judge.** It counts language features directly:

- **Orders per reply** (`imperative_ratio`): share of a reply's sentences that
  give an order.
- **Orders per sentence** (`is_imperative`): each sentence counts once.
  This is the primary outcome, because most Llama replies are one sentence
  long, which makes the per-reply share crude.
- **Hedges per reply** (`hedge_rate`): share of sentences with a softener.

Models: linear mixed model per reply, logistic mixed model per sentence, both
with a random intercept per persona, lateral as the reference level.

**Q2 and Q3 use the judge rubric.** Six items, 1 to 5, in two groups:

| Group | Item | What it asks |
|---|---|---|
| Empirical fidelity | `role_consistency` | Does it read as someone in this role and seniority? |
| Empirical fidelity | `contextual_fit` | Does it engage with this specific message, not any message? |
| Empirical fidelity | `corpus_plausibility` | Could it have appeared in a real corporate archive? |
| Communication performance | `clarity` | Is it clear what it says and wants? |
| Communication performance | `politeness_appropriateness` | Is the politeness level fitting, not maximal? |
| Communication performance | `conflict_management` | Where it pushes back, does it protect the relationship? |

**Q2** scores a real reply and a generated reply to the same email, then runs
equivalence tests (TOST). Equivalence, not a plain difference test, because the
claim is "these are alike", and a non-significant difference alone cannot
support that.

**Q3** uses the interaction term of a generator-by-judge model. Three numbers
separate three explanations for the same pattern: generator quality (one model
writes better), judge generosity (one model scores everything higher), and
self-preference (what is left over, the interaction). Only the third answers Q3.

**Q2 also has a model-free check**, which matters as much as the judge: can a
plain classifier tell real from generated. The judge can say "equivalent" while
a TF-IDF classifier separates them at 0.98 AUC. Both are reported.

---

## 6. The three big problems

### 6.1 Mirroring

**What it is.** The generated reply is built mostly from the sender's own
words, and often hands the request straight back. Asked to approve two vacation
days, the persona replies "Can you confirm that these dates are acceptable?".

**How it is measured.** `borrowed_words`: the share of a reply's distinct
content words that already appear in the incoming email. A reply scoring 0.80
or more is flagged. The measure scores 0.834 AUC against 100 hand-coded
replies, beating three cleverer alternatives.

**Where it stands.**

| | Mean borrowed words, at equal length | Flagged |
|---|---:|---:|
| Llama 3.2 3B | 0.565 | 19.1% |
| DeepSeek V4 | 0.469 | 8.2% |
| gpt-oss-120b | 0.435 | 4.4% |
| gpt-oss-20b | 0.373 | 4.9% |
| Real replies | 0.294 | 9.8% |

**Two prompt fixes failed.** Telling the persona it must act (section 43)
changed how replies were phrased and not what they did. Making it decide before
writing (section 49) looked better only because replies got longer; at equal
length it moved the wrong way. In two cases the stated plan says "I will act"
and the email still hands the task back.

**What the evidence now points at.** Model size, not prompting. The larger
models mirror far less with the identical prompt. But size does not fix
everything: all four models still borrow more than real writers, and the larger
ones remain easy to spot by stock phrases ("I'll", "let me know").

**The open route.** Three attempts at a better automatic measure failed
(semantic similarity, asking a model directly, splitting the signal). What is
left is human coding. A blind coding page is built and waiting: 50 emails, two
replies each from two models, model hidden, random order. No person has coded
anything yet. Every validation so far rests on Claude's own first-pass codes.

### 6.2 Q1 has no significant effect

**Real email has the effect.** People give more orders when writing down:
+0.253 on the logit scale, p<.001, in every robustness check, across 2,202
emails from 107 senders. This is the benchmark (section 48).

**The simulator cannot show it clearly.**

| Run | Writing down | p |
|---|---:|---:|
| 240 replies (section 39) | +0.163 | .401 |
| 1,440 replies (section 51) | +0.134 | .092 |
| 1,440 replies, 2 draws averaged (section 54) | +0.092 | .130 |

Going from 240 to 1,440 replies made the estimate about 2.4 times more precise
and still did not reach significance. Adding a second draw moved the estimate
down rather than tightening it into significance.

**The newest result changes the reading.** With two draws the simulator's
writing-down effect is now **significantly smaller than real email's**, p=.043.
Before that, the two could only be called indistinguishable. So the current
honest statement is not "the simulator matches real email but too noisily to
prove it". It is "the simulator under-produces the effect".

**Reaching 80% power on an effect this size needs about 4,028 replies**, which
is roughly 14 more hours of local generation. Whether that is worth spending on
one 3B model with 10 personas is an open supervisor question.

**One clear positive.** The simulator hedges less when writing down, and real
email does not do that (−0.060, p=.004, survives correction). It is the first
place the simulator clearly behaves unlike real email. It appeared only after
the Sep 5 prompt change, so it needs replication before being reported.

### 6.3 Unstable results

Three different kinds of instability, often confused with each other.

**(a) The model disagrees with itself.** Same prompt, same model, different
draw:

| Measure | Agreement between two draws |
|---|---|
| Decision field | 60% at 183 pairs (kappa 0.25); 67.7% at 1,440 cells (kappa 0.352) |
| Orders per reply | r = 0.15 at 183 pairs; ICC 0.346 at 1,440 cells |
| Borrowed-words mean | Stable. Its draw-to-draw noise floor is −0.004, far below real effects. |
| Mirroring flag counts | Not stable. Flag changes of the size earlier sections reported also occur between identical prompts. |

Read every paired comparison against this floor. A single draw is a noisy
measurement of any one cell. Spearman-Brown says three draws would raise
reliability to about 0.61 for orders per reply, which helps but does not make a
single cell trustworthy.

**(b) Numbers go stale when the pipeline changes.** The prompt changed on
Sep 5 and silently invalidated Q1, Q3 and Q2 results generated before it.
Nothing caught this for two weeks. Q1 (section 51) and Q3 (section 52) have
been re-run since; both changed, and Q3's self-preference went from "not
confirmed" (p=.142) to a real finding (+0.397, p=.005). Run manifests now store
a prompt hash so this is detectable.

**Still stale: Q2 (section 38).** Its 183 judged pairs date from Sep 2. It is
the last result resting on the old prompt. Re-judging needs a reusable module
built first, then about 366 judge calls, 2 to 3 hours. Its headline, that role
consistency fails equivalence, should not be trusted until then.

**(c) Upstream data fixes invalidate caches.** The corpus rebuild (section 37)
and the persona-statistics fix (section 31) both changed prompt text and forced
full regeneration. This is by design, since the cache keys on rendered prompt
text, but it means any data fix costs a regeneration run.

---

## 7. What is being tested now

1. **Bigger models as the mirroring answer.** Four models now answer the same
   183 emails with the same prompt. Documented in `PROGRESS_llms.md`. Mirroring
   drops a lot with size; separability by word choice does not.
2. **Draws instead of cells for Q1.** Section 54 added a second draw across all
   1,440 cells. A third draw is the next obvious step, and cheaper than more
   cells.
3. **Q1 on the larger models.** Not run yet. DeepSeek on the old 240-cell
   design did show the writing-down effect (+0.438, p<.001), which the 3B model
   could not. gpt-oss-20b and gpt-oss-120b have not been run on any Q1 grid.
4. **Human coding.** The blind coding page is built and waiting for a coder.
   This unblocks the mirroring measure's validity and is a dry run for the
   November round.
5. **Not built yet, and named in the research plan:** a contamination probe
   (did the model memorize Enron?) and an anonymized-stimulus arm. Both are
   cheap and both close an obvious examiner objection.

---

## 8. Working conventions that matter

**Writing style is a rule, not a preference.** Short sentences. One idea per
sentence. Plain words. Result first, then the reason. No rhetorical build-up,
no long comma chains, and no dashes bolted onto clauses. This applies to chat
replies, progress files, commit messages, code comments and docstrings. The
user has corrected this several times.

**Where things live.** Code is in WSL at `~/projects/thesis`, pushed to
`github.com/eunai9/llm-org-comm-thesis`. The OneDrive folder holds only
`Expose.pdf`, reference papers and `CLAUDE.md`. Reach the repo with
`wsl.exe -e bash -c "cd ~/projects/thesis && ..."`.

**The user reads results on GitHub, not locally.** Commit and push after every
result, including the code behind it. An unpushed result is invisible to them.

**Commit only your own work.** Other sessions often have uncommitted changes in
the same repo. Check `git diff` per file before staging.

**Quality gates before any commit:** `black`, `ruff`, `mypy`, and the full test
suite. All four must pass.

**One-off scripts are a recurring smell.** Several results were first produced
by a scratch script, then had to be rebuilt as a module so they could be
reproduced. Put settled logic in `src/thesis/` with an entry point, and write
the tests.

**A recurring bug class: re-runs overwriting earlier outputs.** Figures and
manifests were repeatedly written to one fixed filename, so a new run silently
replaced the numbers an already-written section cited. Every analysis entry
point now takes a `--figure-prefix` and a `--manifest` path. Keep doing that.

**Be honest about corrections.** Several sections in these logs record a number
being wrong and then fixed. That is the convention: never swap a number
quietly. Say what changed and why.

---

## 9. Open questions for the supervisor

1. Ethics approval for the November human-coding round: what is needed, what is
   the lead time?
2. Which models produce the final results? Departmental compute, research
   credit, or reframe the thesis around open-weight models?
3. How to present the power-score null. It does not track seniority, and
   splitting it into its two halves did not rescue it.
4. Who checks the 50-thread reconstruction sample: self-review or a second
   reader?
5. New: is Q1 worth about 14 more hours of generation for 80% power, given the
   answer would still come from one 3B model with 10 personas?
6. New: the Enron text goes to third-party servers (NVIDIA, Groq) on the free
   tiers. The corpus is public, but the supervisor should know.
