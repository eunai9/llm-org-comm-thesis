# LLM Comparison Progress Log

This file compares the four models used in the free-tier simulator work, one
topic at a time: mirroring, separability from real writing, and the Q1
hierarchy effect. Each section holds every model's numbers together, so they
can be read at once.

`PROGRESS_nvidia.md` still holds the day-by-day build log: what was tried,
what broke, and the exact commit history. This file only reorganizes its
numbers by topic. No number here is new. This work is still a trial. Nothing
here replaces a result in `PROGRESS.md` yet, except where a section says so
directly (the Q1 real-email comparison).

---

## Status at a glance

| | Llama 3.2 3B | DeepSeek V4 Flash | gpt-oss-20b | gpt-oss-120b |
|---|---|---|---|---|
| 183 real-email pairs generated | Yes (see `PROGRESS.md`) | Yes | Yes | Yes |
| Mirroring measured | Yes | Yes | Yes | Yes |
| Told apart from real writing (AUC) measured | Yes | Yes | Yes | Yes |
| Q1 grid (240 replies) generated | Yes | Yes | Not yet | Not yet |
| Q1 checked against the real-email benchmark | Not yet here (see `PROGRESS.md` section 48) | Yes | Not yet | Not yet |
| Hand-coded by a person | No | No | No | No |

Other open items: a run without the "act" instruction, an embedding map and
review pack for the newer models, a judge study across model families, and
committing the blind-coding module. See "Next steps" at the end.

---

## The models compared

| Model | Family | Reached through | Setting | Notes |
|---|---|---|---|---|
| Llama 3.2 3B | Meta | Local, on this laptop (Ollama) | none | 3B parameters. Runs on 8 GB of memory. |
| DeepSeek V4 Flash | DeepSeek | NVIDIA free tier | none | |
| gpt-oss-20b | OpenAI | NVIDIA free tier | `@low` reasoning effort | Without it, some replies run past the JSON until the token limit. |
| gpt-oss-120b | OpenAI | Groq free tier | `@low` reasoning effort | NVIDIA does not serve this model. This laptop cannot run it: it needs about 60 GB of memory, and the laptop has 16 GB plus 1 GB of graphics memory. |

All four answer the same 183 real email threads with the same prompt,
including the "act" instruction from `PROGRESS.md` section 43 (the persona
must act on a request, not hand it back). The two exceptions, kept for the
historical record, are noted where they appear: a Llama run made before that
instruction existed, and the original Q1 pilot from `PROGRESS.md` section 39.

---

## A few terms, explained once

- **Client**: the code that sends a prompt to a model service and reads the
  answer back.
- **Free tier**: the free level of NVIDIA's or Groq's hosted service. No
  payment is used anywhere in this project.
- **Reasoning effort**: a setting that tells a model how much it should
  "think" before answering. `gpt-oss` models need a low setting here, or they
  sometimes keep writing instead of stopping.
- **Cache**: the folder `runs/_cache`. It stores every model reply this
  project has received. A later run with the same prompt and model reads the
  saved reply instead of calling the model again.
- **Borrowed words**: the share of a reply's own distinct content words that
  already appear in the incoming email. 0 means none of them, 1 means all.
  Used for the mirroring measure.
- **AUC**: shown one real reply and one AI reply, how often a classifier
  ranks the real one as more likely real. 0.5 means pure guessing, 1 means
  always right. Used for the "can it be told apart" measure.
- **Imperative sentence**: a sentence that tells the reader to do something,
  such as "Send me the numbers by Friday." Used for the Q1 measure.
- **p-value**: the chance of seeing a result this large if there were really
  no effect. Below 0.05 is usually called significant.

---

## Why use free hosted models

The thesis will not pay for any LLM API. This was decided with the
supervisor. All generated replies before this work came from a small local
model, `llama3.2:3b`. This laptop has 8 GB of memory for its Linux system, so
about 8B parameters is the upper limit for anything running locally. A 3B
model cannot stand in for a named, citable frontier model in a results table.

NVIDIA's Developer Program and Groq both give free access to larger models
through an API. Using them keeps the no-payment decision and removes the
small-model problem.

Limits to keep in mind:

- NVIDIA's terms allow "testing and evaluation", not "production". NVIDIA's
  own forum says research counts as allowed use.
- Each prompt contains Enron text, and it goes to a third-party server. The
  corpus is public, but the supervisor should know this.
- Either service can remove a model or change its free tier at any time. The
  cache protects every reply already received.

---

## Clients, limits, and pacing

**Which NVIDIA models answer at all.** The catalog lists 82 models. Each
candidate got one tiny request.

| Model | Result |
|---|---|
| `nvidia/nemotron-3-super-120b-a12b` | Answers |
| `deepseek-ai/deepseek-v4-flash-0731` | Answers |
| `openai/gpt-oss-20b` | Answers |
| `google/gemma-4-31b-it` | No answer within 120 s |
| `mistralai/mistral-nemotron` | No answer within 120 s |
| `meta/llama-3.2-90b-vision-instruct` | No answer within 120 s |
| `mistralai/mistral-large-2-instruct` | Not served (error 404) |
| `nvidia/llama-3.1-nemotron-70b-instruct` | Not served (error 404) |

DeepSeek V4 Flash was picked as the main reply writer: fastest, shortest
valid output, and a well-known model.

**The reasoning-effort mechanism.** A model name may end in `@low`, `@medium`
or `@high`, for example `openai/gpt-oss-20b@low`. The client sends the plain
name to the service and adds the setting to the request. The full name stays
on every saved reply and in the cache key, so replies made with different
settings never mix.

**Rate limits and pacing.**

| | NVIDIA free tier | Groq free tier |
|---|---|---|
| Requests per minute | about 40 | 30 |
| Tokens per minute | not published | 8,000 |
| Requests per day | not published | 1,000 |
| Tokens per day | not published | 200,000 |
| Client's gap between calls | 1.5 s | 15 s |
| Timeout per try | 120 s | 120 s |
| Retries | 5 | 5 |

Groq's token-per-minute cap binds before its request cap. A reply of this
project costs about 1,930 tokens, so only about 4 fit in a minute. A first
test at 2 seconds per call drew two 429 errors; 15 seconds keeps it inside
the token pace. NVIDIA often holds a request without answering rather than
refusing it outright: in one 8-reply test, 6 requests stalled and every
retry worked. The timeout was cut from an original 300 s to 120 s, because a
stuck request at 300 s with 5 retries could block a run for up to 30 minutes.

**One client body for both services.** NVIDIA and Groq both speak the same,
OpenAI-compatible request format. The shared parts, request shape, spacing,
retries, and response mapping, live in one module, and each service's client
is a thin subclass of it. Two things differ: Groq's schema request uses
strict mode, which guarantees the reply matches the shape exactly, while
NVIDIA's does not (this is why `gpt-oss` needed `@low` there); and Groq's
request drops the model's own reasoning text, which this project never reads.

**Running long jobs safely.** A run of several hours can die for reasons
unrelated to the code.

- The laptop must stay on and awake. Everything runs inside WSL on this
  machine; the model runs on a remote server, but the program that sends
  requests and saves replies runs here. Sleep was set to "Never" while
  plugged in.
- A run started from a Claude Code session is attached to that session. If
  the session or the editor closes, the run can die with it. Long runs are
  started in the user's own terminal window, inside `tmux`, with the window
  left open.
- Every finished reply is saved to the cache immediately. If a run stops for
  any reason, the same command continues where it left off. Only time is
  lost, never a result.

---

## Generation results: speed, reliability, cost

Llama's full 183-pair run predates this log; its numbers are in
`PROGRESS.md`. For the three free-tier models:

| | DeepSeek V4 Flash | gpt-oss-20b | gpt-oss-120b |
|---|---:|---:|---:|
| Pairs written | 183 of 183 | 183 of 183 | 183 of 183 |
| Time to generate | about 6.5 hours, 3 attempts (stalls forced 2 restarts) | 376 seconds | 34 minutes |
| Empty replies | 0 | 0 | 0 |
| Replies with a placeholder such as `[Name]` | not measured | 3 | 7 |
| Decisions | 92 accept, 61 none, 30 defer | 136 accept, 33 defer, 9 decline, 5 none | 102 accept, 53 defer, 24 none, 4 decline |
| Median length | 38 words | 30 words | 27 words |
| Cost | $0 | $0 | $0 |

Real replies have a median length of 44 words. The three free-tier models
are all shorter, and the OpenAI models are shorter still than DeepSeek.

For gpt-oss-120b's run, the daily token cap never bit: 129 new replies at
about 1,930 tokens each is roughly 249,000 tokens, above the stated 200,000
per day. Either the cap is looser than documented, or it is counted
differently than the request pace suggests.

---

## Mirroring: how much does a reply copy the sender's own words

A reply mirrors when it is built mostly from the sender's own words, often
handing a request back to the sender instead of acting on it. This was the
main failure found in the small local model (`PROGRESS.md` section 42). The
question here is whether larger models do it too.

**How it is measured.**

- **Borrowed words**: the share of a reply's own distinct content words that
  already appear in the incoming email.
- **Flagged**: a reply with borrowed words of 0.80 or more. This cutoff was
  chosen by looking at 100 coded Llama replies. Those codes are Claude's
  first pass, not a person's; see the caveat below.
- **Length matters**: borrowed words is a share of a reply's own distinct
  vocabulary. A longer reply has more room for words the sender never used,
  so it scores lower even if it copies just as much. Two versions of the
  table follow: replies as written, and every reply cut to the same length
  before scoring.

**At full length, as each model actually wrote it.**

| Model | Mean borrowed words | Flagged | Mean length |
|---|---:|---:|---:|
| Llama 3.2 3B, no instruction | 0.579 | 25.7% | 19.8 words |
| Llama 3.2 3B, with instruction | 0.565 | 19.1% | 22.7 words |
| DeepSeek V4 Flash | 0.413 | 1.1% | 40.1 words |
| gpt-oss-20b | 0.322 | 1.1% | 32.8 words |
| gpt-oss-120b | 0.405 | 2.2% | 28.6 words |
| Real replies | 0.301 | | 65.0 words |

At full length, gpt-oss-20b sits closest to real replies. But this ranking
mixes copying with length. All three free-tier models write longer than
Llama and shorter than real people, so length alone moves this table.

**At the same length**, every reply cut to the length of its Llama
(with-instruction) partner, which removes that length effect.

| | Mean borrowed words | Flagged |
|---|---:|---:|
| Llama 3.2 3B | 0.565 | 19.1% |
| DeepSeek V4 Flash | 0.469 | 8.2% |
| gpt-oss-20b | 0.373 | 4.9% |
| gpt-oss-120b | 0.435 | 4.4% |
| Real replies | 0.294 | 9.8% |

![Mean borrowed words by model, every reply cut to the Llama length, next to the real replies.](docs/figures/nvidia_four_models_mirroring.png)

![How much of a reply is built from the sender's own words, DeepSeek against real replies.](docs/figures/mirroring_deepseek_generated_vs_real.png)

**Paired tests, each free-tier model against Llama at the same length.**

| Model | Score change | p-value | Newly flagged | No longer flagged | p-value (flag test) |
|---|---:|---:|---:|---:|---:|
| DeepSeek V4 Flash | -0.095 | < 0.0001 | 12 | 32 | 0.004 |
| gpt-oss-20b | -0.191 | < 0.0001 | 7 | 33 | < 0.0001 |
| gpt-oss-120b | -0.129 | < 0.0001 | 3 | 30 | < 0.0001 |

The score test (Wilcoxon signed-rank) asks whether the borrowed-words scores
moved. The flag test (McNemar) counts only the replies whose flagged status
changed. All three free-tier models mirror significantly less than Llama.
Size did not help within the OpenAI family: gpt-oss-120b borrows more of the
sender's words than gpt-oss-20b, 0.435 against 0.373, though it is flagged
slightly less often.

**What this means.** Mirroring looks mainly like a weakness of the small
local model, not something all LLMs do. All four models still borrow more
than real writers, 0.294, so none of them fully matches human behavior.

**Caveats.**

- The 0.80 cutoff and the AUC check behind it come from 100 Llama replies
  coded by Claude, not by a person. No person has coded any reply yet.
  Claude read the 5 highest-scoring DeepSeek replies by hand; none of them
  actually hands the request back, which suggests a high score may mean
  "stays on topic" rather than "mirrors" for the larger models. This needs a
  person's judgment to settle.
- The measure and its cutoff were built and validated on Llama only.
  Applying the same cutoff to a different model's writing style is an
  assumption, not something separately checked.

---

## Can a generated reply be told apart from a real one

**How it is measured.**

- **Length-only classifier**: sees only each reply's word count.
- **Text classifier**: sees which words a reply uses (TF-IDF features with
  logistic regression), cross-validated over 5 rounds.
- **Same length**: each real reply is cut to the length of its AI partner
  before the text classifier runs, so whatever separation remains cannot be
  explained by length.
- Before comparing, signature and address lines were removed from real
  replies (routing addresses, company names, street addresses, phone lines,
  and lines that are mostly email addresses). This changed the numbers only
  a little; it did not change any conclusion.

**Combined AUC table**, all rows measured with the same prompt (the
with-instruction Llama grid).

| Model | Length only | Full text | Text, same length |
|---|---:|---:|---:|
| Llama 3.2 3B | 0.837 | 0.919 | 0.881 |
| DeepSeek V4 Flash | 0.571 | 0.978 | 0.977 |
| gpt-oss-20b | 0.687 | 0.974 | 0.970 |
| gpt-oss-120b | 0.748 | 0.969 | 0.962 |

For reference, the earlier, no-instruction Llama grid gave 0.879 (length
only), 0.903 (full text), 0.841 (same length).

![Length only, full text and text at the same length, for the four models.](docs/figures/nvidia_four_models_auc.png)

- **Length alone barely gives DeepSeek away** (0.571), a little more for
  gpt-oss-20b (0.687), and more again for gpt-oss-120b (0.748), because its
  replies are the shortest of the three at a 27-word median.
- **Length still gives Llama away** (0.837), because its replies are much
  shorter than real ones.
- **Every free-tier model is still easy to spot by its words**, at 0.96 to
  0.98 even at the same length. Being bigger did not make gpt-oss-120b
  harder to tell from a real person than gpt-oss-20b.

**Which words give each model away.** Share of replies containing the word,
model against real replies (real percentages vary slightly by row, because
real replies are cut to each model's own length for that comparison). "n/a"
means the word was not among the top words reported for that model.

| Word | Real | Llama | DeepSeek | gpt-oss-20b | gpt-oss-120b |
|---|---:|---:|---:|---:|---:|
| "I'll" | 5 to 6% | 29% | 74% | 59% | 61% |
| "let" | 9 to 10% | n/a | 45% | 53% | 38% |
| "know" | 12 to 13% | n/a | 42% | 50% | n/a |
| "review" | 2 to 3% | n/a | 29% | n/a | 23% |
| "confirm" | 0 to 1% | n/a | 19% | 13% | n/a |
| "got" | 1 to 2% | n/a | 11% | 18% | 22% |
| "team" | 0% | n/a | n/a | 14% | 12% |
| "forward" | 3% | n/a | n/a | n/a | 19% |

These come from stock phrases such as "I'll review...", "Let me know..." and
"I'll flag...". Real replies are instead marked by sign-off names and words
like "attached", "department" and "office". Real people sign with their own
name and mention attachments; the models rarely do.

**What this means.** For Llama, "real and AI replies are separable, mostly
by length" was true. For all three free-tier models, it is not: they write
closer to a realistic length, and are still easy to spot by word choice.
Model size (20B against 120B) moved neither the mirroring result nor this
one.

**Caveats.**

- The line-removal rule that cleans signature and header lines is a one-off
  script. The corpus cleaner in `thesis.data.rfc822` does not use it yet.
- One number is unexplained: the full-text AUC for Llama without the
  instruction is 0.903 by this method, while `PROGRESS.md` reports 0.927
  by an earlier one. The length-only numbers match, so the method is
  probably the same; the gap is unresolved.

---

## Does hierarchy change how directive a reply is (Q1)

Q1 asks whether a persona's place in the hierarchy changes how directive its
replies are, that is, whether it gives more orders when writing down to a
junior person than when writing to a peer or up to someone senior. Every
Llama-only test before this found no clear effect (`PROGRESS.md` sections 33
to 39), with one borderline exception. The open question is whether a larger
model shows a pattern the small one does not.

**How it is measured.** The Q1 grid is 10 personas times 3 directions times
4 incoming tones times 2 task types, one reply each, 240 replies total. This
is the design fixed in `PROGRESS.md` section 39.

- **Reply level**: the share of a reply's sentences that are imperative,
  compared across directions with a linear mixed model that allows for each
  persona's own habits.
- **Sentence level**: each sentence counts once, as an order or not,
  compared with a logistic mixed model. Its coefficients are also turned
  into the probability that a sentence is an order.
- Every coefficient below is the difference from writing to a peer. Positive
  means more imperative.

**Results**, all three grids using the same measurement code.

| Grid | Reply level, down | Reply level, up | Sentence level, down | Sentence level, up |
|---|---:|---:|---:|---:|
| Llama, no instruction (original pilot) | +0.027 (p=.672) | +0.083 (p=.192) | +0.163 (p=.401) | +0.395 (p=.046) |
| Llama, with instruction | +0.056 (p=.412) | +0.029 (p=.670) | +0.197 (p=.298) | +0.151 (p=.437) |
| DeepSeek V4 Flash | +0.099 (p=.003) | +0.015 (p=.647) | +0.438 (p<.001) | +0.059 (p=.654) |

Probability that a sentence is an order, from the sentence-level model:

| Grid | Writing down | Writing to a peer | Writing up |
|---|---:|---:|---:|
| Llama, no instruction | 31.7% | 28.3% | 36.9% |
| Llama, with instruction | 40.7% | 36.0% | 39.6% |
| DeepSeek V4 Flash | 33.5% | 24.5% | 25.6% |

![Predicted probability of an imperative sentence by direction, for Llama and DeepSeek with the same prompt.](docs/figures/nvidia_q1_direction.png)

- **DeepSeek gives more orders when writing down.** Both measures agree, and
  both stay significant after a Bonferroni correction for four tests
  (needs p < 0.0125): 0.003 and 0.0004.
- **DeepSeek's writing up looks like writing to a peer.** Both contrasts are
  far from significant.
- **Llama shows no reliable effect in either direction**, with either prompt.
- **The one borderline Llama result did not hold.** With the act
  instruction, its "up" contrast fell from p=.046 to p=.437, which reads as
  noise rather than a real effect.
- **DeepSeek's replies carry more sentences**: 882 across the grid against
  343 for Llama with the same prompt, 3.71 sentences per reply against 1.43.
  `PROGRESS.md` section 34 warned that one-sentence replies make the
  reply-level measure crude. DeepSeek has none, so its estimate rests on
  more data.
- **Two side results.** The decision depends on direction for both models
  (Llama p=.023, DeepSeek p=.002), but this plain chi-square test ignores
  that replies from one persona are alike, so it is only a hint. Hedging
  falls for both models when writing down, but DeepSeek almost never hedges
  at all (at most 0.014 in any direction, against 0.08 to 0.19 for Llama).

**Against the real-email benchmark.** `PROGRESS.md` section 48 measured the
same three things in 2,202 real Enron emails, and compared them only with
the Llama grid. This log adds the DeepSeek side of that comparison, using a
rough z-test on the difference (rough because both sides' standard errors
are backed out of a coefficient and a p-value).

| Contrast | DeepSeek | Real email | Difference | p |
|---|---:|---:|---:|---:|
| Orders per email, down | +0.098 (p=.003) | +0.043 (p=.002) | +0.056 | .125 |
| Orders per email, up | +0.016 (p=.640) | +0.018 (p=.133) | -0.002 | .955 |
| Orders per sentence, down | +0.439 (p<.001) | +0.253 (p<.001) | +0.185 | .169 |
| Orders per sentence, up | +0.059 (p=.654) | +0.070 (p=.096) | -0.011 | .936 |
| Hedges per email, down | -0.013 (p=.040) | -0.005 (p=.526) | -0.009 | .362 |
| Hedges per email, up | -0.007 (p=.274) | -0.005 (p=.429) | -0.002 | .809 |

| Chance a sentence is an order | Writing down | To a peer | Writing up |
|---|---:|---:|---:|
| Real email | 17.5% | 14.2% | 15.0% |
| DeepSeek | 33.5% | 24.5% | 25.6% |

![Chance that a sentence gives an order, by direction, for real email and DeepSeek.](docs/figures/nvidia_q1_deepseek_vs_real.png)

- **DeepSeek gets the shape right.** Writing down is highest on both lines,
  and writing up sits close to writing to a peer on both.
- **DeepSeek's swing is bigger, but not provably so.** Down minus peer is
  +9.0 points in DeepSeek against +3.3 points in real email. No contrast
  between DeepSeek and real email is significant.
- **Llama sits near the real effect size but detects nothing.** Its down
  effect is +0.198 against the real +0.253 (p=.298), because its 240 short
  replies are too few to detect an effect this size. DeepSeek finds it
  because its replies carry far more sentences.

**Caveats.**

- One reply per cell, 10 personas. In the reply-level model the persona
  variance is 0 for both grids, so the model works like a plain regression;
  the fit warns it sits at the edge of its range.
- The sentence-level p-values are approximate (a Wald test from a
  variational Bayes fit).
- The DeepSeek grid has 238 replies, not 240: 2 came back without valid
  JSON, one writing up and one writing down.
- Real email and the simulator are not the same kind of sample: 2,202 real
  emails from 107 senders against 238 simulator replies from 10 personas,
  one reply each. Real email has no decision field, so decisions cannot be
  compared. The z-test is rough for the reason stated above.
- gpt-oss-20b and gpt-oss-120b have not been run on the Q1 grid yet, so this
  section cannot yet say whether the DeepSeek pattern is typical of larger
  models or specific to DeepSeek.

---

## Where the code lives

**The clients.** `src/thesis/llm/openai_compatible.py` holds the shared
request and retry logic. `nvidia_client.py` and `groq_client.py` are thin
subclasses. `base.py` lists `"nvidia"` and `"groq"` as providers, and
`sim/run.py` never bills either one. `.env.example` documents both key
names; the real keys live only in `.env`, which git ignores.

**The mirroring fix.** `analysis/mirroring.py`'s `pair_key` matches two runs
by persona and message rather than by the model's role label, so two
different models answering the same email can be compared at all.
`compare_runs` also runs the same-length comparison and saves it in the
manifest.

**The Q1 backend.** `analysis/q1.py` takes `--local MODEL`, `--nvidia MODEL`,
or `--grid PATH` to analyse an existing file without calling any model, plus
`--compare-real` to print the comparison against the real-email benchmark in
one command. `analysis/q1_real.py`'s `implied_se` is public so both modules
share one formula.

Exact commits and the day-by-day build order, including two corrections
made along the way, are in `PROGRESS_nvidia.md`.

---

## Next steps

Most valuable first.

1. **Run the Q1 grid with gpt-oss-20b and gpt-oss-120b.** This would show
   whether the DeepSeek writing-down effect is typical of larger models, or
   specific to DeepSeek. Both models generate fast enough that this takes
   minutes, not hours, and each can then be checked against the real-email
   benchmark with `--compare-real`.
2. **Hand-code a sample of replies by a person.** The mirroring cutoff and
   its validation rest on Claude's first-pass codes of Llama replies only.
   A coding page is already built: 50 emails, two replies each from two
   models, in random order, with the model hidden. It is waiting for a
   coder. One small fix is needed first: the first pass used a label,
   `wrong_register`, that the codebook never defined.
3. **A run without the act instruction.** This would show whether a larger
   model follows the instruction better than the small local one did
   (`PROGRESS.md` section 43 found only a small effect on Llama). The code
   already has a `--prompt-variant` mechanism; a third variant without the
   instruction would fit it.
4. **Embedding map and review pack on the newer models.** These only read
   saved replies, so they need no new generation.
5. **Judge study (Q3).** Four models across three families are now
   available. One can write and another can judge, then the roles can be
   swapped. This needs new model calls.
6. **Move the line-removal rule into the corpus cleaner**, so every analysis
   uses the same clean real-reply text instead of a one-off script.
7. **Rename one summary key.** `mirroring.py` saves its comparison under
   `compared_with_previous_prompt`, which is the wrong name for a comparison
   between models.
8. **Back up `runs/_cache`.** It holds every reply this project has
   received and exists only on this laptop. It must never go into git,
   because the prompts contain Enron text.
9. **Commit the rest of the code.** Still uncommitted: the blind-coding
   module and its tests, and the codebook fix that defines `wrong_register`.
