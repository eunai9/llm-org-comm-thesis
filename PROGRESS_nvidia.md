# NVIDIA Free-Tier Progress Log

A plain-language record of the work that uses NVIDIA's free hosted models.
It is kept apart from `PROGRESS.md` on purpose. This work is still a trial.
Nothing here replaces a result in `PROGRESS.md` yet.

---

## Status at a glance

| Stage | Status |
|---|---|
| Decide whether the NVIDIA free tier fits the thesis | Done, see section 1 |
| Build a client for the NVIDIA API | Done, see section 2 |
| Find which models work on this account | Done, see section 3 |
| Test run: 10 replies | Done, see section 4 |
| Full run: all 183 reply pairs with DeepSeek | Done, see section 6 |
| Mirroring measure on the DeepSeek replies | Done, see section 8 |
| Can DeepSeek replies be told apart by length or words? | Done, see section 9 |
| Hand-code a sample of DeepSeek replies | Not started |
| DeepSeek run without the act instruction | Not started |
| Judge study with a second model family | Not started |
| Commit the client code | Done (Sep 13) |
| Back up the reply cache | Not done yet |

---

## A few terms, explained once

- **NVIDIA build (build.nvidia.com)**: a website where NVIDIA runs many
  open-weight models on its own servers. Anyone in the free NVIDIA Developer
  Program can call them through an API key.
- **Client**: the piece of code that sends a prompt to a model service and
  reads the answer back.
- **Timeout**: how long the client waits for an answer before it gives up
  on one try.
- **Retry**: sending the same request again after a failed try.
- **Cache**: the folder `runs/_cache`. It stores every model reply this
  project has received. A later run with the exact same prompt and model
  reads the saved reply instead of calling the model again.
- **Valid JSON**: the reply arrives in the fixed structure the analysis
  expects (subject, body, decision, confidence, short reasoning). An invalid
  reply cannot be used.
- **Median**: the middle value when all values are sorted. Half the values
  are below it and half are above it.

---

## 1. Why try the NVIDIA free tier (Sep 12)

**The problem it solves.** All generated replies so far come from small
local models such as `llama3.2:3b`. This laptop's Linux system has 8 GB of
memory, so about 8B parameters is the upper limit. A 3B model on a laptop
cannot stand in for a named, citable model in a results table. The code
itself says so in `ollama_client.py`.

**Why NVIDIA.** The thesis will not pay for any LLM API. This was decided
with the supervisor. NVIDIA's free tier costs nothing and offers much larger
models. So it keeps the budget decision and removes the small-model problem.

**What the free tier offers.**

- It serves 82 models through an API in the same format as OpenAI's.
- The address is `https://integrate.api.nvidia.com/v1`.
- The limit is about 40 requests per minute.

**Limits to keep in mind.**

- NVIDIA's terms allow "testing and evaluation", not "production". NVIDIA's
  forum says research counts as allowed use.
- Each prompt contains Enron text, and it goes to NVIDIA's servers. The
  corpus is public, but the supervisor should know about this.
- NVIDIA can remove a model at any time. The cache protects replies that
  were already received.

---

## 2. Building the client (Sep 12)

**Why this step.** The project had clients for Anthropic and for local
Ollama models only. A new client was needed before any NVIDIA model could
be called.

**What was built.**

- `src/thesis/llm/nvidia_client.py`: the new client. It asks the model for
  valid JSON in the same structure as the other clients.
- `src/thesis/analysis/pairs.py`: a new option, `--nvidia MODEL`. It
  generates the replies with an NVIDIA model.
- `src/thesis/sim/run.py`: NVIDIA replies are never priced. Before this
  change, the run would have crashed on the missing price.
- `src/thesis/llm/base.py`: `"nvidia"` added to the list of providers.
- `.env.example`: a new line for `NVIDIA_API_KEY`. The real key goes in
  `.env`, which git ignores.
- `tests/test_nvidia.py`: 13 new tests. They need no network and no key.

**How NVIDIA replies are marked.** Each reply is saved with the model name
`nim/<model>`, for example `nim/deepseek-ai/deepseek-v4-flash-0731`. This
keeps NVIDIA rows easy to find. The cost ledger records them at $0.

**How the client handles the rate limit and failures.**

| Setting | Value | Reason |
|---|---:|---|
| Gap between calls | 1.5 s | 40 requests per minute is one call every 1.5 s |
| Timeout per try | 120 s | A normal reply takes under 15 s. Changed from 300 s, see section 5 |
| Retries | 5 | So one request gets 6 tries in total |
| Wait before each retry | 2, 4, 8, 16, 32 s | The wait doubles each time |

**Checks.** black, ruff, mypy and the full test suite all pass.

---

## 3. Which models work on this account (Sep 12)

**Why this step.** The public list shows 82 models. Not all of them answer
on a free account. Each candidate got one tiny request.

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

**Which of them return valid JSON.** The three working models then got one
made-up email and the real reply schema.

| Model | Valid JSON | Time | Output tokens |
|---|---|---:|---:|
| `deepseek-v4-flash-0731` | Yes | 11.6 s | 213 |
| `nemotron-3-super-120b-a12b` | Yes | not recorded | 548 |
| `gpt-oss-20b` | No | 73.9 s | 2,048 |

`gpt-oss-20b` kept writing past the JSON until it hit the 2,048-token limit.

**Choice.** DeepSeek V4 Flash is used as the reply writer. It is the
fastest, it gives the shortest valid output, and it is a well-known model.
Nemotron is kept as the backup.

---

## 4. Test run: 10 replies (Sep 12)

**Why this step.** One made-up email does not show how the model handles
the real prompts. Ten real pairs are a cheap check before a run of several
hours.

**Command.**

```
python -m thesis.analysis.pairs --nvidia deepseek-ai/deepseek-v4-flash-0731 --limit 10 --out <test file>
```

The test output was written outside the repository.

**Results.**

| | Real reply | DeepSeek | Llama 3.2 3B (local) |
|---|---:|---:|---:|
| Median length (words) | 32.5 | 40.5 | 17 |

- Valid JSON: 10 of 10.
- Decisions: 8 accept, 1 defer, 1 none.
- Placeholders such as `[Manager's Name]`: 0 of 10. The replies use the real
  names from the incoming email.
- The replies read like fluent business email. They often invent details
  that the real replies do not contain.

DeepSeek's length is much closer to the real replies than Llama's.

---

## 5. The free tier stalls often (Sep 12)

**Why this matters.** Speed decides whether a full run is practical.

**What happened.** NVIDIA often holds a request without answering. In the
10-reply test, 6 of the 8 new requests stalled at least once. Every retry
worked in the end.

| Seconds per reply (8 new replies) |
|---|
| 33, 47, 142, 149, 154, 163, 274, 317 |

- Only 2 replies came back on the first try.
- The median is about 150 s per reply.
- The 8 replies took 21 minutes together.

**Fix.** The timeout was cut from 300 s to 120 s. With 300 s and 5 retries,
one stuck request could block a run for up to 30 minutes. The first test
attempt was stuck for 9 minutes this way.

---

## 6. Full run: all 183 reply pairs (Sep 12)

**Why this step.** The test showed valid output. The analyses need the full
set of 183 pairs, the same set used for Llama in `PROGRESS.md`.

**How it was run.** The run takes several hours, so it was started in the
user's own Ubuntu window inside `tmux`, not from a Claude session. See
section 7 for why.

**First attempt.** It started at 17:21. It finished 43 pairs and then
stopped at about 18:40. One request timed out 6 times in a row, so the
client gave up and the run stopped.

**Nothing was lost.** The 43 finished pairs were in the cache. They needed
only 35 distinct replies. Some pairs share the exact same prompt: the same
persona answers the same incoming email, because the thread has two real
replies. Those pairs share one saved reply.

**Restart loop.** The run was restarted inside a loop. When it stops with an
error, the loop waits 60 seconds and starts it again. A restart skips every
saved reply within seconds.

```
cd ~/projects/thesis && set -a && source .env && set +a
for i in $(seq 1 30); do
  .venv/bin/python -m thesis.analysis.pairs --nvidia deepseek-ai/deepseek-v4-flash-0731 \
    --out data/interim/pairs_deepseek.parquet 2>&1 | tee -a runs/nvidia_full.log
  [ "${PIPESTATUS[0]}" -eq 0 ] && break
  echo "restart $i"; sleep 60
done
```

**Result.** The run finished at 23:49. It stopped 3 times in total, the
first attempt included.

| | Value |
|---|---:|
| Pairs written | 183 of 183 |
| Empty replies | 0 |
| Decisions | 92 accept, 61 none, 30 defer |
| Median length, real replies (words) | 44 |
| Median length, DeepSeek replies (words) | 38 |
| Cost | $0 |

**Where the output is.**

- `data/interim/pairs_deepseek.parquet`: the 183 pairs. Git ignores it.
- `runs/nvidia_full.log`: the run log. Git ignores it.
- The Llama pairs file `data/interim/real_vs_generated_pairs.parquet` was
  not touched.

---

## 7. Running long jobs safely (Sep 12)

**Why this step.** A run of several hours can die for reasons unrelated to
the code. Two risks were checked.

- **The laptop goes to sleep.** Windows was set to sleep after 30 minutes
  without use, even on the charger. Sleep stops every run. It was set to
  "Never" while plugged in.
- **WSL shuts down.** A run started from a Claude session is attached to
  that session. If the session or VS Code closes, WSL may shut down and
  kill the run. So long runs are started in the user's own Ubuntu window,
  inside `tmux`, with the window left open.

A test with a detached process showed that WSL stayed up while the user's
Ubuntu window was open. It does not show what happens with no window open.

**The safety net.** Every finished reply is saved to the cache at once. If a
run is killed for any reason, the same command continues where it stopped.
Only time is lost.

---

## 8. Mirroring measure on the DeepSeek replies (Sep 13)

**Result first.** With the same prompt, DeepSeek mirrors less than Llama,
even when both replies have the same length. Length explains about one third
of the gap. DeepSeek still takes more of the sender's words than real people
do. Claude read its highest-scoring replies, and none of them hands the
request back. No person has checked this yet.

**Correction.** The first version of this section compared DeepSeek with the
wrong Llama replies. Since section 43 of `PROGRESS.md` (Sep 5), the persona
prompt tells the persona to act on a request, not hand it back. This "act"
instruction is part of the default prompt, so the DeepSeek run used it. The
first version compared DeepSeek with Llama replies made before the
instruction existed. That comparison changed two things at once: the model
and the prompt. This version compares DeepSeek with the Llama replies made
with the same instruction (`real_vs_generated_pairs_act.parquet`). The
conclusion did not change. The numbers changed a little.

**Why this step.** Mirroring was the main failure of the local Llama model
(`PROGRESS.md` section 42). A reply mirrors when it is built mostly from the
sender's own words, often handing the request back to the sender. The
question is whether DeepSeek does it too. If not, mirroring is a weakness of
the small model. If yes, it is a general LLM behaviour. The measure only
reads the saved replies, so it needs no new generation.

**How mirroring is measured.**

- **Borrowed words**: the share of a reply's distinct content words that
  already appear in the incoming email. 0 means none of them, 1 means all
  of them.
- **Flagged**: a reply with borrowed words of 0.80 or more. Section 42 chose
  this cut-off by looking at 100 coded Llama replies. Those codes are
  Claude's first pass from section 35, not a person's.

**A code fix first.** `compare_runs` in `mirroring.py` compares two runs
reply by reply: the same persona answering the same email. It matched the
runs on the full `cell_id`. That id begins with the role label, which is
`sim_local` for Llama and `sim_nvidia` for DeepSeek. So Llama and DeepSeek
matched 0 pairs. It now ignores the role label, and all 183 pairs match. A
new test covers this. This code change is not committed yet.

**Same inputs.** In all 183 matched pairs, the incoming email, the real
reply, the persona and the direction are identical. Both runs used the same
prompt, including the act instruction. Only the model differs.

**First result.**

| | Mean borrowed words | Flagged | Mean length |
|---|---:|---:|---:|
| Llama 3.2 3B, with the instruction | 0.565 | 19.1% | 22.7 words |
| DeepSeek V4 Flash, with the instruction | 0.413 | 1.1% | 40.1 words |
| Real replies, cut to DeepSeek's length | 0.277 | 9.8% | |

![How much of a reply is built from the sender's own words, DeepSeek against real replies.](docs/figures/mirroring_deepseek_generated_vs_real.png)

The top panel is DeepSeek ("AI replies" in the figure). The bottom panel is
the real replies, each cut to the length of its DeepSeek partner. Almost no
DeepSeek reply reaches 0.80.

**Length check.** DeepSeek's replies are almost twice as long as Llama's.
Borrowed words is a share of a reply's distinct words. A longer reply has
more room for words the sender never used, so it scores lower even if it
copies just as much. So each DeepSeek reply was cut to the length of its
Llama partner and scored again.

| | Mean borrowed words | Flagged | Mean length |
|---|---:|---:|---:|
| Llama 3.2 3B, with the instruction | 0.565 | 19.1% | 22.7 words |
| DeepSeek, full reply | 0.413 | 1.1% | 40.1 words |
| DeepSeek, cut to Llama's length | 0.469 | 8.2% | 22.3 words |
| Real replies, cut to Llama's length | 0.294 | 9.8% | 21.4 words |

Two paired tests compare each DeepSeek reply with the Llama reply to the
same email:

- **The score test** (Wilcoxon signed-rank) asks whether the scores moved.
  At full length, DeepSeek scores 0.151 lower on average. At the same
  length, it scores 0.095 lower. Both p < 0.0001.
- **The flag test** (McNemar) counts only the replies whose flag changed. At
  the same length, 32 replies stop being flagged and 12 become flagged.
  p = 0.004.

What the table shows:

- **The gap is not only length.** The full gap is 0.151. After the cut,
  0.095 remains. So length explains about one third of it.
- **The flagged rate depends mostly on length.** At full length, 1.1% of
  DeepSeek replies are flagged. At Llama's length it is 8.2%, close to the
  9.8% for real replies.
- **DeepSeek still borrows more than people do.** At the same length its
  mean is 0.469, against 0.294 for real replies.

The length check numbers come from a one-off script. The pipeline does not
save them yet.

**Reading the top replies.** Claude read the 5 highest-scoring DeepSeek
replies. No person has checked them yet. None of them hands the request back. The top one (0.93) gives
a clear instruction, using the sender's names for the deal. Others make the
confirmation the sender asked for, or acknowledge the message and say what
happens next. They score high because they reuse the sender's topic words,
such as names, deals and forms. This is one reader, an AI, and 5 replies.
The measure was checked against codes of Llama replies only, and those codes
are Claude's first pass from section 35, not a person's. So for
DeepSeek, a high score may mean "stays on topic" rather than "mirrors".

**By direction.** Only 2 DeepSeek replies are flagged, both writing down to
a junior. A split by direction says nothing with 2 replies, so its figure is
left out.

**What this means.** Mirroring looks mainly like a weakness of the small
model, not a general LLM behaviour. Two checks are still missing:

- **Hand-coding a sample of DeepSeek replies by a person.** Section 35 coded
  100 Llama replies, but Claude did that first pass. No person has coded any
  reply yet. Codes from a person would show whether the measure means the
  same thing for both models.
- **A DeepSeek run without the act instruction.** Section 43 found that the
  instruction barely changed Llama's mirroring. It left open whether a
  larger model follows it better. There is no DeepSeek run without the
  instruction, so this section cannot say how much the instruction helps
  DeepSeek.

**Command.**

```
python -m thesis.analysis.mirroring --pairs data/interim/pairs_deepseek.parquet \
  --out outputs/tables/mirroring_scores_deepseek.csv \
  --figure-prefix mirroring_deepseek_ \
  --manifest outputs/manifests/mirroring_deepseek.json \
  --compare-to data/interim/real_vs_generated_pairs_act.parquet
```

The Llama results and figures were not touched.

---

## 9. Can DeepSeek replies be told apart from real ones? (Sep 13)

**Result first.** Length alone no longer gives DeepSeek away. A classifier
that sees only word counts separates DeepSeek from real replies at 0.57,
close to guessing. For Llama with the same prompt it is 0.84. But DeepSeek's
words give it away almost perfectly, at 0.98. This does not come from
leftover signature lines in the real emails. It comes from DeepSeek's own
stock phrases. For example, it writes "I'll" in 74% of its replies, against
6% of real replies.

**Why this step.** `PROGRESS.md` found that Llama replies are easy to tell
from real ones, and that length explains most of it. DeepSeek writes much
closer to real length: 38 words against 44 at the median. So the question
was whether length still separates them, and if not, whether anything else
does.

**How separation is measured.**

- **AUC**: shown one real and one AI reply, how often the classifier ranks
  the real one as more likely real. 0.5 means pure guessing. 1 means always
  right.
- **Length-only classifier**: sees only each reply's word count.
- **Text classifier**: sees which words a reply uses. It is the model-free
  classifier from `fidelity.py` (TF-IDF word features with logistic
  regression).
- **Cross-validated**: each classifier is trained on part of the pairs and
  scored on pairs it has not seen, in 5 rounds.
- **Same length**: each real reply is cut to the length of its AI partner.
  Whatever separation remains cannot be length.

The length-only result for Llama without the instruction is 0.910. That
matches the 0.908 in `PROGRESS.md`, so the method is the same.

**Leftover lines in the real replies.** In the first text run, many words
pointing to "real" were not writing at all: Enron, Corp, North America,
Houston, 1400, Smith, ECT. They come from signature blocks ("Enron North
America Corp.", "1400 Smith Street", "Houston, Texas 77002") and from copied
email headers ("To: Kim Ward/HOU/ECT@ECT"). The corpus cleaner does not
remove these lines.

So a one-off rule removed them. It removes a line when the line:

- starts with To:, cc:, bcc:, Subject:, Sent by: or From:
- contains an Enron routing address, such as /HOU/ or @ECT
- holds only an Enron company name
- contains the street address or the Houston city line
- is a phone or fax line
- consists mostly of email addresses

The rule was applied to real and AI replies alike. It changed 59 of the 183
real replies and removed 280 lines. Mean real length fell from 65.0 to 58.9
words. No reply became empty. It changed 1 AI reply in total. Sentences that
mention Enron were kept, such as a note about "Enron's Domestic Affiliates".

**Results.** Cross-validated AUC, before and after removing the lines.

| Run | Length only | Full text | Text, same length |
|---|---|---|---|
| Llama, no instruction | 0.910 → 0.879 | 0.912 → 0.903 | 0.840 → 0.841 |
| Llama, instruction | 0.879 → 0.837 | 0.928 → 0.919 | 0.883 → 0.881 |
| DeepSeek, instruction | 0.603 → 0.571 | 0.981 → 0.978 | 0.978 → 0.977 |

![Length only, full text and text at the same length, for Llama and DeepSeek with the same prompt.](docs/figures/nvidia_length_vs_text_auc.png)

What the table shows:

- **Length no longer gives DeepSeek away.** After cleaning, the length-only
  AUC is 0.571 for DeepSeek, against 0.837 for Llama with the same prompt.
- **DeepSeek's words give it away almost perfectly.** The text AUC is 0.978,
  and 0.977 at the same length.
- **Removing the leftover lines changed almost nothing.** DeepSeek's text
  AUC moved from 0.981 to 0.978. The idea that these lines drove the result
  was wrong.
- **DeepSeek is easier to spot by its words than Llama.** At the same
  length, 0.977 against 0.881.

**Which words give DeepSeek away.** After cleaning, with real replies cut to
the same length:

| Word | DeepSeek replies containing it | Real replies containing it |
|---|---:|---:|
| "I'll" | 74% | 6% |
| "let" | 45% | 10% |
| "know" | 42% | 12% |
| "review" | 29% | 3% |
| "confirm" | 19% | 1% |
| "today" | 14% | 3% |
| "flag" | 11% | 0% |

These come from stock phrases such as "I'll review ...", "Let me know ..."
and "I'll flag ...". Llama writes "I'll" in 29% of its replies. The words
that now point to real replies are sign-off names (Kim, Sara, Perlingiere,
Shackleton) and words like "attached", "department" and "office". Real people
sign with their name and refer to attachments. DeepSeek rarely does.

**Limits.**

- The line-removal rule is a one-off script. The corpus cleaner does not use
  it yet.
- The text classifier splits its rounds by pair, not by thread, as in the
  original `fidelity.py`. The length classifier splits by thread.
- One number is unexplained. The full-text AUC for Llama without the
  instruction is 0.912 here, while `PROGRESS.md` reports 0.927. The
  length-only numbers match. The uncleaned real text also gives 0.912.

**What this means.** "Real and AI replies are separable, mostly by length"
was true for Llama. It is not true for DeepSeek. DeepSeek writes at a
realistic length and is still easy to spot, by its stock phrases. So the
stronger model does not make the replies harder to tell apart. It moves the
giveaway from length to word choice.

---

## 10. Next steps

1. **Hand-code a sample of DeepSeek replies.** Hand-coding means a person
   reads each reply and picks a label from the section 35 codebook. The
   mirroring measure was checked on Llama replies only, against Claude's
   first-pass codes. No person has coded any reply yet. Section 8 suggests
   that a high score means something different for DeepSeek. Codes from a
   person would settle this. The codebook needs one fix first: the first
   pass used a label, `wrong_register`, that the codebook does not define.
2. **A DeepSeek run without the act instruction.** This answers the question
   section 43 left open: does a larger model follow the instruction better?
   The instruction is always on in `prompt.py` now, so this first needs a
   small code option to switch it off. The run takes about 4 to 5 hours on
   the free tier.
3. **Save the length check in code.** Its numbers now come from a one-off
   script. It should be part of `mirroring.py`, so the numbers can be
   reproduced.
4. **Embedding map and review pack on DeepSeek.** These also only read the
   saved replies.
5. **Judge study (Q3) with a second model family.** For example, Nemotron as
   judge and DeepSeek as writer. This needs new model calls.
6. **Rename one summary key.** `mirroring.py` saves the comparison under
   `compared_with_previous_prompt`. For a comparison between models, that
   name is wrong.
7. **Back up `runs/_cache`.** It holds hours of NVIDIA replies and exists
   only on this laptop. It must not go into git, because the prompts contain
   Enron text.
8. **Commit the code.** Done on Sep 13, in three commits: the NVIDIA client
   and its tests (`57035ab`), the `compare_runs` fix (`ec0829e`), and the
   cost ledger with the DeepSeek mirroring manifest (`cac4688`).
9. **Move the line removal into the corpus cleaner.** Section 9 removed
   signature, address and header lines with a one-off script. If the cleaner
   in `thesis.data.rfc822` did this, every analysis would use the same clean
   text. Earlier results that use the real replies would then need a re-run.
