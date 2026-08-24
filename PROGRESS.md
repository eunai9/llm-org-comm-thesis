# Thesis Progress Log

A plain-language record of what has been built and found so far. Written for
you to read, not for a computer to run. Updated after each work session.

- **Code and data live in:** WSL (Linux environment on your Windows machine),
  pushed to GitHub at https://github.com/eunai9/llm-org-comm-thesis
- **Research plan (Exposé) lives in:** your OneDrive Thesis folder, unchanged

---

## Status at a glance

| Stage | Status |
|---|---|
| Project setup (coding environment, GitHub) | Done |
| Download and clean the Enron email dataset | Done |
| Figure out *who* sent each email (identity) | Done |
| Look up each person's job title / rank | Done |
| Reconstruct email conversation threads | Done |
| Measure "power" expressed in each email's writing style | Done (see note below) |
| Draw the samples the simulator and judge will use | Done |
| Plumbing for talking to the AI models (cost, caching) | Done |
| Build the AI agent simulator | Done, except the second AI provider |
| Half-price bulk submission (batching) | Done |
| Give each persona a "memory" of recent context | Done |
| Two working demos (terminal + browser), runnable by anyone | Done |
| Build the AI judge (scoring rubric and pipeline) | Done, on the free path |
| Statistics that compare real vs. AI-written emails | Done, on the free path |
| AI replies to a real email, compared to the real reply | Done, on the free path |
| Get API keys / decide on budget | **Decided: staying free — see note below** |

---

## A few terms, explained once

- **Corpus** — the whole email dataset (all 500k+ Enron emails).
- **Deduplicate / dedup** — remove repeated copies. The Enron export saved
  a copy of each email once per folder it appeared in (e.g. once in the
  sender's "Sent" folder, once in each recipient's "Inbox"), so the raw file
  count overstates how many *actual distinct emails* exist.
- **Unit test** — a small automated check that verifies one specific thing
  the code should do. Having many of these means mistakes get caught
  immediately instead of showing up later in your results.
- **Coverage** — what percentage of emails we can actually use, after
  applying a filter (e.g. "we know who sent it").
- **Power score** — a single number per email meant to capture how much
  linguistic and behavioral "authority" the writer is projecting (giving
  directions vs. hedging, being replied to quickly vs. slowly, and so on).
- **Construct validity** — whether a measurement actually measures the thing
  it claims to. For the power score, the check is: do people in more senior
  roles actually score higher? If not, the measurement isn't capturing what
  it was designed to capture, whatever else it might be picking up.

---

## What's been done

### 1. Project setup (Aug 7–8)

Set up a proper software project so the work is organized, testable, and
backed up:

- Created a Python coding environment in WSL (a Linux system that runs
  alongside Windows — better suited for this kind of data work).
- Created a private-turned-public GitHub repository
  (`llm-org-comm-thesis`) as an off-machine backup and version history —
  every change is recorded and recoverable.
- Set up automated code-quality checks (formatting, type-checking, and the
  unit tests mentioned above) that run before anything is considered "done."

### 2. Downloading the Enron email dataset (Aug 8)

- Downloaded the **official** Enron email archive (published by Carnegie
  Mellon University in 2015) rather than a third-party copy — this is the
  version other researchers cite, so your data source is easy to justify.
- Verified the download wasn't corrupted (checked its digital fingerprint
  against the one CMU published).
- **Result:** 150 people's mailboxes, 517,401 individual email files, 2.6 GB.

### 3. Cleaning and parsing the emails (Aug 8)

Wrote code that reads each raw email file and extracts the useful parts:
who sent it, who received it, when, the subject, and the actual written
text (with quoted reply chains and email signatures stripped out, since
those aren't the sender's own words).

**Two real bugs were found and fixed while testing this against the actual
data** (not just made-up test examples):

1. **Duplicate removal wasn't working.** The system Enron used to export
   these emails gave every *copy* of a file its own unique ID — so the
   "official" ID couldn't be used to tell that two files were the same
   email. This meant duplicates were slipping through uncaught. Fixed by
   comparing the actual email content instead of relying on the ID.
2. **The program would have crashed partway through** on emails containing
   certain non-English characters (accented names, etc.) in the sender/
   recipient fields. Fixed and added a permanent check so it can't happen
   again.

**Result after cleaning:**

| | |
|---|---:|
| Raw files | 517,401 |
| **Actual unique emails** | **254,359** |
| (i.e., the raw count was ~2x too high) | |
| Emails that turned out to be empty forwards (no real content) | 16,686 |

⚠️ **Important for your writing:** always cite **254,359** as the number of
emails in this study, not 517,401 — that raw number double-counts.

### 4. Figuring out who sent each email (Aug 9)

An email address alone doesn't tell you *who* a person is or what job they
had — and the same person often used more than one address. So the next
step was matching addresses to actual people, using each of the 150
mailbox owners' own "Sent" folder as the anchor (i.e., "whichever address
this person sends *from* is their address").

Found and fixed one more bug here: one real employee's email address
contained an apostrophe (`paul.y'barbo@enron.com`), which broke a database
query that wasn't expecting punctuation like that. Fixed, and it won't
recur.

**Result — this is the key number for your Q1 (hierarchy) analysis:**

- 191 person-addresses identified across 146 of the 150 mailboxes
- Of the emails that are usable for analysis (right length, right date
  range, internal senders), **44.8% come from a person we can identify.**
  This was expected to land somewhere between 25–45%, so landing at 44.8%
  means the hierarchy analysis (Q1) is on solid footing.

---

### 5. Grouping emails into conversations (Aug 9)

To study how people talk to each other, we need to know which emails are
replies to which — i.e. group them into conversations.

Normally email carries hidden technical labels that make this easy. **This
dataset has none of them** (the Enron export stripped them out — I checked all
254,359 emails and not one has them). So instead, emails are grouped when all
three of these are true:

1. they have the same subject line (ignoring "Re:" and "Fw:"),
2. at least two of the same people are involved, and
3. they happened within 30 days of each other.

The rules are deliberately strict. Wrongly gluing two unrelated conversations
together is much worse for this project than splitting one conversation in
two, because these conversations get fed to the AI as examples — and a
nonsense example produces a nonsense result.

**One thing worth knowing:** a large share of what first looked like
"conversations" turned out to be automated mail — daily newsletters and system
alerts that always share a subject line. The biggest was 251 emails, all from
a single sender, over four months ("Williams Energy News Live"). These aren't
conversations at all, so they're now automatically detected and marked
separately.

**Results:**

| | |
|---|---:|
| Real conversations found | 18,467 |
| Of those, with 3+ emails (best for AI examples) | 8,959 |
| Automated newsletters/alerts, correctly excluded | 14,532 |
| Emails that were one-offs, not part of any conversation | 157,993 |

8,959 usable conversations is far more than the ~200 this project needs, so
there's plenty to choose from.

⚠️ **Still to do here:** because this method is based on matching rather than
exact labels, it can make mistakes. I've generated a sample of 50
conversations for someone to read through and check by hand
(`data/interim/threads_review_sample.txt`). That accuracy figure should go in
your thesis as a stated limitation.

---

### 6. Job titles and seniority ranking (Aug 9)

This finishes the piece that was blocked: knowing not just *who* sent an
email, but what their **job title and seniority level** was, so hierarchy can
actually be compared.

**Both of the two sources we agreed on turned out to be dead links** — I
checked thoroughly (see the sourcing conversation for details) and neither
was actually retrievable anymore. Rather than give up, I searched further and
found a **live, working alternative**: a 156-person employee list
(name, department, job title, and a Junior/Senior label) published by a
statistics professor who has done formal research on this exact dataset, with
a legitimate academic citation. Full details of what was tried and why it was
swapped are recorded in `data/external/SOURCES.md`.

**How the matching works:** Enron's own email system named each person's
mailbox folder in a very consistent way — surname, plus first-initial (e.g.
Sally Beck → `beck-s`). So instead of fuzzy-guessing who's who from names in
email headers (error-prone), the code rebuilds that exact folder-naming
pattern from the employee list's names and matches it directly against the
real mailbox folder names. This is precise rather than approximate.

**I also built a job-title → seniority-rank table by hand** — 36 distinct
titles found in the employee list ("VP Trading", "Mgr Trading", "Director",
etc.), each assigned a rank from 1 (entry-level) to 6 (President/CEO), done
**before** looking at any result, so the ranking couldn't be unconsciously
shaped to produce a nicer-looking answer. This table is committed to the
project (`data/external/title_to_rank.csv`) and can go straight into a
thesis appendix.

**Results:**

| | |
|---|---:|
| Employees in the source list | 156 |
| Successfully matched to a mailbox | 129 (82.7%) |
| **Eligible emails with a known sender title/rank** | **47,567 / 117,794 (40.4%)** |

The 27 unmatched people break down cleanly:
- A handful are genuinely two different people who share a surname and first
  initial (e.g. two people named "Dean, C..."), correctly left unresolved
  rather than guessed at
- A few appear in the employee list but never sent an email that survived
  into the usable dataset
- A very small number are one-off naming quirks (e.g. someone who went by
  their middle name, or a two-word surname the automatic pattern didn't
  expect) — noted rather than hand-patched, since chasing 3–4 individual
  people with special-case code isn't worth the added complexity

**A nice sanity check:** the by-hand rank table was cross-checked against the
employee list's own Junior/Senior label after the fact. They agree
**completely** — every rank category (Manager, Director, VP, etc.) lines up
with exactly one of Junior or Senior, with zero contradictions. That's good
independent evidence the ranking is sound.

⚠️ **40.4% is now the real, final number for how much of the dataset can be
used in the hierarchy analysis (Q1).** This is slightly lower than the
44.8% mentioned before, because now the bar is "we know their exact job
title," not just "we know who they are." Both are healthy numbers for this
kind of study.

---

### 7. The power score (Aug 10)

This is the measurement Q1 (the hierarchy question) depends on most: a
per-email score meant to capture how much authority a person's writing
projects, combining two kinds of signal that were built and frozen
*before* either was ever run against real results:

- **Writing-style signal** — how often someone gives direct instructions vs.
  hedges ("maybe", "perhaps"), defers ("if it's not too much trouble"), or
  makes personal commitments ("I'll take care of it"). Measured with a
  natural-language-processing tool (spaCy) run over every one of the
  237,627 usable emails.
- **Behavioral signal** — how central someone is in the email network, how
  often they start conversations vs. get the last word, and whether people
  reply to them faster than they reply to others.

**This step ran into serious engineering trouble before it worked.** A
single email in the dataset — not a real message, more like a 1.7-million
character block of pasted text — was large enough to overwhelm the
text-processing tool's memory needs and crashed the environment outright,
twice, before the cause was pinned down. The fix excludes the 46 emails
in the whole dataset (0.02%) that are implausibly long to be real
correspondence, which then let the real run complete cleanly in about 77
minutes. A second, unrelated crash turned out to be a Windows configuration
issue (too much memory reserved for the Linux environment, leaving Windows
itself unable to breathe) and was fixed separately. Both fixes are committed
so this won't recur.

**The result — reported honestly, exactly as planned in advance:**

| Seniority rank | Mean power score | Emails |
|---|---:|---:|
| 1 — Junior employee | **+0.088** | 27,467 |
| 2 — Manager | −0.044 | 10,906 |
| 3 — Director | −0.025 | 17,729 |
| 4 — Vice President | +0.003 | 18,092 |
| 5 — Managing Director | −0.010 | 6,703 |
| 6 — President / CEO | +0.016 | 3,796 |

**The score does not track seniority.** If it worked as hoped, the numbers
would climb steadily from rank 1 to rank 6. Instead junior employees score
highest, executives are barely above zero, and the statistical correlation
between rank and score is essentially flat (slightly negative, technically:
Spearman's ρ = −0.065 — for reference, 0 means no relationship at all).

This was flagged as a real possibility from the start, specifically so it
wouldn't be tempting to quietly adjust the formula until it "worked." The
formula stays exactly as originally frozen. **This is now a real, reportable
finding for the thesis** — either the writing-style/network theory of
"power" needs revisiting for this dataset, or (more likely, worth checking
next) the two pieces that make up this score need to be looked at
separately rather than only as a combined number, and compared against the
upcoming AI-judged labels as a second opinion.

---

### 8. Sampling: drawing the three sets everything downstream uses (Aug 14)

Before you looked at options for revising the power score, you asked what
the next step would be if you simply left it as-is for now. This section is
that next step, and it does not depend on however the power-score question
eventually gets resolved.

Everything from here on — the AI simulator, the AI judge, the labelling —
needs specific, fixed sets of real emails to work with, drawn **once**, with
one random seed, so every later result traces back to the same starting
point. Three sets, drawn from the 47,567 emails that pass every filter
(real correspondence, reasonable length, sender's job title known, inside
the study window):

| Sample | What it's for | Drawn |
|---|---|---:|
| S_label | Emails an AI will label (purpose, tone, etc.) to check the power score and train further labelling | 3,000 / 3,000 |
| S_shots | Real email threads the simulator will be prompted with, to write realistic replies | 200 / 200 |
| S_real_eval | The *actual* real replies inside those same 200 threads, so a real reply and an AI-written reply can be judged side-by-side, answering the exact same message | 302 / 400 |

**S_real_eval came up short of the target (302, not 400) — reported
honestly rather than padded.** There simply aren't 400 real replies sitting
inside emails that also pass every other filter; some of the 200 threads
only had one or two eligible replies. This doesn't threaten anything
downstream — it just means slightly fewer real-vs-AI paired comparisons
later.

**A performance problem, caught and fixed before it could recur:** the
first real run of this step took **47 minutes**, which was surprising for
what should be quick. The cause was a common but easy-to-miss inefficiency
— checking each of the ~18,000 real conversation threads one at a time in a
slow way, instead of all at once. Rewritten to check them all in a single
step, the exact same result now takes **under one second**. Worth
mentioning only because it's the kind of thing that would have made
re-running this step later (e.g. after any small config change) a
half-hour tax for no reason.

---

### 9. Plumbing for talking to the AI models (Aug 15)

Phase 3 (the AI simulator) starts here. Before any AI can be asked to write
an email, there has to be a reliable, cheap, and *reproducible* way to talk
to it. That layer is now built and tested.

Three pieces, each solving a specific problem:

**Cost control.** Every call is priced before it's made, checked against a
budget ceiling, and recorded afterwards in a plain spreadsheet file you can
open and read. If a planned run would cost more than the configured limit,
it refuses to start rather than discovering the overspend afterwards. Prices
are deliberately set to the *standard* rates rather than the discounted
promotional ones that expire at the end of August — a budget warning that
fires slightly early is useful; one that fires too late isn't.

**A permanent archive of every AI response.** This is the most important
piece, and it's about your thesis rather than about speed. **AI responses
aren't repeatable** — asking the same question twice can give different
answers, and on the model we're using there's no "randomness setting" to
lock down. So exact regeneration is genuinely impossible, and claiming
otherwise in the write-up would be false. What *is* possible is saving every
response permanently, so that:

- Re-running the analysis in January costs **nothing** and gives identical
  results.
- The whole thing can run with no internet at all — which is also how a
  reviewer could check your work without needing an API key of their own.
- The archive can be published with the thesis, making "we archived all N
  responses" something a reader can verify rather than just believe.

The archive is filed by a fingerprint of the **exact text** sent to the AI.
That detail matters: if it were filed by "which template was used" instead,
then editing a prompt's wording without renaming it would silently serve back
answers generated by the *old* wording — producing an analysis of prompts
that were never actually sent. Filing by exact text makes that mistake
impossible.

**Guardrails against three costly mistakes.** The model we're using rejects
the "creativity dial" (temperature) outright — the exposé had planned to use
it, so this is now a stated limitation rather than a surprise 
mid-experiment. It also has a minimum prompt length below which the cost-saving
cache silently does nothing, and that minimum *differs between models*. And
its "thinking" time counts against the same budget as its visible answer, so
a budget sized for the answer alone gets the answer cut off. All three are
now recorded in code and covered by tests, so they fail loudly at build time
instead of quietly during a paid run.

**One thing verified rather than assumed:** the archive's fingerprints are
byte-for-byte identical across separate program runs. Python deliberately
randomizes some internal hashing between runs, so this was worth actually
testing rather than trusting — if it weren't stable, the January re-run would
miss every saved response and silently re-bill the entire project.

---

### 10. The simulator itself (Aug 15–17)

This is the machine that writes the simulated emails. Five parts:

**The personas.** Ten role archetypes — five seniority levels across two
departments (Trading and Legal). They are **never named real Enron people**,
for two reasons that both matter. The obvious one is ethics: these are real
people's real messages. The less obvious one is that today's AI models have
memorised big parts of this famous dataset, so an AI asked to write "as Jeff
Skilling" might simply be *recalling* him rather than *simulating* a role —
which would quietly invalidate the whole fidelity question (Q2). Instead each
persona is a statistical summary of how people at that level actually wrote.

⚠️ **A problem found and fixed along the way:** summarising a group doesn't
automatically anonymise it. Four of the ten role groups turned out to contain
only one or two real people — so the "archetype" would effectively have *been*
that individual, reintroducing exactly the problem the approach was meant to
avoid. The code now requires at least three people behind every persona, and
falls back to a broader average when a group is too small. Which four personas
this affected is recorded and reported rather than hidden.

⚠️ **A limitation worth stating in the thesis:** at the most senior level, both
departments fell back to the same broader average, so those two personas differ
only by their *label*. A "department effect" measured at that level would be
measuring nothing real. The code detects and flags this automatically.

**The scenarios.** 144 synthetic workplace situations (six task types ×
three directions × two stakes levels × four tones — see entries 17 and 18
below for the tone dimension and the sixth task type). They contain no Enron
content at all — they're written from scratch. Real emails enter the study
separately, as the `S_shots` stimuli.

**Memory.** Each persona carries a small "memory" of recent working life,
following the standard published method for AI agents. It's retrieved once per
persona and reused across scenarios — both cheaper and a cleaner comparison,
since two scenarios that differ only in stakes can't accidentally differ in
what the persona happened to remember.

**The prompt.** Carefully ordered so the expensive, unchanging part can be
cached and reused. Getting this order wrong wouldn't produce an error — it
would just quietly multiply the bill.

**The runner.** Walks the whole experiment, checks the archive before paying
for anything, records where every result came from, and refuses to start a real
run from uncommitted code (so every future figure traces back to exact code).

**Three bugs caught by tests before any money was involved**, all silent ones:

1. **Repeats collapsed into one.** The design asks the AI the same question
   five times to measure how much its answers *vary*. Because the question is
   identical each time, the archive was handing back the first answer for all
   five — so measured variation would have been exactly zero, and the
   "diversity" finding would have been describing the filing system rather than
   the AI.
2. **The bulk-submission ID limit.** Our descriptive labels run to 70
   characters; the bulk API accepts shorter ones. This would have failed only
   at the moment a large paid submission was accepted.
3. **Stub test runs were being billed** at real prices in the cost log — the
   file the thesis's total-cost figure is summed from.

---

### 11. Running it for free, before spending anything (Aug 15–17)

You asked whether the prototype could be tested without paying. It can, and it
now runs three ways:

| Mode | Cost | The text is… |
|---|---|---|
| `--offline` | free | fake (fill-in-the-blank templates) |
| `--local llama3.2:3b` | free | **really generated**, by a small AI on your own PC |
| *(default)* | paid | generated by the top-tier models the thesis names |

The middle option is the useful one. A small open AI model now runs locally on
your machine (installed without needing administrator rights), so prompts can
be tested on real generated text before any money is spent.

**Safeguards, so free output can never be mistaken for a result:** locally
generated emails are stamped `local/…` in the results file, flagged in the run
record, and refused by the cost calculation. This matters beyond quality — the
thesis names *specific* AI models, so a small model on a laptop can't stand in
for one in a results table however good it looks.

**It immediately earned its keep.** The first real local run showed the AI was
opening emails with the literal word *"decline."* — nobody writes an email that
starts with a category label. The instruction wording was at fault, not the
model. After the fix, measured on the same twelve emails: emails starting with
a decision word went from **4 out of 12 to 0 out of 12**.

**Deliberately not fixed yet:** the generated emails are much shorter than the
persona's target, and the stated decision sometimes doesn't match the email's
content. These are most likely limits of the small local model rather than
faults in the prompt — and over-tuning the prompt against a weak model risks
making it *worse* for the real one. Worth rechecking once a real key exists.

---

### 12. Two ways to actually see it working (Aug 19)

Up to this point, the only way to look at a generated reply was to read a
data file with code. That's fine for analysis, wrong for a demo. Two proper
demos now exist, both free, both showing the exact same underlying behavior:

- **A terminal version** (`python -m thesis.sim.demo`) — pick a role and a
  scenario from a menu, see the reply printed nicely.
- **A browser version** (`python -m thesis.sim.webdemo`) — the same thing as
  an actual web page with dropdowns, easier to read and easier to show
  someone else on the same screen. This is what you asked for as a "web
  demo" — confirmed as a **local-only** page (nothing reachable outside your
  own machine), since a page reachable from the internet would need real
  hosting and would cost money, which isn't what you wanted.

**Two real problems, caught by actually testing rather than assuming it
worked:**
- The web page's server appeared unreachable on the first attempt — turned
  out to be an unrelated technical quirk in how the background process was
  started, not a real networking problem. Confirmed by restarting it
  correctly.
- A generation request then timed out with no trace of it on the server —
  meaning it never actually arrived. Traced to the *test script itself*
  (a Windows PowerShell quirk), not the actual page — your real browser
  doesn't have this problem, confirmed by testing the identical request
  successfully through a different path.

**A third real gap, found by trying to hand this to someone else:** neither
demo could actually run on a different computer, because this project
deliberately never uploads the processed email dataset (~270MB) to GitHub.
Fixed by freezing the real, computed numbers into a small file that *is*
uploaded (`data/external/personas_snapshot.json`), so anyone with the code —
your supervisor included — gets the real numbers without needing the dataset
itself. A written guide, **`RUNNING_THE_DEMO.md`**, walks through the whole
setup from scratch. It was tested twice on a completely clean checkout before
being trusted — the first attempt caught a real missing step in my own
instructions, which was fixed and re-verified before being finalized.

---

### 13. Giving each persona a memory (Aug 21)

One piece of the simulator had existed in code but never actually worked:
each role was supposed to carry a small memory of "recent context" that
shapes how it writes, but nothing had ever generated that content. Every
reply you've seen in either demo, until now, was written with an empty
memory.

That's now fixed. For each of the 10 roles, the AI was asked to invent
15 small, plausible things that might have happened recently in that kind of
job (never naming real people or real events), and then to summarize a few
general "tendencies" from those — the same two-layer memory design used in
well-known published AI-agent research. Retrieval (deciding which few
memories are most relevant to a given reply) already existed and had already
been tested; what was missing was only the step that invents the memories in
the first place.

**Measured, not assumed working:** pulled an actual generated reply and
confirmed the memory content genuinely appears in what gets sent to the AI,
then ran real replies end-to-end to confirm nothing broke. The AI didn't
always produce a full set of summarized tendencies — it fell short on some
roles — and that shortfall is reported as-is rather than padded, the same
approach taken with every other honest gap in this project so far.

⚠️ **One caveat to remember later:** this memory content was invented by the
small, free local AI model — reasonable for testing and demos, but the
actual thesis results should have memory generated by the *same*, real AI
model used for the main experiment. Mixing a weak model's invented memories
into a strong model's writing would be a real, confusing side-effect, not
just a style choice — worth revisiting once a real key exists.

---

### 14. Starting the AI judge — the part that scores the emails (Aug 21)

The judge (the part that scores how good each email is) is now under
construction, using the exact same approach that worked for the simulator —
build it fully, test it for free, run it for real later, if that ever
happens.

**One real limitation, stated plainly rather than glossed over.** The
research design calls for the judge to always be a *different* AI model
family than the one that wrote the email — this is what lets the study
detect whether a model is quietly biased toward favoring its own writing.
That specific comparison genuinely cannot be done on the free path. Everything
else about the judge — the actual scoring questions, how a question is
asked, how the scoring pipeline runs — does not need that comparison to be
built and tested.

**What's built:** six scoring questions, in two groups (does this sound like
the right person for the role, and separately, is it well-written), each
scored 1–5 with a required short quote justifying the score. Critically, the
judge is never told whether it's looking at a real email or an AI-written
one — that blindness is what makes the eventual real-vs-AI comparison
trustworthy rather than something the judge could shortcut by just
recognizing which is which. Three different phrasings of the same six
questions were also built, because how *consistent* the judge is when the
same question is worded differently is itself something worth measuring,
not something to paper over.

**Verified live, for free:** ran it against two made-up example emails — one
professional, one deliberately unprofessional — using the free local model.
It correctly scored the unprofessional one low across every dimension and
the professional one reasonably, confirming the mechanism works before any
money is involved.

---

### 15. Can the AI tell a real email from an AI-written one? (Aug 21–22)

Two more pieces, finishing the judge and starting to make sense of its
output.

**A second, separate way of asking the judge to look at an email:** instead
of scoring it on the six questions, just ask "does this look real, or
written for a study?" on a 1–5 scale. This is deliberately a *different*
request from the six-question scoring, not an extra seventh question bolted
on — asking the judge to hunt for signs of fakery in the same breath as
asking it to judge quality risked contaminating the quality scores
themselves.

**The statistics that actually compare real vs. AI-written emails.** Four of
them, matching the research plan exactly:
- Is there a real difference in how the two are scored? (a standard
  significance test)
- Can we *positively* say they're similar, rather than just failing to find
  a difference? (a stricter, more honest framing your own plan specifically
  called for)
- Can the judge itself tell them apart, directly?
- Can a much simpler, non-AI method (just counting which words appear) tell
  them apart?

⚠️ **Two of the four need a kind of data this project can't produce yet** —
a real email and an AI-written reply to the *exact same* incoming message,
so the comparison isn't muddied by comparing different situations. That
pairing needs a simulator capability (answering a real email directly,
rather than a made-up scenario) that hasn't been built. Those two statistics
are built and thoroughly tested on made-up example numbers, ready for when
that data exists.

**The other two ran on real data already collected:** the judge's own
guess at telling real from AI-written scored barely better than a coin
flip (matches what you'd hope to see — it means the AI-written emails are
hard to spot). The simple word-counting method, by contrast, told them apart
*perfectly* — but with only 5 real Enron emails and 5 emails about made-up
scenarios, that's almost certainly because the topics discussed are just
obviously different, not a real finding about writing quality. Worth
knowing, not worth reading much into yet.

---

### 16. Comparing an AI reply to the actual real reply, on the same email (Aug 22)

Until now, the AI simulator only ever answered made-up practice situations.
This teaches it to answer a **real** email — the exact same one a real
Enron employee actually replied to — so the AI's reply and the real
person's reply can be compared head to head, on identical footing. This is
what the "statistics that compare real vs. AI-written emails" (section 15)
were actually waiting for: without this, they had nothing genuinely
matched to compare, only different topics being compared to each other.

Of the 200 real conversations sampled earlier for this purpose, 190 replies
turned out usable (133 distinct conversations; some conversations had more
than one real person reply, and each of those is now its own comparison).
The rest are skipped for the same reason a handful of roles were already
left out of the simulator entirely — either the real replier's seniority
was the single most senior tier (deliberately excluded from the start, to
avoid the AI "recognizing" a real famous person) or their department wasn't
one of the two modeled (Trading or Legal).

**Two real bugs, caught before they could quietly produce wrong numbers:**
one conversation with two different real repliers was initially being
tracked as if it were only one comparison, silently losing the second;
and an identifier was being shortened in a way that — very rarely, but
possibly — could make two different comparisons look like the same one.
Both fixed and checked.

**Ran the whole thing for real, for the first time:** generated 6 AI
replies to real emails, compared each against the real person's actual
reply, and ran the statistics. The result is a good, concrete illustration
of exactly why two different statistical tests are used side by side: the
simpler test found "no clear difference" on every dimension — which sounds
like good news — but the stricter test (the one that requires *positive*
evidence of similarity, not just an absence of detected difference)
correctly refused to call them equivalent on any dimension. On two of the
six ("does the reply actually engage with this specific situation" and
"is disagreement handled well"), the real gap was more than double what
the stricter test would accept. With only 6 examples and the smallest free
AI model, this isn't a real finding yet — but it's the pipeline working
correctly end to end for the first time, and it's already a preview of the
kind of honest, non-oversold result this project is built to produce.

---

### 17. Adding tone as a fourth dimension of the scenario grid (Aug 23, corrected Aug 24)

The scenario grid gained a fourth factor: **tone** of the message a persona
receives. Four levels: Deferential, Warm, Neutral, Assertive. Each task type
now has four hand-written versions of the same underlying request, one per
tone — same ask, different register. "Neutral" is each task type's plainest
phrasing; the other three rephrase it in a different voice. The persona is
never told how to respond — it only ever sees a stimulus already written in
one of these four registers and replies however it naturally would. This
tests whether the tone of an incoming message shapes the reply, on top of
who it's from.

**This was wrong the first time.** The version built and reported on Aug 23
instead gave the *persona* an explicit instruction ("Write in a deferential
tone: downplay your own authority...") layered on top of a fixed, tone-less
incoming message — testing instruction-following, not whether an incoming
message's tone shapes a reply. A pilot run under that version even reported
a real "tone dominates hierarchy" finding, which does not carry over: it was
answering a different, less interesting question. Caught when re-explaining
the design and corrected the same day — the fix rewrote every incoming
message into four tone variants and removed the instruction sentence
entirely. No pilot has been rerun against the corrected version yet.

**Naming note (unaffected by the correction):** the fourth level was
originally going to be "Aggressive." Renamed to "Assertive" — real business
email in this corpus essentially never reaches outright hostility, so an
"aggressive" condition would have tested how far the AI can be steered into
an unrealistic register rather than a genuine organizational behavior.

**What this cost:** the scenario grid was 30 situations (5 task types × 3
directions × 2 stakes); crossing in 4 tones makes it 120. To keep the total
number of AI replies generated from growing 4x along with it, the number of
repeated draws per situation (needed to tell a real effect apart from the
AI's own randomness — see the terms section at the top) was cut from 5 to 2,
the lowest number that still lets that distinction be made at all. Net
effect: total generations go from 3,000 to 4,800 — a real increase, but far
short of the 12,000 a straight 4x would have been.

---

### 18. A sixth task type: confirming details (Aug 23)

Added one more situation to the scenario grid: **confirm_details** — someone
checking that a plan already in motion is still on track ("just confirming
we're still moving forward with the numbers from last week's call, right?"),
which only needs a quick yes/no rather than a real decision. This fills a
gap the other five didn't cover: `approve_or_decline` already carries risk
and a real choice, `request_information` asks for new data, but nothing
tested the lightweight, low-effort acknowledgment end of workplace email.

Task types are now 6 instead of 5, so the scenario grid grows from 120 to
144 situations (6 × 3 directions × 2 stakes × 4 tones), and total AI-reply
generations from 4,800 to 5,760.

---

### 19. A mixed-model crash that only shows up when a persona effect is genuinely zero (Aug 24)

While generating the (since-superseded) 240-reply direction × tone pilot,
the Q1 mixed model crashed partway through with a numerical error
(`LinAlgError: Singular matrix`), not a Python bug. The cause: for
`hedge_rate`, persona genuinely explains close to none of the variance —
the same thing the very first Q1 pilot already found, where its estimated
persona effect was already ~0. Statistics software fits that "zero"
by searching for the best value of a number that can't go below zero, and
right at that lower boundary the default search method's internal math can
divide by (numerically) zero. Two other search methods that don't rely on
that math (Powell, Nelder-Mead) handle it fine, so the model now tries the
default first and automatically falls back to those if it fails — a
one-line change in practice, verified with a test that reproduces the exact
zero-effect situation and confirms it no longer crashes.

---

## What's next

**Everything currently useful to build for free is built, and now connected
end to end** — sampling real conversations, generating AI replies (to made-up
situations or to real emails), scoring both with the judge, and comparing them
statistically, all for free, all tested.

You and your supervisor have decided not to invest money in this project.
That's a settled decision, not something pending. Given that, here's the
honest, current state:

- ✅ **Fully possible for free, and worth continuing:** running this pipeline
  on more than 6 examples to get a real (if still not thesis-grade) read on
  the pattern, more testing, more demos, refining prompts.
- ⚠️ **Genuinely out of reach on the free path, by the research design
  itself:** the actual final thesis results, and specifically the
  comparison of whether the judge favors its own kind of AI over another.
  You mentioned exploring a different way to access AI models for that
  piece later — happy to help once that direction is clearer.

Still open, but not blocking anything:

- The power-score method question — you're reading through the options for a
  more current, LLM-based approach. It only needs deciding before the labelling
  step (`S_label`) actually runs.

---

## If a paid option ever becomes relevant again

Keeping this for reference only — not the current plan. **The short
version:** two accounts, both prepaid, roughly **$150 total** for
the whole thesis. Worth asking your supervisor about a research budget first —
this is a normal thing for a department to cover.

**Anthropic** — <https://console.anthropic.com>
1. Sign up, then add credits under *Billing*.
2. Create a key under *API keys*. It is shown **once** — copy it immediately.

**OpenAI** — <https://platform.openai.com/signup>
1. Sign up. ⚠️ This is *not* the same as a ChatGPT subscription — a ChatGPT
   Plus plan gives no API access, so don't buy one expecting it to count.
2. Add credits under *Billing*.
3. ⚠️ Check *Organization → General* for identity verification. Some models are
   gated behind it and it **can take days**, so start this early rather than
   discovering it the week something is due.
4. Create a key under *API keys*, and copy it immediately.

**Then, in the project folder**, make a file called `.env` containing:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

That file is already excluded from GitHub, so the keys cannot be committed by
accident. **Never paste a key into a chat or a document.**

**Then check it worked:**

```
python -m thesis.llm.verify_models --list
```

That prints every model your account can actually use — which is what gets
recorded in the config, rather than trusting a website that may be out of date.

---

## Where to look

- **All code and this file:** https://github.com/eunai9/llm-org-comm-thesis
- **Auto-generated data summary:** `outputs/manifests/corpus_report.md`
- **Conversations to hand-check:** `data/interim/threads_review_sample.txt`
- **Where the employee/title data came from, and what didn't work:**
  `data/external/SOURCES.md`
