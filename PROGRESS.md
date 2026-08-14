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
| Build the AI agent simulator | Not started |
| Build the AI judge | Not started |

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

## What's next

Data processing (Phase 2) is now fully complete — every piece the plan
called for in August is built, run on the real corpus, and committed.

- Start building the AI agent simulator (Phase 3) — the personas, memory,
  and prompt design that generate simulated email responses, using
  S_shots as the real stimuli.
- The power-score method question is still open — you're reading through
  options for a more current, LLM-based approach before deciding. Nothing
  else is blocked on that decision; it can be revisited any time before the
  labelling step (S_label) is actually run.

---

## Where to look

- **All code and this file:** https://github.com/eunai9/llm-org-comm-thesis
- **Auto-generated data summary:** `outputs/manifests/corpus_report.md`
- **Conversations to hand-check:** `data/interim/threads_review_sample.txt`
- **Where the employee/title data came from, and what didn't work:**
  `data/external/SOURCES.md`
