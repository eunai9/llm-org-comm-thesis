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
| Look up each person's job title / rank | Not started — waiting on a source document (see "What's next") |
| Reconstruct email conversation threads | Not started |
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

## What's next

Two things are ready to start:

1. **Look up each identified person's job title** (Employee, Manager,
   Director, VP, etc.), so we can rank them by seniority. This needs a
   published Enron employee-title list to join against — **I need you to
   point me to the specific source you intend to cite** (there are a couple
   of versions researchers use), so I use the one that matches how you want
   to reference it in your methods section.
2. **Reconstruct email conversation threads** (which emails are replies to
   which). One caveat already discovered: none of the emails in this corpus
   carry the technical headers that would normally make this easy, so
   thread reconstruction will rely on matching subject lines and
   participants instead — a slightly less precise method, which will be
   clearly documented as a limitation.

---

## Where to look

- **All code and this file:** https://github.com/eunai9/llm-org-comm-thesis
- **Auto-generated data summary:** `outputs/manifests/corpus_report.md` in
  the repo (the source of the numbers above — regenerated by running one
  command, so the numbers can never go stale or be mistyped)
