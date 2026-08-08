# Corpus report

Generated 2026-08-08T21:27:41+00:00 by `python -m thesis.data.corpus_report`.

## Size, after deduplication

| Quantity | Value |
|---|---:|
| Files on disk | 517,401 |
| Distinct Message-IDs | 517,401 |
| **Unique messages** | **254,359** |
| Duplication factor | 2.0341x |
| Recipient rows | 1,611,163 |

`distinct_message_ids` equals the file count exactly. The JavaMail export
minted one Message-ID per file, so deduplication keys on a content
fingerprint instead. **Report 254,359 as N, not
517,401.**

## Corpus properties

- Messages with `In-Reply-To`/`References`: **0**.
  The export stripped them, so header-based thread reconstruction is
  impossible and a subject-plus-participants fallback is the only option.
- Empty after quote and signature stripping: 16,686
  (6.6% of unique).
  These are forwards carrying no newly authored text; they are excluded from
  the eligible pool.

## Identity resolution

| Quantity | Value |
|---|---:|
| Mailboxes with outgoing mail | 146 of 150 |
| Distinct owner addresses | 191 |
| Owners with more than one address | 40 |

## Sender coverage — the Q1 gate

| Quantity | Value |
|---|---:|
| Unique messages from a known owner | 102,972 (40.5%) |
| Eligible messages | 117,794 |
| **Eligible from a known owner** | **52,712 (44.8%)** |
| Distinct senders in that pool | 191 |
| Median messages per sender | 108 |
| Senders with at least 100 messages | 100 |

**Caveat.** This is coverage by *identifiable person*, not by *title*. Joining
the published employee-title list will reduce it further, since not every
mailbox owner appears there. Treat 44.8% as the
ceiling on empirical Q1 coverage.
