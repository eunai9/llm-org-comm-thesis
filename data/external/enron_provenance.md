# Enron corpus provenance

Recorded automatically by `scripts/fetch_enron.sh`. Cite the CMU release,
not a Kaggle mirror.

| Field | Value |
|---|---|
| Source URL | <https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz> |
| Upstream release date | 2015-05-07 |
| Retrieved | 2026-08-08 20:51:30Z |
| Archive sha256 | `b3da1b3fe0369ec3140bb4fbce94702c33b7da810ec15d718b3fadf5cd748ca7` |
| Archive bytes | 443254787 |
| Mailbox directories | 150 |
| Message files on disk | 517401 |

Note: the on-disk file count is **not** the number of unique messages. Each
message is stored once per mailbox folder it appears in, so senders' Sent
items and recipients' Inbox copies duplicate one another. Deduplication
happens in `thesis.data.ingest`; the unique count is reported there and
belongs in the thesis rather than this raw figure.
