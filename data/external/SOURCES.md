# External data sources

Every file in this directory is vendored from a published source. Record the
URL, the retrieval date, and the licence for each one. These files are
committed to git because they are small, hand-curated, and load-bearing for
reproducibility.

| File | Source | Retrieved | Notes |
|---|---|---|---|
| `enron_employees.tsv` | Perry & Wolfe (2011); preprocessing by Zhou, Goldberg, Magdon-Ismail & Wallace (2007). Redistributed at github.com/patperry/interaction-proc/tree/master/data/enron | 2026-08-09 | 156 employees: name, department, title, gender, seniority (Junior/Senior). Two originally-planned sources turned out to be dead links after exhaustive checking (see note below) — this is a live, working substitute found during that search, from a co-author (Patrick Perry, NYU Stern) with a real citation chain. |
| `title_to_rank.csv` | Authored for this thesis, hand-enumerated from all 34 distinct titles in `enron_employees.tsv` | 2026-08-09 | **Frozen before measuring any coverage or hierarchy outcome** — every title mapped once, by hand, before results were looked at. 6-level ordinal ladder (Employee/Manager/Director/VP/Managing Director/President-CEO). The `given_seniority` column cross-checks against Perry & Wolfe's own Junior/Senior label; the two never disagree except at the Manager boundary (see roles.py). |
| `alias_overrides.csv` | Authored for this thesis | — | Hand-reviewed alias-cluster merges, not yet needed (mailbox-directory-name matching in `roles.py` has been sufficient so far) |

**On the two originally-planned sources:** the CALO/Cohen-style "Enron Employee Information.csv" (Hardin & Sarkis, JSE) was never actually
publicly hosted — even the authors' own tutorial code has the line that loads
it commented out, and the Wayback Machine shows 404 for every snapshot back
to 2018. Agarwal et al.'s (2012, ACL) supervisor-pair gold standard is
real and peer-reviewed, but its data was only ever distributed as a database
file requiring manual restoration, and after checking the paper's official
listing, the hosting university's documentation site, GitHub, and the Wayback
Machine, no live public copy could be found. Both dead ends are logged here
rather than silently swapped out.
