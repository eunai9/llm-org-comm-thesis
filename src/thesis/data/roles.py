"""Join mailbox owners to their job title and seniority rank.

Two things are combined:

1. :data:`data/external/enron_employees.tsv` -- 156 employees with name,
   department, title, gender, and a coarse Junior/Senior label. Provenance and
   why the two originally-planned sources were abandoned are documented in
   ``data/external/SOURCES.md``.
2. :data:`data/external/title_to_rank.csv` -- a 6-level ordinal ladder (1
   Employee .. 6 President/CEO), hand-enumerated from all 36 distinct titles
   and frozen *before* any coverage number was computed. Tuning this mapping
   after seeing how well it predicts something would be circular.

The join key is the mailbox directory name, not fuzzy name matching. Enron's
maildir convention names each mailbox ``surname-firstinitial`` (occasionally
with a numeric suffix to break a collision), and the employee list uses full
names -- so the same key can be derived from both sides and compared exactly.
This is far more reliable than comparing free-text display names, which vary
in punctuation, routing decoration, and capitalization.

Run with ``python -m thesis.data.roles``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pyarrow as pa

from thesis.data.identity import MailboxOwner, owner_address_index, resolve_owners
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import EXTERNAL_DIR, INTERIM_DIR, MESSAGES_PARQUET_GLOB, ensure_dirs

log = get_logger(__name__)

EMPLOYEES_PATH = EXTERNAL_DIR / "enron_employees.tsv"
TITLE_RANK_PATH = EXTERNAL_DIR / "title_to_rank.csv"

_SUFFIXES = re.compile(r"\b(jr|sr|ii|iii|iv)\.?$", re.IGNORECASE)
_NON_ALPHA = re.compile(r"[^a-z]")


@dataclass(frozen=True, slots=True)
class Employee:
    """One row of the vendored employee list."""

    employee_id: int
    full_name: str
    department: str
    department_long: str
    title: str
    gender: str
    given_seniority: str


@dataclass(frozen=True, slots=True)
class RoleInfo:
    """The final, per-address answer to "what was this person's role?"."""

    employee_id: int
    full_name: str
    mailbox: str
    department: str
    department_long: str
    title: str
    seniority_rank: int
    rank_label: str
    given_seniority: str


def load_employees(path: Path = EMPLOYEES_PATH) -> list[Employee]:
    """Parse the vendored TSV. No header row; columns are positional."""
    employees: list[Employee] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            eid, name, dept, dept_long, title, gender, seniority = line.rstrip("\n").split("\t")
            employees.append(
                Employee(
                    employee_id=int(eid),
                    full_name=name,
                    department=dept,
                    department_long=dept_long,
                    title=title,
                    gender=gender,
                    given_seniority=seniority,
                )
            )
    return employees


def load_title_rank_table(path: Path = TITLE_RANK_PATH) -> dict[str, tuple[int, str]]:
    """Load the frozen title -> (ordinal rank, label) table."""
    table: dict[str, tuple[int, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            table[row["title"]] = (int(row["seniority_rank"]), row["rank_label"])
    return table


def derive_mailbox_key(full_name: str) -> str | None:
    """Derive the Enron ``surname-firstinitial`` mailbox key from a full name.

    Robust to a middle name or initial (an extra middle token is simply
    ignored -- the rule uses the *first* token's initial, not the
    second-to-last) and to a trailing generational suffix ("Jr.", "III").
    Returns None for input too sparse to derive a key from.
    """
    tokens = [t for t in re.split(r"\s+", full_name.strip()) if t]
    tokens = [t for t in tokens if not _SUFFIXES.fullmatch(t)]
    if len(tokens) < 2:
        return None
    first, surname = tokens[0], tokens[-1]
    surname_clean = _NON_ALPHA.sub("", surname.lower())
    initial = _NON_ALPHA.sub("", first.lower())[:1]
    if not surname_clean or not initial:
        return None
    return f"{surname_clean}-{initial}"


@dataclass(frozen=True, slots=True)
class MatchResult:
    matched: dict[str, Employee]  # mailbox -> employee
    unmatched_employees: tuple[Employee, ...]
    unmatched_mailboxes: tuple[str, ...]
    ambiguous_keys: tuple[str, ...]  # derived key matched >1 mailbox candidate


def match_employees(employees: Sequence[Employee], owners: Sequence[MailboxOwner]) -> MatchResult:
    """Match employees to mailboxes via the derived directory-name key.

    A key claimed by more than one employee is left unmatched rather than
    assigned to either -- the same conservative rule :func:`identity.
    owner_address_index` applies to contested addresses.
    """
    by_mailbox = {owner.mailbox: owner for owner in owners}

    key_to_employees: dict[str, list[Employee]] = {}
    for employee in employees:
        key = derive_mailbox_key(employee.full_name)
        if key is not None:
            key_to_employees.setdefault(key, []).append(employee)

    matched: dict[str, Employee] = {}
    ambiguous: list[str] = []
    for key, candidates in key_to_employees.items():
        if len(candidates) > 1:
            ambiguous.append(key)
            continue
        if key in by_mailbox:
            matched[key] = candidates[0]

    matched_employee_ids = {e.employee_id for e in matched.values()}
    unmatched_employees = tuple(e for e in employees if e.employee_id not in matched_employee_ids)
    unmatched_mailboxes = tuple(sorted(set(by_mailbox) - set(matched)))

    return MatchResult(
        matched=matched,
        unmatched_employees=unmatched_employees,
        unmatched_mailboxes=unmatched_mailboxes,
        ambiguous_keys=tuple(sorted(ambiguous)),
    )


def build_role_index(
    employees: Sequence[Employee],
    owners: Sequence[MailboxOwner],
    title_ranks: dict[str, tuple[int, str]],
) -> tuple[dict[str, RoleInfo], MatchResult]:
    """Build address -> :class:`RoleInfo` for every address of a matched mailbox."""
    match = match_employees(employees, owners)
    address_to_mailbox = owner_address_index(owners)

    by_mailbox_role: dict[str, RoleInfo] = {}
    for mailbox, employee in match.matched.items():
        rank, label = title_ranks[employee.title]
        by_mailbox_role[mailbox] = RoleInfo(
            employee_id=employee.employee_id,
            full_name=employee.full_name,
            mailbox=mailbox,
            department=employee.department,
            department_long=employee.department_long,
            title=employee.title,
            seniority_rank=rank,
            rank_label=label,
            given_seniority=employee.given_seniority,
        )

    role_index = {
        address: by_mailbox_role[mailbox]
        for address, mailbox in address_to_mailbox.items()
        if mailbox in by_mailbox_role
    }
    return role_index, match


def rank_label_relation(rows: Sequence[RoleInfo]) -> dict[str, dict[str, int]]:
    """Cross-tabulate the derived rank label against the source's own label.

    This is the validation the frozen table earns by being frozen: an
    independent cross-check computed *after* the fact, not a knob tuned to
    produce agreement.
    """
    table: dict[str, dict[str, int]] = {}
    for row in rows:
        table.setdefault(row.rank_label, {}).setdefault(row.given_seniority, 0)
        table[row.rank_label][row.given_seniority] += 1
    return table


def role_coverage(messages_glob: str, role_index: dict[str, RoleInfo]) -> dict[str, int | float]:
    """Measure what share of the corpus (and of the eligible pool) has a role.

    Addresses are bound via a registered table rather than interpolated into
    SQL, for the same reason as :func:`thesis.data.identity.sender_coverage`
    -- a real Enron address containing an apostrophe breaks string
    interpolation, not just in theory but in this exact corpus.
    """
    con = duckdb.connect()
    table = pa.table({"address": pa.array(sorted(role_index), type=pa.string())})
    con.register("known", table)

    row = con.execute(
        """
        SELECT
            count(*)                                                        AS total,
            count(*) FILTER (WHERE from_addr IN (SELECT address FROM known)) AS known,
            count(*) FILTER (
                WHERE NOT is_empty_after_clean
                  AND n_tokens_clean BETWEEN 20 AND 600
                  AND date BETWEEN '1999-01-01' AND '2002-06-30'
                  AND from_addr LIKE '%@enron.com'
            ) AS eligible,
            count(*) FILTER (
                WHERE NOT is_empty_after_clean
                  AND n_tokens_clean BETWEEN 20 AND 600
                  AND date BETWEEN '1999-01-01' AND '2002-06-30'
                  AND from_addr LIKE '%@enron.com'
                  AND from_addr IN (SELECT address FROM known)
            ) AS eligible_known
        FROM read_parquet(?)
        """,
        [messages_glob],
    ).fetchone()
    con.close()

    assert row is not None
    total, known, eligible, eligible_known = (int(v) for v in row)
    return {
        "total_messages": total,
        "messages_with_role": known,
        "share_of_total": round(known / total, 4) if total else 0.0,
        "eligible_messages": eligible,
        "eligible_with_role": eligible_known,
        "share_of_eligible": round(eligible_known / eligible, 4) if eligible else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", default=MESSAGES_PARQUET_GLOB)
    parser.add_argument("--employees", type=Path, default=EMPLOYEES_PATH)
    parser.add_argument("--title-ranks", type=Path, default=TITLE_RANK_PATH)
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    employees = load_employees(args.employees)
    title_ranks = load_title_rank_table(args.title_ranks)
    owners = resolve_owners(args.messages)

    role_index, match = build_role_index(employees, owners, title_ranks)
    coverage = role_coverage(args.messages, role_index)
    crosstab = rank_label_relation(list(role_index.values()))

    matched_mailboxes = len(match.matched)
    unmatched_employees = len(match.unmatched_employees)
    unmatched_mailboxes = len(match.unmatched_mailboxes)
    addresses_with_role = len(role_index)
    unmatched_names = sorted(e.full_name for e in match.unmatched_employees)[:20]
    eligible_with_role = int(coverage["eligible_with_role"])
    eligible_messages = int(coverage["eligible_messages"])
    share_of_eligible = float(coverage["share_of_eligible"])

    report: dict[str, object] = {
        "employees_in_source": len(employees),
        "mailboxes_with_outgoing_mail": len(owners),
        "matched_mailboxes": matched_mailboxes,
        "unmatched_employees": unmatched_employees,
        "unmatched_mailboxes": unmatched_mailboxes,
        "ambiguous_keys": list(match.ambiguous_keys),
        "addresses_with_role": addresses_with_role,
        "coverage": coverage,
        "rank_label_vs_given_seniority": crosstab,
        "unmatched_employee_names": unmatched_names,
    }
    (INTERIM_DIR / "roles_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    log.info("employees in source        %8d", len(employees))
    log.info("matched to a mailbox       %8d", matched_mailboxes)
    log.info("unmatched employees        %8d", unmatched_employees)
    log.info("unmatched mailboxes        %8d", unmatched_mailboxes)
    log.info("addresses with a role      %8d", addresses_with_role)
    log.info(
        "eligible messages w/ role  %8d / %8d  (%.1f%%)",
        eligible_with_role,
        eligible_messages,
        100 * share_of_eligible,
    )
    if match.unmatched_employees:
        log.info("first unmatched names: %s", unmatched_names[:5])


if __name__ == "__main__":
    main()
