"""Role-resolution tests.

Real names from the vendored employee list back the mailbox-key tests, since
the whole design bet is that "surname-firstinitial" reliably reproduces the
actual Enron maildir directory name for real people, not just tidy fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from thesis.data.identity import MailboxOwner
from thesis.data.roles import (
    Employee,
    build_role_index,
    derive_mailbox_key,
    load_employees,
    load_title_rank_table,
    match_employees,
    rank_label_relation,
)


def employee(
    employee_id: int,
    name: str,
    *,
    department: str = "Trading",
    department_long: str = "ENA Gas West",
    title: str = "Trader",
    gender: str = "Male",
    given_seniority: str = "Junior",
) -> Employee:
    return Employee(employee_id, name, department, department_long, title, gender, given_seniority)


def owner(mailbox: str, address: str) -> MailboxOwner:
    return MailboxOwner(mailbox, address, (address,), (), n_sent=10)


# --------------------------------------------------------- mailbox key derivation


@pytest.mark.parametrize(
    ("full_name", "expected"),
    [
        ("John Arnold", "arnold-j"),
        ("Harry Arora", "arora-h"),
        ("Sally Beck", "beck-s"),
        ("Phillip K. Allen", "allen-p"),  # middle initial must not shift the key
        ("Vince J Kaminski", "kaminski-v"),
        ("Don Baughman Jr.", "baughman-d"),  # generational suffix stripped
        ("Louise Kitchen III", "kitchen-l"),
    ],
)
def test_derive_mailbox_key_matches_real_enron_convention(full_name: str, expected: str) -> None:
    assert derive_mailbox_key(full_name) == expected


def test_derive_mailbox_key_on_sparse_input() -> None:
    assert derive_mailbox_key("Cher") is None
    assert derive_mailbox_key("") is None
    assert derive_mailbox_key("   ") is None


# ---------------------------------------------------------------------- matching


def test_matches_employee_to_mailbox_by_derived_key() -> None:
    employees = [employee(1, "Sally Beck")]
    owners = [owner("beck-s", "sally.beck@enron.com")]
    result = match_employees(employees, owners)
    assert result.matched == {"beck-s": employees[0]}
    assert not result.unmatched_employees
    assert not result.unmatched_mailboxes


def test_unmatched_employee_is_reported_not_dropped_silently() -> None:
    employees = [employee(1, "Nobody Here")]
    result = match_employees(employees, [])
    assert result.matched == {}
    assert result.unmatched_employees == (employees[0],)


def test_unmatched_mailbox_is_reported() -> None:
    owners = [owner("ghost-g", "ghost@enron.com")]
    result = match_employees([], owners)
    assert result.unmatched_mailboxes == ("ghost-g",)


def test_two_employees_sharing_a_derived_key_are_left_unmatched() -> None:
    """A collision must not be guessed at -- same conservative rule as identity.py."""
    employees = [employee(1, "Sally Beck"), employee(2, "Steve Beck")]
    owners = [owner("beck-s", "sally.beck@enron.com")]
    result = match_employees(employees, owners)
    assert result.matched == {}
    assert result.ambiguous_keys == ("beck-s",)
    assert set(result.unmatched_employees) == set(employees)


# ------------------------------------------------------------------- role index


def test_build_role_index_carries_department_and_rank() -> None:
    employees = [employee(1, "Sally Beck", department="Other", title="VP")]
    owners = [
        MailboxOwner(
            "beck-s", "sally.beck@enron.com", ("sally.beck@enron.com", "s.beck@enron.com"), (), 10
        )
    ]
    title_ranks = {"VP": (4, "Vice President")}

    role_index, match = build_role_index(employees, owners, title_ranks)

    assert match.matched == {"beck-s": employees[0]}
    # Every alias of the matched mailbox gets the same role.
    assert set(role_index) == {"sally.beck@enron.com", "s.beck@enron.com"}
    role = role_index["s.beck@enron.com"]
    assert role.department == "Other"
    assert role.seniority_rank == 4
    assert role.rank_label == "Vice President"
    assert role.mailbox == "beck-s"


def test_unmatched_mailbox_contributes_no_role() -> None:
    owners = [owner("ghost-g", "ghost@enron.com")]
    role_index, _ = build_role_index([], owners, {})
    assert role_index == {}


# --------------------------------------------------------------------- crosstab


def test_rank_label_relation_counts_by_given_seniority() -> None:
    employees = [
        employee(1, "Sally Beck", title="VP", given_seniority="Senior"),
        employee(2, "Robert Badeer", title="Manager", given_seniority="Junior"),
    ]
    owners = [owner("beck-s", "a@enron.com"), owner("badeer-r", "b@enron.com")]
    title_ranks = {"VP": (4, "Vice President"), "Manager": (2, "Manager")}

    role_index, _ = build_role_index(employees, owners, title_ranks)
    crosstab = rank_label_relation(list(role_index.values()))

    assert crosstab["Vice President"] == {"Senior": 1}
    assert crosstab["Manager"] == {"Junior": 1}


# ------------------------------------------------------- vendored file integrity


def test_vendored_employees_file_parses(tmp_path: Path) -> None:
    """Guards against the vendored file silently changing shape."""
    from thesis.paths import EXTERNAL_DIR

    path = EXTERNAL_DIR / "enron_employees.tsv"
    if not path.is_file():
        pytest.skip("vendored employee file not present in this environment")
    employees = load_employees(path)
    assert len(employees) > 100
    assert all(e.full_name for e in employees)
    assert {e.given_seniority for e in employees} <= {"Junior", "Senior"}


def test_title_rank_table_covers_every_vendored_title() -> None:
    """The frozen table must have zero drift from the file it was built from."""
    from thesis.paths import EXTERNAL_DIR

    employees_path = EXTERNAL_DIR / "enron_employees.tsv"
    ranks_path = EXTERNAL_DIR / "title_to_rank.csv"
    if not employees_path.is_file() or not ranks_path.is_file():
        pytest.skip("vendored files not present in this environment")

    employees = load_employees(employees_path)
    ranks = load_title_rank_table(ranks_path)
    missing = {e.title for e in employees} - set(ranks)
    assert not missing, f"titles missing from the frozen rank table: {missing}"


def test_title_rank_table_is_a_monotone_ladder() -> None:
    """Ranks 1..6 should each be used, with no gaps -- a sanity check on the
    hand-authored table, not a claim about the data itself."""
    from thesis.paths import EXTERNAL_DIR

    ranks_path = EXTERNAL_DIR / "title_to_rank.csv"
    if not ranks_path.is_file():
        pytest.skip("vendored file not present in this environment")

    table = load_title_rank_table(ranks_path)
    used_ranks = {rank for rank, _label in table.values()}
    assert used_ranks == {1, 2, 3, 4, 5, 6}
