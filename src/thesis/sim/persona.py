"""Role archetypes for the simulator, derived from corpus aggregates.

**Personas are role archetypes, never named Enron individuals.** The plan is
explicit about why, and both reasons matter:

- *Ethics.* These are real people's real messages. A simulator prompted to
  "write as Jeff Skilling" turns a research artifact into an impersonation of
  an identifiable person.
- *Confounding.* Frontier models have memorized substantial parts of this
  corpus and its cast. A model asked for a named executive may be **recalling**
  rather than **simulating**, which would silently invalidate the Q2 fidelity
  claim -- the thesis would be measuring memorization and reporting it as
  behavioral realism.

So a persona is a (seniority rank x department) cell summarized by aggregate
writing-style statistics, carrying no names, no addresses, and no message text.

**The single-person problem, and the guard for it.** Aggregating does not by
itself de-identify: several cells in this corpus contain exactly one person
(Legal/Manager is one individual with ~2,000 messages; Legal/Managing Director
is one individual with ~136). An "archetype" computed from a single person is
that person's writing style with a label on it -- reintroducing precisely the
identifiability the archetype approach exists to avoid.

:data:`MIN_PEOPLE_PER_PERSONA` therefore sets a floor on how many distinct
people must stand behind a persona. A cell below the floor falls back to the
rank-level aggregate pooled across departments, trading departmental precision
for a defensible claim that no persona describes one identifiable person. Which
personas fell back is recorded on the persona itself and reported, rather than
being an invisible internal detail -- it is a limitation the thesis should
state, not hide.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import duckdb
import pyarrow as pa

from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import EXTERNAL_DIR, INTERIM_DIR, MESSAGES_PARQUET_GLOB

log = get_logger(__name__)

FEATURES_PATH = INTERIM_DIR / "features.parquet"

# A frozen snapshot of derive_personas() output, computed from the real
# corpus and committed to the repo. The corpus itself (~270MB of processed
# Enron mail) is deliberately not committed, so anyone without it -- a
# supervisor testing the demo on their own machine, a reviewer without the
# raw data -- cannot compute personas live. This snapshot lets the demo run
# for them anyway, using the exact numbers the corpus actually produced
# rather than asking them to reprocess 237k messages first.
PERSONAS_SNAPSHOT_PATH = EXTERNAL_DIR / "personas_snapshot.json"

# The two real, named departments in the vendored employee list. The third
# value is a residual "Other" bucket of unrelated functions, which is not a
# coherent organizational unit and would make a meaningless archetype.
DEPARTMENTS: tuple[str, ...] = ("Trading", "Legal")

# Ranks 1-5 (Employee .. Managing Director). Rank 6 (President/CEO) is
# excluded deliberately: it is the smallest population in the corpus and by far
# the most famous, so it is where memorization and re-identification risk are
# both highest -- exactly the combination the archetype design exists to avoid.
RANKS: tuple[int, ...] = (1, 2, 3, 4, 5)

# Minimum distinct people standing behind a persona. Three is a floor, not a
# strong anonymity guarantee; it is chosen to keep the plan's 5x2 grid intact
# while ruling out the one- and two-person cells that are effectively
# individuals.
MIN_PEOPLE_PER_PERSONA = 3

Derivation = Literal["cell", "rank_pooled"]


@dataclass(frozen=True, slots=True)
class PersonaStyle:
    """Aggregate writing-style statistics for one role archetype.

    Every field is a mean over many messages. None of them can reconstruct an
    individual message, and none carries identifying text.
    """

    mean_tokens: float
    mean_recipients: float
    imperative_ratio: float
    hedge_rate: float
    deference_rate: float
    question_ratio: float


@dataclass(frozen=True, slots=True)
class Persona:
    """One role archetype: a rank, a department, and how that role writes."""

    persona_id: str
    seniority_rank: int
    rank_label: str
    department: str
    style: PersonaStyle
    n_people: int
    n_messages: int
    derivation: Derivation

    @property
    def is_pooled(self) -> bool:
        """Whether this persona fell back to a rank-level aggregate."""
        return self.derivation == "rank_pooled"


_AGGREGATE_SQL = """
    SELECT
        r.rank,
        {group_department} AS dept,
        count(*) AS n_msgs,
        count(DISTINCT r.address) AS n_people,
        avg(m.n_tokens_clean) AS mean_tokens,
        avg(m.n_recipients) AS mean_recipients,
        avg(f.imperative_ratio) AS imperative_ratio,
        avg(f.hedge_rate) AS hedge_rate,
        avg(f.deference_rate) AS deference_rate,
        avg(f.question_ratio) AS question_ratio
    FROM read_parquet(?) AS m
    JOIN roles AS r ON r.address = m.from_addr
    JOIN read_parquet(?) AS f USING (message_uid)
    WHERE r.dept IN ('Trading', 'Legal') AND r.rank BETWEEN 1 AND 5
    GROUP BY r.rank, {group_department}
"""


def _role_relation(role_by_address: Mapping[str, tuple[int, str]]) -> pa.Table:
    """Register address -> (rank, department) as a queryable table."""
    return pa.Table.from_pylist(
        [
            {"address": address, "rank": rank, "dept": dept}
            for address, (rank, dept) in role_by_address.items()
        ],
        schema=pa.schema(
            [
                pa.field("address", pa.string()),
                pa.field("rank", pa.int32()),
                pa.field("dept", pa.string()),
            ]
        ),
    )


def _aggregate(
    con: duckdb.DuckDBPyConnection,
    messages_glob: str,
    features_path: str,
    *,
    by_department: bool,
) -> dict[tuple[int, str], dict[str, float]]:
    """Aggregate style statistics, grouped by rank and optionally department."""
    sql = _AGGREGATE_SQL.format(group_department="r.dept" if by_department else "'*'")
    rows = con.execute(sql, [messages_glob, features_path]).fetchall()
    columns = (
        "n_msgs",
        "n_people",
        "mean_tokens",
        "mean_recipients",
        "imperative_ratio",
        "hedge_rate",
        "deference_rate",
        "question_ratio",
    )
    return {(int(row[0]), str(row[1])): dict(zip(columns, row[2:], strict=True)) for row in rows}


def derive_personas(
    role_by_address: Mapping[str, tuple[int, str]],
    rank_labels: Mapping[int, str],
    messages_glob: str = MESSAGES_PARQUET_GLOB,
    features_path: str = str(FEATURES_PATH),
    *,
    min_people: int = MIN_PEOPLE_PER_PERSONA,
) -> list[Persona]:
    """Build one persona per (rank x department) cell.

    Cells with fewer than ``min_people`` distinct senders fall back to the
    rank-level aggregate pooled across departments, so that no persona
    describes a single identifiable person.
    """
    con = duckdb.connect()
    con.register("roles", _role_relation(role_by_address))
    by_cell = _aggregate(con, messages_glob, features_path, by_department=True)
    by_rank = _aggregate(con, messages_glob, features_path, by_department=False)
    con.close()

    personas: list[Persona] = []
    for rank in RANKS:
        for department in DEPARTMENTS:
            cell = by_cell.get((rank, department))
            derivation: Derivation = "cell"

            if cell is None or int(cell["n_people"]) < min_people:
                pooled = by_rank.get((rank, "*"))
                if pooled is None:
                    log.warning("no data for rank %d; skipping both departments", rank)
                    continue
                observed = 0 if cell is None else int(cell["n_people"])
                log.info(
                    "rank %d/%s has %d distinct sender(s) (< %d); "
                    "falling back to the rank-level aggregate",
                    rank,
                    department,
                    observed,
                    min_people,
                )
                cell = pooled
                derivation = "rank_pooled"

            personas.append(
                Persona(
                    persona_id=f"r{rank}_{department.lower()}",
                    seniority_rank=rank,
                    rank_label=rank_labels.get(rank, f"rank {rank}"),
                    department=department,
                    style=PersonaStyle(
                        mean_tokens=round(float(cell["mean_tokens"]), 1),
                        mean_recipients=round(float(cell["mean_recipients"]), 2),
                        imperative_ratio=round(float(cell["imperative_ratio"]), 3),
                        hedge_rate=round(float(cell["hedge_rate"]), 3),
                        deference_rate=round(float(cell["deference_rate"]), 3),
                        question_ratio=round(float(cell["question_ratio"]), 3),
                    ),
                    n_people=int(cell["n_people"]),
                    n_messages=int(cell["n_msgs"]),
                    derivation=derivation,
                )
            )
    return personas


def freeze_personas(personas: Sequence[Persona], path: Path = PERSONAS_SNAPSHOT_PATH) -> None:
    """Write derived personas to disk, so they can be loaded without the corpus.

    Regenerate with ``python -m thesis.sim.persona`` whenever the underlying
    corpus or role data changes -- this is a generated artifact, not a hand
    edited one, and should be treated the same way as any other frozen report
    in this project: reproducible from code, not maintained by hand.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(persona) for persona in personas]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_frozen_personas(path: Path = PERSONAS_SNAPSHOT_PATH) -> list[Persona]:
    """Load personas from a snapshot rather than deriving them from the corpus.

    The only path available to someone who has the code but not the ~270MB
    processed corpus -- see :data:`PERSONAS_SNAPSHOT_PATH`.
    """
    if not path.is_file():
        msg = (
            f"no persona snapshot at {path}. Either process the corpus and "
            "call freeze_personas(), or fetch the committed snapshot from "
            "the repository."
        )
        raise FileNotFoundError(msg)
    records = json.loads(path.read_text(encoding="utf-8"))
    return [
        Persona(
            persona_id=r["persona_id"],
            seniority_rank=r["seniority_rank"],
            rank_label=r["rank_label"],
            department=r["department"],
            style=PersonaStyle(**r["style"]),
            n_people=r["n_people"],
            n_messages=r["n_messages"],
            derivation=r["derivation"],
        )
        for r in records
    ]


def render_persona_block(persona: Persona) -> str:
    """The persona section of the prompt.

    Written as a role description with observed tendencies rather than as
    instructions to imitate a person. The style numbers are given as plain
    descriptions ("typically writes around N words") because a model follows a
    described habit far more reliably than a raw decimal it has to interpret.
    """
    style = persona.style
    return "\n".join(
        [
            f"Your role: {persona.rank_label}, in the {persona.department} "
            "function of a large US energy company in 2001.",
            "",
            "You are a composite role archetype, not any specific individual. "
            "Write the way someone in this position would write, based on the "
            "following observed tendencies of people at this level:",
            "",
            f"- Typical message length: about {style.mean_tokens:.0f} words.",
            f"- Typical number of recipients: about {style.mean_recipients:.1f}.",
            f"- Gives direct instructions in roughly {style.imperative_ratio:.0%} of sentences.",
            f"- Uses hedging language ('maybe', 'I think') in about {style.hedge_rate:.0%}.",
            f"- Uses deferential phrasing in about {style.deference_rate:.0%}.",
            f"- Asks questions in about {style.question_ratio:.0%} of sentences.",
        ]
    )


def collapsed_ranks(personas: Sequence[Persona]) -> list[int]:
    """Ranks where every department pooled to the same rank-level aggregate.

    At such a rank the personas differ only by their department *label*: the
    style statistics behind them are identical, so the department factor of the
    experimental design is not actually varying anything there. That is a real
    limitation of the design rather than a bug, and it needs stating in the
    thesis -- a department effect estimated at such a rank would be measuring
    the label alone.
    """
    by_rank: dict[int, list[Persona]] = {}
    for persona in personas:
        by_rank.setdefault(persona.seniority_rank, []).append(persona)

    collapsed = []
    for rank, group in by_rank.items():
        if len(group) > 1 and all(p.is_pooled for p in group):
            styles = {p.style for p in group}
            if len(styles) == 1:
                collapsed.append(rank)
    return sorted(collapsed)


def main() -> None:
    """Derive personas from the real corpus and freeze them to disk.

    Run with ``python -m thesis.sim.persona`` whenever the corpus or role
    data changes, to keep the committed snapshot in sync with reality.
    """
    from thesis.data.identity import resolve_owners
    from thesis.data.roles import build_role_index, load_employees, load_title_rank_table

    configure_logging()
    title_ranks = load_title_rank_table()
    role_index, _match = build_role_index(
        load_employees(), resolve_owners(MESSAGES_PARQUET_GLOB), title_ranks
    )
    personas = derive_personas(
        {a: (r.seniority_rank, r.department) for a, r in role_index.items()},
        {rank: label for _, (rank, label) in title_ranks.items()},
    )
    freeze_personas(personas)
    log.info("wrote %d persona(s) to %s", len(personas), PERSONAS_SNAPSHOT_PATH)
    log.info(json.dumps(coverage_report(personas), indent=2))


def coverage_report(personas: Sequence[Persona]) -> dict[str, object]:
    """Summary of how the personas were derived, for honest reporting."""
    pooled = [p.persona_id for p in personas if p.is_pooled]
    collapsed = collapsed_ranks(personas)
    return {
        "n_personas": len(personas),
        "min_people_per_persona": MIN_PEOPLE_PER_PERSONA,
        "n_pooled_to_rank_level": len(pooled),
        "pooled_personas": sorted(pooled),
        "ranks_with_no_department_contrast": collapsed,
        "ranks_with_no_department_contrast_note": (
            "at these ranks both departments fell back to the same rank-level "
            "aggregate, so the personas differ only by label; do not interpret "
            "a department effect here"
        ),
        "excluded_rank_6_reason": (
            "smallest population and highest memorization/re-identification "
            "risk; excluded by design"
        ),
    }


if __name__ == "__main__":
    main()
