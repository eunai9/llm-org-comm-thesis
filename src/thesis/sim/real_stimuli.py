"""Reply to a real message, not only a synthetic scenario.

Every generated reply so far in this project answers a scenario invented
from scratch (:mod:`thesis.sim.scenario`). That design is deliberate for the
synthetic grid -- it lets every persona face an identical, controlled
situation -- but it also means there has never been a genuinely *paired*
real-vs-generated example: a real reply and a generated reply answering the
exact same incoming message. Without that pairing, the paired statistics in
:mod:`thesis.analysis.fidelity` (paired Wilcoxon, TOST equivalence) have
nothing real to run on -- built and tested against synthetic data, but idle.

This module builds that pairing, from data the project already has:
``S_shots`` (200 real threads, each stimulus = the thread's first message)
and ``S_real_eval`` (the real reply inside each of those same threads,
sampled specifically so the two align -- see ``sampling.py``). For each
thread, this reuses the persona whose (seniority_rank, department) matches
the *real replier's* actual role, and asks it to answer the exact same
incoming message the real person answered.

**Not every thread can be used.** A real replier at rank 6, or outside
Trading/Legal, has no matching persona -- rank 6 is excluded from personas
entirely (smallest population, highest memorization risk; see
``persona.py``), and only two departments were ever modeled. Rather than
force a fallback persona that would misrepresent who actually replied, those
threads are skipped and the count reported, the same honesty this project
has applied to every other real shortfall (``S_real_eval``'s own 302-of-400,
the memory-generation reflection shortfall, and so on).

**Direction is derived, not assumed.** ``S_shots``/``S_real_eval`` carry the
stimulus sender's and the real replier's ``seniority_rank`` directly. Whether
the persona is writing "up", "down", or "lateral" is computed by comparing
the two, exactly the manipulation the rest of the grid already uses --
not a new concept, just this module's version of assigning it from real
roles instead of a synthetic design cell.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import duckdb
import pandas as pd

from thesis.logging_setup import get_logger
from thesis.paths import DATA_DIR, MESSAGES_PARQUET_GLOB
from thesis.sim.grid import GridCell
from thesis.sim.persona import Persona
from thesis.sim.scenario import Direction, Scenario

log = get_logger(__name__)

S_SHOTS_PATH = DATA_DIR / "samples" / "s_shots.parquet"
S_REAL_EVAL_PATH = DATA_DIR / "samples" / "s_real_eval.parquet"


@dataclass(frozen=True, slots=True)
class RealStimulusPair:
    """Everything needed to generate a reply, and to compare it afterward.

    ``cell`` is a normal GridCell -- it flows through build_request() and
    run_grid() exactly like a synthetic-scenario cell, no special-casing
    needed downstream. ``real_reply_text`` and ``stimulus_text`` exist only
    for the calling code's later comparison (judging, fidelity statistics);
    neither is part of the request the persona is asked to answer beyond
    what stimulus_text already contributes as the incoming message.
    """

    thread_id: str
    cell: GridCell
    real_reply_text: str
    stimulus_text: str


def _direction_for(stimulus_rank: int, replier_rank: int) -> Direction:
    """Up/lateral/down from the persona's point of view, replying as replier_rank
    to a message from stimulus_rank."""
    if replier_rank > stimulus_rank:
        return "down"
    if replier_rank < stimulus_rank:
        return "up"
    return "lateral"


def _persona_lookup(personas: Sequence[Persona]) -> dict[tuple[int, str], Persona]:
    return {(p.seniority_rank, p.department): p for p in personas}


def build_real_stimulus_pairs(
    personas: Sequence[Persona],
    role_by_address: Mapping[str, tuple[int, str]],
    model: str,
    role_label: str,
    *,
    messages_glob: str = MESSAGES_PARQUET_GLOB,
    s_shots_path: object = S_SHOTS_PATH,
    s_real_eval_path: object = S_REAL_EVAL_PATH,
) -> list[RealStimulusPair]:
    """Build one pair per usable thread: real stimulus in, ready-to-run cell out.

    ``role_by_address`` is the same address -> (rank, department) mapping
    ``derive_personas`` takes, reused here to find the real replier's actual
    department (S_real_eval carries seniority_rank directly, but not
    department).

    Returned in a stable order (the query is explicitly ORDER BY'd) rather
    than whatever order the database happens to produce -- DuckDB does not
    guarantee row order for a query with no ORDER BY, and was observed to
    return a genuinely different order across separate calls in the same
    process. Anyone sampling a subset of the result with a fixed seed (as
    the fidelity-check scripts do) needs that determinism, or reproducible
    sampling silently is not.
    """
    con = duckdb.connect()
    roles_table = pd.DataFrame(
        [{"address": a, "rank": r, "dept": d} for a, (r, d) in role_by_address.items()]
    )
    con.register("roles", roles_table)

    rows = con.execute(
        """
        SELECT
            shots.thread_id,
            shots.message_uid AS stimulus_uid,
            stim_msg.subject AS stimulus_subject,
            stim_msg.body_clean AS stimulus_body,
            reply.message_uid AS reply_uid,
            reply_msg.subject AS reply_subject,
            reply_msg.body_clean AS reply_body,
            shots.seniority_rank AS stimulus_rank,
            reply.seniority_rank AS replier_rank,
            roles.dept AS replier_department
        FROM read_parquet(?) AS shots
        JOIN read_parquet(?) AS reply USING (thread_id)
        JOIN read_parquet(?) AS stim_msg ON stim_msg.message_uid = shots.message_uid
        JOIN read_parquet(?) AS reply_msg ON reply_msg.message_uid = reply.message_uid
        LEFT JOIN roles ON roles.address = reply.from_addr
        WHERE shots.is_root
        ORDER BY shots.thread_id, reply.message_uid
        """,
        [str(s_shots_path), str(s_real_eval_path), messages_glob, messages_glob],
    ).fetchall()
    con.close()

    columns = [
        "thread_id",
        "stimulus_uid",
        "stimulus_subject",
        "stimulus_body",
        "reply_uid",
        "reply_subject",
        "reply_body",
        "stimulus_rank",
        "replier_rank",
        "replier_department",
    ]
    lookup = _persona_lookup(personas)
    pairs: list[RealStimulusPair] = []
    n_skipped_no_persona = 0

    for values in rows:
        row = dict(zip(columns, values, strict=True))
        persona = lookup.get((row["replier_rank"], row["replier_department"]))
        if persona is None:
            n_skipped_no_persona += 1
            continue

        direction = _direction_for(row["stimulus_rank"], row["replier_rank"])
        stimulus_text = f"Subject: {row['stimulus_subject']}\n\n{row['stimulus_body']}"
        scenario = Scenario(
            scenario_id=f"real_{row['thread_id']}",
            task_type="real_stimulus",
            direction=direction,
            stakes="routine",
            style="neutral",
            situation="A real workplace email exchange.",
            incoming_message=stimulus_text,
        )
        # Full reply_uid, not just thread_id: a thread can have multiple
        # real repliers (up to 11 in this corpus), each matched to a
        # different persona -- keying on thread_id alone collided their
        # cell_ids. Unlike batch.py's custom_id, there is no external length
        # limit on cell_id, so the full id is used rather than a truncated
        # prefix, which a fixture with a long shared prefix showed can
        # collide even though real content-hash uids make it astronomically
        # unlikely in practice.
        cell = GridCell(
            cell_id=f"{role_label}__real_{row['thread_id']}__{row['reply_uid']}",
            persona=persona,
            scenario=scenario,
            replicate=1,
            model=model,
            role_label=role_label,
        )
        pairs.append(
            RealStimulusPair(
                thread_id=row["thread_id"],
                cell=cell,
                real_reply_text=f"Subject: {row['reply_subject']}\n\n{row['reply_body']}",
                stimulus_text=stimulus_text,
            )
        )

    log.info(
        "built %d real-stimulus pair(s); skipped %d thread(s) with no matching persona "
        "(rank 6 or outside Trading/Legal)",
        len(pairs),
        n_skipped_no_persona,
    )
    return pairs
