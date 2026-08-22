"""Real-stimulus pairing tests.

Fixtures write real Parquet stores (messages, S_shots, S_real_eval) so the
SQL runs exactly as it does in production, the same pattern used elsewhere
in this project (identity.py, roles.py, network.py, sampling.py). What's
tested is what this module exists to guarantee: threads with no matching
persona are skipped and counted rather than guessed at, direction is
derived correctly from the two real ranks, cell ids stay unique even when
one thread has several real repliers, and the real reply text never leaks
into the request a persona is asked to answer.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from thesis.sim.persona import Persona, PersonaStyle
from thesis.sim.real_stimuli import build_real_stimulus_pairs

MESSAGE_SCHEMA = pa.schema(
    [
        pa.field("message_uid", pa.string(), nullable=False),
        pa.field("subject", pa.string()),
        pa.field("body_clean", pa.string()),
    ]
)
SHOTS_SCHEMA = pa.schema(
    [
        pa.field("thread_id", pa.string(), nullable=False),
        pa.field("message_uid", pa.string(), nullable=False),
        pa.field("is_root", pa.bool_(), nullable=False),
        pa.field("seniority_rank", pa.int32()),
    ]
)
REAL_EVAL_SCHEMA = pa.schema(
    [
        pa.field("thread_id", pa.string(), nullable=False),
        pa.field("message_uid", pa.string(), nullable=False),
        pa.field("seniority_rank", pa.int32()),
        pa.field("from_addr", pa.string()),
    ]
)


def _write(tmp_path: Path, name: str, rows: list[dict[str, object]], schema: pa.Schema) -> Path:
    path = tmp_path / name
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return path


def _style() -> PersonaStyle:
    return PersonaStyle(
        mean_tokens=50.0,
        mean_recipients=2.0,
        imperative_ratio=0.15,
        hedge_rate=0.03,
        deference_rate=0.005,
        question_ratio=0.09,
    )


def _persona(persona_id: str, rank: int, department: str) -> Persona:
    return Persona(
        persona_id=persona_id,
        seniority_rank=rank,
        rank_label="Director",
        department=department,
        style=_style(),
        n_people=10,
        n_messages=1000,
        derivation="cell",
    )


_PERSONAS = [
    _persona("r2_trading", 2, "Trading"),
    _persona("r3_legal", 3, "Legal"),
    _persona("r4_trading", 4, "Trading"),
]
_ROLE_BY_ADDRESS = {
    "junior@enron.com": (2, "Trading"),
    "senior@enron.com": (4, "Trading"),
    "lawyer@enron.com": (3, "Legal"),
    "ceo@enron.com": (6, "Trading"),  # rank 6 -- no persona exists
    "other_dept@enron.com": (2, "Other"),  # department outside Trading/Legal
}


def _write_fixture(
    tmp_path: Path,
    threads: list[tuple[str, int, str, int]],
    # (thread_id, stimulus_rank, replier_address, replier_rank) per real reply
) -> tuple[Path, Path, Path]:
    messages: list[dict[str, object]] = []
    shots_rows: list[dict[str, object]] = []
    real_eval_rows: list[dict[str, object]] = []
    for thread_id, stimulus_rank, replier_address, replier_rank in threads:
        stim_uid = f"stim_{thread_id}_{len(shots_rows)}"
        reply_uid = f"reply_{thread_id}_{len(real_eval_rows)}"
        messages.append(
            {
                "message_uid": stim_uid,
                "subject": f"Subject for {thread_id}",
                "body_clean": f"Stimulus body {thread_id}",
            }
        )
        messages.append(
            {
                "message_uid": reply_uid,
                "subject": f"Re: {thread_id}",
                "body_clean": f"Real reply body {thread_id}",
            }
        )
        # Only add the stimulus row once per thread (multiple real repliers
        # share the same root message), matching the real S_shots shape.
        if not any(r["thread_id"] == thread_id for r in shots_rows):
            shots_rows.append(
                {
                    "thread_id": thread_id,
                    "message_uid": stim_uid,
                    "is_root": True,
                    "seniority_rank": stimulus_rank,
                }
            )
        real_eval_rows.append(
            {
                "thread_id": thread_id,
                "message_uid": reply_uid,
                "seniority_rank": replier_rank,
                "from_addr": replier_address,
            }
        )

    messages_path = _write(tmp_path, "messages.parquet", messages, MESSAGE_SCHEMA)
    shots_path = _write(tmp_path, "shots.parquet", shots_rows, SHOTS_SCHEMA)
    real_eval_path = _write(tmp_path, "real_eval.parquet", real_eval_rows, REAL_EVAL_SCHEMA)
    return messages_path, shots_path, real_eval_path


def test_direction_is_derived_from_the_two_real_ranks(tmp_path: Path) -> None:
    messages_path, shots_path, real_eval_path = _write_fixture(
        tmp_path,
        [
            ("t_up", 4, "junior@enron.com", 2),  # replier junior, stimulus senior -> up
            ("t_down", 2, "senior@enron.com", 4),  # replier senior, stimulus junior -> down
            ("t_lateral", 2, "junior@enron.com", 2),  # equal ranks -> lateral
        ],
    )
    pairs = build_real_stimulus_pairs(
        _PERSONAS,
        _ROLE_BY_ADDRESS,
        "claude-opus-5",
        "sim_test",
        messages_glob=str(messages_path),
        s_shots_path=shots_path,
        s_real_eval_path=real_eval_path,
    )
    by_thread = {p.thread_id: p for p in pairs}
    assert by_thread["t_up"].cell.scenario.direction == "up"
    assert by_thread["t_down"].cell.scenario.direction == "down"
    assert by_thread["t_lateral"].cell.scenario.direction == "lateral"


def test_thread_with_no_matching_persona_is_skipped(tmp_path: Path) -> None:
    """Rank 6 has no persona (excluded by design); a real replier there must
    be skipped, not force-mapped onto some other persona."""
    messages_path, shots_path, real_eval_path = _write_fixture(
        tmp_path,
        [
            ("t_ok", 2, "junior@enron.com", 2),
            ("t_rank6", 2, "ceo@enron.com", 6),
        ],
    )
    pairs = build_real_stimulus_pairs(
        _PERSONAS,
        _ROLE_BY_ADDRESS,
        "claude-opus-5",
        "sim_test",
        messages_glob=str(messages_path),
        s_shots_path=shots_path,
        s_real_eval_path=real_eval_path,
    )
    assert {p.thread_id for p in pairs} == {"t_ok"}


def test_department_outside_trading_or_legal_is_skipped(tmp_path: Path) -> None:
    messages_path, shots_path, real_eval_path = _write_fixture(
        tmp_path,
        [("t_other", 2, "other_dept@enron.com", 2)],
    )
    pairs = build_real_stimulus_pairs(
        _PERSONAS,
        _ROLE_BY_ADDRESS,
        "claude-opus-5",
        "sim_test",
        messages_glob=str(messages_path),
        s_shots_path=shots_path,
        s_real_eval_path=real_eval_path,
    )
    assert pairs == []


def test_one_thread_with_multiple_real_repliers_yields_multiple_pairs(tmp_path: Path) -> None:
    """A thread can have more than one real reply (up to 11 in this corpus);
    each gets its own persona-matched pair, not just the first."""
    messages_path, shots_path, real_eval_path = _write_fixture(
        tmp_path,
        [
            ("t_multi", 2, "junior@enron.com", 2),
            ("t_multi", 2, "lawyer@enron.com", 3),
        ],
    )
    pairs = build_real_stimulus_pairs(
        _PERSONAS,
        _ROLE_BY_ADDRESS,
        "claude-opus-5",
        "sim_test",
        messages_glob=str(messages_path),
        s_shots_path=shots_path,
        s_real_eval_path=real_eval_path,
    )
    assert len(pairs) == 2
    assert {p.thread_id for p in pairs} == {"t_multi"}


def test_cell_ids_stay_unique_across_multiple_repliers_in_one_thread(tmp_path: Path) -> None:
    """The real bug found running this against the actual corpus: keying
    cell_id on thread_id alone collided two different repliers' cells."""
    messages_path, shots_path, real_eval_path = _write_fixture(
        tmp_path,
        [
            ("t_multi", 2, "junior@enron.com", 2),
            ("t_multi", 2, "lawyer@enron.com", 3),
        ],
    )
    pairs = build_real_stimulus_pairs(
        _PERSONAS,
        _ROLE_BY_ADDRESS,
        "claude-opus-5",
        "sim_test",
        messages_glob=str(messages_path),
        s_shots_path=shots_path,
        s_real_eval_path=real_eval_path,
    )
    cell_ids = [p.cell.cell_id for p in pairs]
    assert len(cell_ids) == len(set(cell_ids))


def test_real_reply_text_does_not_leak_into_the_cells_incoming_message(tmp_path: Path) -> None:
    """The persona must be asked to answer the stimulus only -- never shown
    the real reply it will later be compared against."""
    messages_path, shots_path, real_eval_path = _write_fixture(
        tmp_path,
        [("t1", 2, "junior@enron.com", 2)],
    )
    pairs = build_real_stimulus_pairs(
        _PERSONAS,
        _ROLE_BY_ADDRESS,
        "claude-opus-5",
        "sim_test",
        messages_glob=str(messages_path),
        s_shots_path=shots_path,
        s_real_eval_path=real_eval_path,
    )
    pair = pairs[0]
    assert pair.real_reply_text not in pair.cell.scenario.incoming_message
    assert "Stimulus body t1" in pair.cell.scenario.incoming_message
    assert "Real reply body t1" in pair.real_reply_text


def test_build_real_stimulus_pairs_returns_a_stable_order(tmp_path: Path) -> None:
    """Caught running this against the real corpus: the underlying query had
    no ORDER BY, so DuckDB returned a genuinely different row order across
    separate calls in the same process -- silently breaking any code (like a
    seeded random.sample) that assumed a reproducible input order."""
    messages_path, shots_path, real_eval_path = _write_fixture(
        tmp_path,
        [
            ("t1", 2, "junior@enron.com", 2),
            ("t2", 3, "lawyer@enron.com", 3),
            ("t3", 4, "senior@enron.com", 4),
        ],
    )
    first = build_real_stimulus_pairs(
        _PERSONAS,
        _ROLE_BY_ADDRESS,
        "claude-opus-5",
        "sim_test",
        messages_glob=str(messages_path),
        s_shots_path=shots_path,
        s_real_eval_path=real_eval_path,
    )
    second = build_real_stimulus_pairs(
        _PERSONAS,
        _ROLE_BY_ADDRESS,
        "claude-opus-5",
        "sim_test",
        messages_glob=str(messages_path),
        s_shots_path=shots_path,
        s_real_eval_path=real_eval_path,
    )
    assert [p.thread_id for p in first] == [p.thread_id for p in second]


def test_persona_matches_the_real_repliers_actual_role(tmp_path: Path) -> None:
    messages_path, shots_path, real_eval_path = _write_fixture(
        tmp_path,
        [("t1", 2, "lawyer@enron.com", 3)],
    )
    pairs = build_real_stimulus_pairs(
        _PERSONAS,
        _ROLE_BY_ADDRESS,
        "claude-opus-5",
        "sim_test",
        messages_glob=str(messages_path),
        s_shots_path=shots_path,
        s_real_eval_path=real_eval_path,
    )
    assert pairs[0].cell.persona.persona_id == "r3_legal"


def test_cell_is_ready_to_build_a_request_through_the_normal_path() -> None:
    """No special-casing needed downstream: a real-stimulus cell must work
    with the exact same build_request() every synthetic cell uses."""
    from thesis.sim.run import build_request

    persona = _persona("r2_trading", 2, "Trading")
    from thesis.sim.grid import GridCell
    from thesis.sim.scenario import Scenario

    cell = GridCell(
        cell_id="x",
        persona=persona,
        scenario=Scenario(
            scenario_id="real_t1",
            task_type="real_stimulus",
            direction="up",
            stakes="routine",
            situation="s",
            incoming_message="What's the status?",
        ),
        replicate=1,
        model="claude-opus-5",
        role_label="sim_test",
    )
    request = build_request(cell, [])
    assert "What's the status?" in request.messages[0].content
