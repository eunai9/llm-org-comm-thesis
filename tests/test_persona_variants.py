from dataclasses import replace

from thesis.sim.persona import Persona, PersonaStyle, render_persona_block
from thesis.sim.scenario import build_scenarios, render_scenario_block

_STYLE = PersonaStyle(
    mean_tokens=60.0,
    mean_recipients=2.0,
    imperative_ratio=0.15,
    hedge_rate=0.03,
    deference_rate=0.01,
    question_ratio=0.1,
)


def _base(rank: int = 3, department: str = "Trading") -> Persona:
    return Persona(
        persona_id=f"r{rank}_{department.lower()}",
        seniority_rank=rank,
        rank_label="Director",
        department=department,
        style=_STYLE,
        n_people=10,
        n_messages=100,
        derivation="cell",
    )


def test_persona_block_without_trait_is_unchanged() -> None:
    assert render_persona_block(_base()) == "\n".join(
        [
            "Your role: Director, in the Trading function of a large US energy company in 2001.",
            "",
            "You are a composite role archetype, not any specific individual. "
            "Write the way someone in this position would write, based on the "
            "following observed tendencies of people at this level:",
            "",
            "- Typical message length: about 60 words.",
            "- Typical number of recipients: about 2.0.",
            "- Gives direct instructions in roughly 15% of sentences.",
            "- Uses hedging language ('maybe', 'I think') in about 3%.",
            "- Uses deferential phrasing in about 1%.",
            "- Asks questions in about 10% of sentences.",
        ]
    )


def test_persona_block_puts_the_trait_after_the_role_line() -> None:
    block = render_persona_block(
        replace(_base(), trait="You joined from another energy firm about 2 years ago.")
    )
    lines = block.splitlines()
    assert lines[1] == "You joined from another energy firm about 2 years ago."
    assert block.replace(
        "You joined from another energy firm about 2 years ago.\n", ""
    ) == render_persona_block(_base())


def test_scenario_block_without_receiver_is_unchanged() -> None:
    scenario = next(s for s in build_scenarios() if s.direction == "up")
    block = render_scenario_block(scenario)
    assert block.splitlines()[3] == "You are writing to someone senior to you."


def test_scenario_block_names_the_receiver() -> None:
    scenario = replace(
        next(s for s in build_scenarios() if s.direction == "up"),
        receiver="a Vice President in Trading",
    )
    lines = render_scenario_block(scenario).splitlines()
    assert lines[3] == "You are writing to a Vice President in Trading."
    assert "senior to you" not in "\n".join(lines)
