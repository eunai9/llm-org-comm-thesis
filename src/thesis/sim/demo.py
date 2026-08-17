"""An interactive terminal demo of the simulator, for showing and testing live.

Every other entry point produces a Parquet file meant for pandas, which is the
right format for analysis and the wrong one for a supervisor meeting or for
sitting down and getting a feel for the system. This picks a persona and a
scenario, generates one reply, and prints it formatted for reading -- nothing
here writes to the results store or the cost ledger, because a demo reply is
exploration, not a run.

Defaults to the local Ollama model, so it needs no API key and costs nothing
to run repeatedly during a demo. Every screen states plainly which model
answered and that free-mode output is not a thesis result -- the same
provenance discipline the rest of the project applies, made visible here
rather than only recorded in a file.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from thesis.data.identity import resolve_owners
from thesis.data.roles import build_role_index, load_employees, load_title_rank_table
from thesis.llm.base import LLMClient
from thesis.llm.ollama_client import OllamaClient, OllamaUnavailableError
from thesis.llm.stub_client import StubClient
from thesis.paths import MESSAGES_PARQUET_GLOB
from thesis.sim.grid import GridCell
from thesis.sim.persona import Persona, derive_personas
from thesis.sim.run import build_request
from thesis.sim.scenario import DIRECTIONS, STAKES, TASK_TYPES, Scenario, build_scenarios
from thesis.sim.schemas import validate_response

console = Console()


def _load_personas() -> list[Persona]:
    title_ranks = load_title_rank_table()
    role_index, _ = build_role_index(
        load_employees(), resolve_owners(MESSAGES_PARQUET_GLOB), title_ranks
    )
    return derive_personas(
        {a: (r.seniority_rank, r.department) for a, r in role_index.items()},
        {rank: label for _, (rank, label) in title_ranks.items()},
    )


def _choose(label: str, options: Sequence[str], describe: Sequence[str] | None = None) -> str:
    """A numbered menu. Returns the chosen option's text."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    for i, opt in enumerate(options, start=1):
        extra = f"  [dim]{describe[i - 1]}[/dim]" if describe else ""
        table.add_row(f"[cyan]{i}[/cyan]", opt + extra)
    console.print(f"\n[bold]{label}[/bold]")
    console.print(table)
    choice = IntPrompt.ask("Pick a number", choices=[str(i) for i in range(1, len(options) + 1)])
    return options[choice - 1]


def _choose_persona(personas: Sequence[Persona]) -> Persona:
    labels = [f"{p.rank_label} — {p.department}" for p in personas]
    notes = [
        "(pooled across departments — see PROGRESS.md)" if p.is_pooled else "" for p in personas
    ]
    picked = _choose("Who is writing?", labels, notes)
    return personas[labels.index(picked)]


def _choose_scenario() -> Scenario:
    console.print("\n[bold]What's the situation?[/bold]")
    if not Confirm.ask("Use one of the study's built-in scenarios?", default=True):
        return _custom_scenario()

    task_type = _choose("Task type", TASK_TYPES)
    direction = _choose("Writing to someone…", list(DIRECTIONS))
    stakes = _choose("Stakes", list(STAKES))
    match = next(
        s
        for s in build_scenarios()
        if s.task_type == task_type and s.direction == direction and s.stakes == stakes
    )
    return match


def _custom_scenario() -> Scenario:
    direction = _choose("Writing to someone…", list(DIRECTIONS))
    situation = Prompt.ask("Briefly, the situation")
    incoming = Prompt.ask("The message they received")
    return Scenario(
        scenario_id="custom",
        task_type="custom",
        direction=direction,  # type: ignore[arg-type]
        stakes="routine",
        situation=situation,
        incoming_message=incoming,
    )


def _choose_client() -> LLMClient:
    console.print("\n[bold]Which model should answer?[/bold]")
    console.print("[dim]Free options need no API key. The paid option needs one configured.[/dim]")
    choice = _choose(
        "Backend",
        [
            "Local model (llama3.2:3b, via Ollama) — real generated text, free",
            "Offline stub — templated text, free",
        ],
    )
    if choice.startswith("Offline"):
        return StubClient()

    client = OllamaClient("llama3.2:3b")
    if not client.is_available():
        console.print(
            "[red]No local Ollama server reachable.[/red] Start it with "
            "[bold]ollama serve[/bold] in another terminal, then try again."
        )
        raise OllamaUnavailableError("Ollama not running")
    return client


def _render(persona: Persona, scenario: Scenario, client: LLMClient) -> None:
    # Routed through the same GridCell -> build_request path a real run uses,
    # rather than building the request ad hoc, so a demo reply is generated
    # from an identical prompt to the one the study would actually send.
    cell = GridCell(
        cell_id="demo",
        persona=persona,
        scenario=scenario,
        replicate=1,
        model=getattr(client, "model", getattr(client, "model_label", "unknown")),
        role_label="demo",
    )
    request = build_request(cell, [])

    console.print(
        Panel(
            f"[bold]{persona.rank_label}[/bold], {persona.department}\n"
            f"Writing [bold]{scenario.direction}[/bold] · {scenario.stakes} stakes\n\n"
            f"[dim]Received:[/dim] {scenario.incoming_message}",
            title="Scenario",
            border_style="blue",
        )
    )

    with console.status("Generating…", spinner="dots"):
        response = client.complete(request)

    if response.parsed is None:
        console.print("[red]No parseable structured output was returned.[/red]")
        console.print(response.text)
        return

    try:
        payload = validate_response(response.parsed)
    except Exception as exc:  # Shown to the user, not swallowed.
        console.print(f"[red]Response failed validation: {exc}[/red]")
        console.print(response.parsed)
        return

    is_real = not response.model.startswith(("stub-", "local/"))
    label_style = "green" if is_real else "yellow"
    label_text = response.model if is_real else f"{response.model} — NOT thesis data"

    console.print(
        Panel(
            f"[bold]Subject:[/bold] {payload['subject']}\n\n"
            f"{payload['body']}\n\n"
            f"[dim]Decision:[/dim] [bold]{payload['decision']}[/bold]  "
            f"[dim]Confidence:[/dim] {payload['confidence']}\n"
            f"[dim]Why:[/dim] {payload['reasoning_brief']}",
            title="Generated reply",
            subtitle=f"[{label_style}]{label_text}[/{label_style}]",
            border_style=label_style,
        )
    )


def main() -> None:
    console.print(
        Panel(
            "Pick a role, a scenario, and a model. Nothing here is saved, "
            "billed, or counted as a thesis result — it's for looking at "
            "the simulator's behavior directly.",
            title="Simulator demo",
            border_style="magenta",
        )
    )

    try:
        client = _choose_client()
    except OllamaUnavailableError:
        return

    personas = _load_personas()

    while True:
        persona = _choose_persona(personas)
        scenario = _choose_scenario()
        try:
            _render(persona, scenario, client)
        except Exception as exc:  # Keeps the demo alive on any single failure.
            console.print(f"[red]Something went wrong: {exc}[/red]")

        if not Confirm.ask("\nGenerate another?", default=True):
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
