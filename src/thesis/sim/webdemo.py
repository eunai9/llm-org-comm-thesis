"""A local, browser-based version of the terminal demo (thesis.sim.demo).

Same purpose, different audience: the terminal demo is fine for the author,
but a supervisor meeting reads much better as a web page with dropdowns and
formatted cards than as a sequence of numbered terminal prompts. The
underlying behavior is identical -- both call the same
:func:`thesis.sim.run.build_request` and the same free-mode clients, and both
apply the same "this is not a thesis result" labeling. This module only
changes the presentation layer.

**Local only, by design.** The server binds to 127.0.0.1 by default: nothing
here is meant to be reachable from outside this machine. It runs entirely on
the free paths (local Ollama model, or the offline stub) -- there is no
"real provider" option in this UI, the same restriction the terminal demo
applies, so a page left open cannot accidentally spend money.

Run with ``python -m thesis.sim.webdemo``, then open the printed URL.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from flask import Flask, jsonify, render_template_string, request

from thesis.llm.base import LLMClient
from thesis.llm.ollama_client import OllamaClient, OllamaUnavailableError
from thesis.llm.stub_client import StubClient
from thesis.logging_setup import get_logger
from thesis.sim.demo import _load_memory, _load_personas
from thesis.sim.grid import GridCell
from thesis.sim.memory import MemoryItem
from thesis.sim.persona import Persona
from thesis.sim.prompt import retrieve_for_group
from thesis.sim.run import build_request
from thesis.sim.scenario import DIRECTIONS, STAKES, TASK_TYPES, TONES, Scenario, build_scenarios
from thesis.sim.schemas import InvalidResponseError, validate_response

log = get_logger(__name__)

app = Flask(__name__)

# Loaded once at startup (from the live corpus if present, else the committed
# snapshot -- see thesis.sim.demo) and reused for every request, rather than
# re-deriving on each page load.
_PERSONAS: list[Persona] = []
_SCENARIOS_BY_KEY: dict[tuple[str, str, str, str], Scenario] = {}
_MEMORY: dict[str, list[MemoryItem]] = {}


def _init_data() -> None:
    global _PERSONAS, _SCENARIOS_BY_KEY, _MEMORY
    _PERSONAS = _load_personas()
    _MEMORY = _load_memory()
    _SCENARIOS_BY_KEY = {(s.task_type, s.direction, s.stakes, s.tone): s for s in build_scenarios()}


def _client_for(backend: str) -> LLMClient:
    if backend == "stub":
        return StubClient()
    if backend == "local":
        client = OllamaClient("llama3.2:3b")
        if not client.is_available():
            raise OllamaUnavailableError(
                "No local Ollama server reachable. Start it with 'ollama serve' "
                "in another terminal, then reload this page."
            )
        return client
    msg = f"unknown backend {backend!r}"
    raise ValueError(msg)


@app.get("/")
def index() -> str:
    return render_template_string(
        _PAGE,
        personas=[asdict(p) for p in _PERSONAS],
        task_types=TASK_TYPES,
        directions=DIRECTIONS,
        stakes=STAKES,
        tones=TONES,
    )


@app.post("/api/generate")
def generate() -> Any:
    payload = request.get_json(force=True, silent=True) or {}

    persona = next((p for p in _PERSONAS if p.persona_id == payload.get("persona_id")), None)
    if persona is None:
        return jsonify({"error": "Unknown persona."}), 400

    scenario: Scenario
    if payload.get("mode") == "custom":
        direction = payload.get("direction")
        situation = (payload.get("situation") or "").strip()
        incoming = (payload.get("incoming") or "").strip()
        if direction not in DIRECTIONS:
            return jsonify({"error": "Pick a direction."}), 400
        if not incoming:
            return jsonify({"error": "Enter the message received."}), 400
        scenario = Scenario(
            scenario_id="custom",
            task_type="custom",
            direction=direction,
            stakes="routine",
            tone="neutral",
            situation=situation or "A workplace situation.",
            incoming_message=incoming,
        )
    else:
        # Coerced to str explicitly: a missing or non-string field then simply
        # fails to match any real key below, which is the correct behavior
        # (falls through to the "pick a..." error) without needing a cast.
        key = (
            str(payload.get("task_type", "")),
            str(payload.get("direction", "")),
            str(payload.get("stakes", "")),
            str(payload.get("tone", "")),
        )
        found = _SCENARIOS_BY_KEY.get(key)
        if found is None:
            return jsonify({"error": "Pick a task type, direction, stakes, and tone."}), 400
        scenario = found

    try:
        client = _client_for(payload.get("backend", "local"))
    except OllamaUnavailableError as exc:
        return jsonify({"error": str(exc)}), 503

    cell = GridCell(
        cell_id="webdemo",
        persona=persona,
        scenario=scenario,
        replicate=1,
        model=getattr(client, "model", getattr(client, "model_label", "unknown")),
        role_label="webdemo",
    )
    memories = retrieve_for_group(persona, scenario.direction, _MEMORY.get(persona.persona_id, []))
    llm_request = build_request(cell, memories)

    try:
        response = client.complete(llm_request)
    except OllamaUnavailableError as exc:
        return jsonify({"error": str(exc)}), 503

    if response.parsed is None:
        return jsonify({"error": "No structured output was returned.", "raw": response.text}), 502

    try:
        result = validate_response(response.parsed)
    except InvalidResponseError as exc:
        return jsonify({"error": f"Response failed validation: {exc}", "raw": response.parsed}), 502

    is_real = not response.model.startswith(("stub-", "local/"))
    return jsonify(
        {
            "subject": result["subject"],
            "body": result["body"],
            "decision": result["decision"],
            "confidence": result["confidence"],
            "reasoning_brief": result["reasoning_brief"],
            "model": response.model,
            "is_real": is_real,
        }
    )


_PAGE = """
<!doctype html>
<title>Simulator demo</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    max-width: 780px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a;
    background: #fafafa;
  }
  h1 { font-size: 1.4rem; margin-bottom: 0.2rem; }
  .subtitle { color: #666; margin-bottom: 1.5rem; font-size: 0.9rem; }
  fieldset { border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; background: #fff; }
  legend { font-weight: 600; padding: 0 0.4rem; }
  label { display: block; margin: 0.6rem 0 0.2rem; font-size: 0.9rem; color: #333; }
  select, textarea, button { font-size: 0.95rem; }
  select, textarea { width: 100%; padding: 0.4rem; border: 1px solid #ccc; border-radius: 6px; }
  textarea { min-height: 4.5rem; resize: vertical; }
  .row { display: flex; gap: 0.8rem; flex-wrap: wrap; }
  .row > div { flex: 1 1 45%; }
  .toggle { display: flex; gap: 1rem; margin-bottom: 0.6rem; }
  #generate {
    width: 100%; padding: 0.7rem; margin-top: 1rem; font-weight: 600;
    background: #2c5cc5; color: white; border: none; border-radius: 6px; cursor: pointer;
  }
  #generate:disabled { background: #99a; cursor: default; }
  #custom-fields { display: none; }
  .card { border-radius: 8px; padding: 1rem 1.2rem; margin-top: 1.2rem; border: 1px solid; }
  .card.real { background: #eefaf0; border-color: #8cc99a; }
  .card.free { background: #fff9e6; border-color: #e0c460; }
  .card.error { background: #fdeeee; border-color: #e0a0a0; }
  .badge { display: inline-block; font-size: 0.78rem; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 4px; margin-bottom: 0.5rem; }
  .badge.real { background: #2f8f4e; color: white; }
  .badge.free { background: #b8901a; color: white; }
  .field-label { color: #666; font-size: 0.82rem; margin-top: 0.6rem; }
  #body-text { white-space: pre-wrap; margin: 0.3rem 0; }
</style>

<h1>Simulator demo</h1>
<div class="subtitle">Pick a role and a scenario, generate a reply. Nothing here is saved, billed, or a thesis result.</div>

<fieldset>
  <legend>Model</legend>
  <label>Backend</label>
  <select id="backend">
    <option value="local">Local model (llama3.2:3b, via Ollama) — real generated text, free</option>
    <option value="stub">Offline stub — templated text, free</option>
  </select>
</fieldset>

<fieldset>
  <legend>Who is writing</legend>
  <label>Role</label>
  <select id="persona">
    {% for p in personas %}
    <option value="{{ p.persona_id }}">{{ p.rank_label }} — {{ p.department }}{% if p.derivation == "rank_pooled" %} (pooled){% endif %}</option>
    {% endfor %}
  </select>
</fieldset>

<fieldset>
  <legend>Situation</legend>
  <div class="toggle">
    <label><input type="radio" name="mode" value="builtin" checked> Built-in scenario</label>
    <label><input type="radio" name="mode" value="custom"> Write my own</label>
  </div>

  <div id="builtin-fields">
    <div class="row">
      <div>
        <label>Task type</label>
        <select id="task_type">
          {% for t in task_types %}<option value="{{ t }}">{{ t.replace("_", " ") }}</option>{% endfor %}
        </select>
      </div>
      <div>
        <label>Writing to someone…</label>
        <select id="direction">
          {% for d in directions %}<option value="{{ d }}">{{ d }}</option>{% endfor %}
        </select>
      </div>
      <div>
        <label>Stakes</label>
        <select id="stakes">
          {% for s in stakes %}<option value="{{ s }}">{{ s }}</option>{% endfor %}
        </select>
      </div>
      <div>
        <label>Tone of the message they received</label>
        <select id="tone">
          {% for t in tones %}<option value="{{ t }}">{{ t }}</option>{% endfor %}
        </select>
      </div>
    </div>
  </div>

  <div id="custom-fields">
    <label>Briefly, the situation</label>
    <textarea id="situation" placeholder="A meeting needs to be rescheduled..."></textarea>
    <label>The message you received</label>
    <textarea id="incoming" placeholder="Can we push our 1pm to tomorrow?"></textarea>
    <label>Writing to someone…</label>
    <select id="direction-custom">
      {% for d in directions %}<option value="{{ d }}">{{ d }}</option>{% endfor %}
    </select>
  </div>
</fieldset>

<button id="generate">Generate reply</button>
<div id="result"></div>

<script>
document.querySelectorAll('input[name="mode"]').forEach(r => r.addEventListener('change', () => {
  const custom = document.querySelector('input[name="mode"]:checked').value === 'custom';
  document.getElementById('builtin-fields').style.display = custom ? 'none' : 'block';
  document.getElementById('custom-fields').style.display = custom ? 'block' : 'none';
}));

document.getElementById('generate').addEventListener('click', async () => {
  const btn = document.getElementById('generate');
  const resultEl = document.getElementById('result');
  const mode = document.querySelector('input[name="mode"]:checked').value;

  const body = {
    backend: document.getElementById('backend').value,
    persona_id: document.getElementById('persona').value,
    mode: mode,
  };
  if (mode === 'custom') {
    body.situation = document.getElementById('situation').value;
    body.incoming = document.getElementById('incoming').value;
    body.direction = document.getElementById('direction-custom').value;
  } else {
    body.task_type = document.getElementById('task_type').value;
    body.direction = document.getElementById('direction').value;
    body.stakes = document.getElementById('stakes').value;
    body.tone = document.getElementById('tone').value;
  }

  btn.disabled = true;
  btn.textContent = 'Generating…';
  resultEl.innerHTML = '';

  try {
    const resp = await fetch('/api/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    const data = await resp.json();

    const card = document.createElement('div');
    if (!resp.ok) {
      card.className = 'card error';
      const p = document.createElement('div');
      p.textContent = data.error || 'Something went wrong.';
      card.appendChild(p);
    } else {
      card.className = 'card ' + (data.is_real ? 'real' : 'free');
      const badge = document.createElement('div');
      badge.className = 'badge ' + (data.is_real ? 'real' : 'free');
      badge.textContent = data.is_real ? data.model : data.model + ' — NOT thesis data';
      card.appendChild(badge);

      const subj = document.createElement('div');
      subj.innerHTML = '<strong>Subject: </strong>';
      subj.appendChild(document.createTextNode(data.subject));
      card.appendChild(subj);

      const bodyEl = document.createElement('div');
      bodyEl.id = 'body-text';
      bodyEl.textContent = data.body;
      card.appendChild(bodyEl);

      const meta = document.createElement('div');
      meta.className = 'field-label';
      meta.textContent = 'Decision: ' + data.decision + '   Confidence: ' + data.confidence;
      card.appendChild(meta);

      const why = document.createElement('div');
      why.className = 'field-label';
      why.textContent = 'Why: ' + data.reasoning_brief;
      card.appendChild(why);
    }
    resultEl.appendChild(card);
  } catch (e) {
    resultEl.innerHTML = '<div class="card error">Request failed: ' + e + '</div>';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Generate reply';
  }
});
</script>
"""


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address. Default is localhost-only; use 0.0.0.0 only if you "
        "understand this exposes the page to your network.",
    )
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    _init_data()
    log.info("open http://127.0.0.1:%d in a browser", args.port)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
