# Running the simulator demo (no thesis environment required)

This lets anyone with the code — no processed corpus, no API keys, no
existing setup — try the email-writing simulator on their own machine.
Nothing here costs money or needs a login: it runs a small AI model locally.

Setup takes about 15–20 minutes, most of it a one-time ~2GB download.

## 1. Get the code

```
git clone https://github.com/eunai9/llm-org-comm-thesis.git
cd llm-org-comm-thesis
```

(Or download the ZIP from that page and extract it, if you don't use git.)

## 2. Set up Python

Needs Python 3.12 or newer ([python.org/downloads](https://www.python.org/downloads/) if you don't have it).

```
python3 -m venv .venv
```

Then activate it — the command differs by OS:

| OS | Command |
|---|---|
| macOS / Linux | `source .venv/bin/activate` |
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (cmd.exe) | `.venv\Scripts\activate.bat` |

Then install dependencies and the project itself (a couple of minutes):

```
pip install -r requirements.txt
pip install -e .
```

(the second line matters — without it, `python -m thesis.sim.demo` fails with
`No module named thesis.sim.demo`)

## 3. Install Ollama (runs the AI model locally)

Download from **[ollama.com/download](https://ollama.com/download)** and install
it — same idea as installing any desktop app, no admin rights normally needed.

On macOS and Windows, Ollama starts automatically in the background after
install. **On Linux**, open a terminal and run:

```
ollama serve
```

(leave that terminal open, or run it in the background)

Then, in your original terminal, download the model (~2GB, one-time):

```
ollama pull llama3.2:3b
```

## 4. Run the demo

Two versions of the same thing — pick whichever you prefer.

**Web version** (a page in your browser — easiest to look at, easiest to show
someone else on the same screen):

```
python -m thesis.sim.webdemo
```

Then open **http://localhost:5050** in your browser. Pick a role and a
scenario with the dropdowns, click "Generate reply". Leave the terminal
window open while you use it — closing it stops the page from working.

**Terminal version** (same underlying behavior, keyboard-driven):

```
python -m thesis.sim.demo
```

Follow the prompts: pick "Local model" when asked which backend, then pick a
role and a scenario.

Either way, a reply takes about 15–30 seconds to generate.

**What you'll see the first time:**
```
No processed corpus found locally -- using the committed persona
snapshot (the same numbers, computed once from the real data).
```
That's expected and correct — it means you're using real numbers computed
from the actual dataset, without needing the multi-hundred-megabyte processed
corpus that isn't included in the repository. (The web version shows this in
the terminal it was launched from, not on the page itself.)

## What this is, and isn't

- **Free and unlimited.** Generate as many replies as you like — nothing is
  billed and nothing is saved anywhere.
- **Not a thesis result.** Every reply is clearly marked
  `local/llama3.2:3b — NOT thesis data`. This is a small model running on a
  laptop, standing in for the much larger models the actual thesis will use.
  It's for seeing how the system works, not for judging research quality.
- **Bodies may look short**, and occasionally a reply's stated decision
  doesn't perfectly match its content. Both are known limits of the small
  local model, not bugs in the setup.

## If something doesn't work

- Either command fails immediately mentioning Ollama → the server isn't
  running. Try `ollama serve` in a separate terminal (Linux), or check the
  Ollama app is running (macOS/Windows).
- The web version's page loads but "Generate reply" shows an error → same
  cause as above; the page itself doesn't need Ollama to load, only to
  generate.
- http://localhost:5050 doesn't load at all → confirm the terminal running
  `python -m thesis.sim.webdemo` is still open and didn't print an error.
- `No module named thesis.sim.demo` → step 2's second `pip install` line
  (`pip install -e .`) was skipped.
- `ensurepip is not available` when creating the venv (Debian/Ubuntu) → run
  `sudo apt install python3.12-venv` (adjust the version number to match
  `python3 --version`), then redo step 2.
- Installation errors during `pip install` → confirm `python3 --version` is
  3.12 or newer.
- Anything else → send the exact error text back to the thesis author.
