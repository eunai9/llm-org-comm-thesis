"""Canonical filesystem locations.

Every module resolves paths through this file rather than hard-coding relative
paths, so behaviour never depends on the current working directory. The repo
root is found by walking upwards for ``pyproject.toml``, which works whether
code is invoked via ``python -m``, pytest, or a notebook.
"""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repository root, identified by the presence of pyproject.toml."""
    origin = (start or Path(__file__)).resolve()
    for candidate in (origin, *origin.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    msg = f"Could not locate pyproject.toml above {origin}"
    raise RuntimeError(msg)


REPO_ROOT: Path = find_repo_root()

CONFIGS_DIR: Path = REPO_ROOT / "configs"
NOTEBOOKS_DIR: Path = REPO_ROOT / "notebooks"
THESIS_DIR: Path = REPO_ROOT / "thesis"

DATA_DIR: Path = REPO_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
EXTERNAL_DIR: Path = DATA_DIR / "external"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SAMPLES_DIR: Path = DATA_DIR / "samples"

RUNS_DIR: Path = REPO_ROOT / "runs"
CACHE_DIR: Path = RUNS_DIR / "_cache"

OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
TABLES_DIR: Path = OUTPUTS_DIR / "tables"
MANIFESTS_DIR: Path = OUTPUTS_DIR / "manifests"

# Figures embedded in committed documentation (PROGRESS.md), as distinct
# from FIGURES_DIR, which holds regenerable run outputs and is gitignored.
# A Markdown file in the repository can only render an image that is also
# in the repository, so these have to live somewhere tracked.
DOCS_DIR: Path = REPO_ROOT / "docs"
DOCS_FIGURES_DIR: Path = DOCS_DIR / "figures"

# Well-known artefacts.
ENRON_CSV: Path = RAW_DIR / "emails.csv"
MESSAGES_PARQUET_GLOB: str = str(INTERIM_DIR / "messages" / "part-*.parquet")
RECIPIENTS_PARQUET: Path = INTERIM_DIR / "recipients.parquet"
MESSAGE_STORE: Path = PROCESSED_DIR / "enron.duckdb"
COST_LEDGER: Path = MANIFESTS_DIR / "cost_ledger.csv"

_WRITABLE_DIRS: tuple[Path, ...] = (
    RAW_DIR,
    EXTERNAL_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    SAMPLES_DIR,
    RUNS_DIR,
    CACHE_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    MANIFESTS_DIR,
    DOCS_FIGURES_DIR,
)


def ensure_dirs() -> None:
    """Create every directory the pipeline writes into. Safe to call repeatedly."""
    for directory in _WRITABLE_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
