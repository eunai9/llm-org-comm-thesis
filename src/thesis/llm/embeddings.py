"""Sentence embeddings from a locally-run model, cached on disk.

The validation work needs one vector per message so that real and generated
replies can be placed in a common space and looked at (t-SNE) and measured
(nearest-neighbour separability). Three properties decide the implementation:

- **It has to be free.** The project has no paid API budget, so embeddings come
  from an open-weights model served by the same local Ollama instance that
  generates the replies -- ``nomic-embed-text`` by default.
- **It has to be stable across re-runs.** A t-SNE map that shifts every time
  the analysis is re-run cannot be discussed in a supervision meeting. Vectors
  are therefore cached by ``sha256(model + text)``, exactly the same argument
  the response cache in :mod:`thesis.llm.cache` makes about generations: the
  model call is the unreproducible part, so it is archived and everything
  downstream reads the archive.
- **It must never be mistaken for a paid call.** Nothing here touches the cost
  ledger, because nothing here costs anything.

The cache is a single Parquet file rather than one file per vector: these are
thousands of small fixed-width rows, which is the case Parquet handles well and
a directory of tiny JSON files handles badly.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from thesis.llm.ollama_client import DEFAULT_HOST
from thesis.logging_setup import get_logger
from thesis.paths import RUNS_DIR

log = get_logger(__name__)

DEFAULT_EMBED_MODEL = "nomic-embed-text"
EMBED_CACHE_PATH: Path = RUNS_DIR / "_embeddings.parquet"

# Ollama holds the whole batch in memory before replying; 32 short emails is
# comfortable on the 8GB WSL instance and still amortises the request overhead.
BATCH_SIZE = 32
TIMEOUT_SECONDS = 300.0


class EmbeddingUnavailableError(RuntimeError):
    """Raised when the local embedding model cannot be reached."""


def _key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    frame = pd.read_parquet(path)
    return {row.key: np.asarray(row.vector, dtype=np.float32) for row in frame.itertuples()}


def _save_cache(cache: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {"key": list(cache), "vector": [v.astype(np.float32).tolist() for v in cache.values()]}
    )
    frame.to_parquet(path, compression="zstd", index=False)


def _request_embeddings(texts: Sequence[str], model: str, host: str) -> list[list[float]]:
    try:
        response = httpx.post(
            f"{host.rstrip('/')}/api/embed",
            json={"model": model, "input": list(texts)},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        msg = (
            f"could not embed with {model} at {host}: {exc}. Start Ollama with "
            f"'ollama serve' and pull the model with 'ollama pull {model}'."
        )
        raise EmbeddingUnavailableError(msg) from exc
    embeddings: list[list[float]] = response.json()["embeddings"]
    return embeddings


def embed_texts(
    texts: Sequence[str],
    *,
    model: str = DEFAULT_EMBED_MODEL,
    host: str = DEFAULT_HOST,
    cache_path: Path = EMBED_CACHE_PATH,
) -> np.ndarray:
    """Embed every text, reusing cached vectors and storing newly computed ones.

    Returns an ``(len(texts), dim)`` array in the order given. Duplicate texts
    are embedded once and shared, which matters because the real corpus repeats
    boilerplate replies verbatim.
    """
    cache = _load_cache(cache_path)
    wanted = {_key(model, text): text for text in texts}
    missing = [text for key, text in wanted.items() if key not in cache]

    for start in range(0, len(missing), BATCH_SIZE):
        batch = missing[start : start + BATCH_SIZE]
        for text, vector in zip(batch, _request_embeddings(batch, model, host), strict=True):
            cache[_key(model, text)] = np.asarray(vector, dtype=np.float32)
        log.info("embedded %d/%d new text(s)", min(start + BATCH_SIZE, len(missing)), len(missing))

    if missing:
        _save_cache(cache, cache_path)

    return np.vstack([cache[_key(model, text)] for text in texts])


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, so a dot product is a cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized: np.ndarray = vectors / np.maximum(norms, 1e-12)
    return normalized
