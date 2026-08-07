"""Typed configuration, loaded from layered YAML files.

Configs are plain YAML in ``configs/``; ``base.yaml`` holds defaults and later
layers override earlier ones. Every model sets ``extra="forbid"`` so a typo in
a key fails loudly at load time instead of being silently ignored -- a config
key that quietly does nothing is a very expensive bug to find in December.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from thesis.paths import CONFIGS_DIR

Provider = Literal["anthropic", "openai"]


class Strict(BaseModel):
    """Base for every config model: reject unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelSpec(Strict):
    """One concrete model, pinned to an exact provider-side identifier."""

    provider: Provider
    model_id: str
    role_label: str = Field(description="Short name used in run manifests and result tables.")


class ModelsConfig(Strict):
    simulator: list[ModelSpec]
    judge: list[ModelSpec]
    labeller: ModelSpec

    def families(self) -> set[Provider]:
        return {m.provider for m in (*self.simulator, *self.judge, self.labeller)}


class CorpusConfig(Strict):
    """Filters defining the eligible message pool. Frozen before sampling."""

    date_start: str
    date_end: str
    min_body_tokens: int
    max_body_tokens: int
    internal_domains: list[str]


class SampleSizes(Strict):
    label: int
    shots: int
    real_eval: int


class PowerConfig(Strict):
    """Weights for the composite power score.

    Freeze this before running validation. Tuning weights until the score
    correlates with seniority is circular, and it is exactly what a defense
    will probe.
    """

    layer_a_weights: dict[str, float]
    layer_b_weights: dict[str, float]


class DataConfig(Strict):
    seed: int
    corpus: CorpusConfig
    sample_sizes: SampleSizes
    power: PowerConfig


class RunConfig(Strict):
    """Guard rails applied to any run that spends money."""

    max_cost_usd: float
    require_clean_git: bool
    use_cache: bool


class Config(Strict):
    project: str
    data: DataConfig
    models: ModelsConfig
    run: RunConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict."""
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)
    with path.open(encoding="utf-8") as handle:
        loaded: Any = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        msg = f"Config file {path} must contain a mapping at the top level"
        raise TypeError(msg)
    return dict(loaded)


def load_config(*names: str, configs_dir: Path | None = None) -> Config:
    """Load and merge the named YAML layers into a validated ``Config``.

    ``load_config("base", "data", "models")`` reads ``configs/base.yaml`` and
    overlays ``data.yaml`` then ``models.yaml``.
    """
    directory = configs_dir or CONFIGS_DIR
    layers = names or ("base", "data", "models")
    merged: dict[str, Any] = {}
    for name in layers:
        merged = _deep_merge(merged, _read_yaml(directory / f"{name}.yaml"))
    return Config.model_validate(merged)
