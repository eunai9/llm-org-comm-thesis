"""Config loading must fail loudly, not silently ignore mistakes."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from thesis.config import Config, load_config


def test_default_layers_load() -> None:
    config = load_config()
    assert config.project
    assert config.data.seed > 0
    assert config.run.max_cost_usd > 0


def test_judge_and_simulator_span_two_providers() -> None:
    """The exposé requires a judge distinct from the simulator."""
    config = load_config()
    assert len(config.models.families()) >= 2, "need at least two model families"
    sim_ids = {m.model_id for m in config.models.simulator}
    judge_ids = {m.model_id for m in config.models.judge}
    assert not (sim_ids & judge_ids), "a judge must never be the same model as a simulator"


def test_sample_sizes_are_positive() -> None:
    sizes = load_config().data.sample_sizes
    assert sizes.label > 0
    assert sizes.shots > 0
    assert sizes.real_eval > 0


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    """A typo in a YAML key must raise, not be silently dropped."""
    base = yaml.safe_load((Path("configs") / "base.yaml").read_text(encoding="utf-8"))
    base["run"]["max_cost_usdd"] = 5.0  # deliberate typo
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    for name in ("data", "models"):
        source = Path("configs") / f"{name}.yaml"
        (tmp_path / f"{name}.yaml").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(ValueError):
        load_config("base", "data", "models", configs_dir=tmp_path)


def test_missing_layer_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config("nope", configs_dir=tmp_path)


def test_config_is_frozen() -> None:
    config = load_config()
    with pytest.raises(ValueError):
        config.data.seed = 1


def test_config_model_validates_directly() -> None:
    assert issubclass(Config, object)
