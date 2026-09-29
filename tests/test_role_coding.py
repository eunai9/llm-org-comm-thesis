from __future__ import annotations

from pathlib import Path

import pandas as pd

from thesis.analysis.role_coding import draw_stratified_sample, render_coding_page


def _fake_grid(direction: str, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": [f"{direction}_{i}" for i in range(n)],
            "direction": [direction] * n,
            "replicate": [1] * n,
            "body": [f"reply {i}" for i in range(n)],
        }
    )


def test_draw_stratified_sample_covers_every_direction(tmp_path: Path) -> None:
    frame = pd.concat([_fake_grid(d, 30) for d in ("up", "lateral", "down")], ignore_index=True)
    path = tmp_path / "grid.parquet"
    frame.to_parquet(path)
    sample = draw_stratified_sample({"model_a": path}, n=9, seed=1)
    assert len(sample) == 9
    assert set(sample["true_direction"]) == {"up", "lateral", "down"}


def test_draw_stratified_sample_is_reproducible(tmp_path: Path) -> None:
    frame = pd.concat([_fake_grid(d, 30) for d in ("up", "lateral", "down")], ignore_index=True)
    path = tmp_path / "grid.parquet"
    frame.to_parquet(path)
    first = draw_stratified_sample({"model_a": path}, n=9, seed=1)
    second = draw_stratified_sample({"model_a": path}, n=9, seed=1)
    assert first["item"].tolist() == second["item"].tolist()


def test_render_coding_page_hides_the_true_direction() -> None:
    sample = pd.DataFrame(
        {"item": [1], "model": ["model_a"], "true_direction": ["down"], "text": ["Send it now."]}
    )
    page = render_coding_page(sample)
    assert "Send it now." in page
    assert "model_a" not in page
