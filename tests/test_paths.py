"""The path layer is load-bearing: everything else resolves through it."""

from __future__ import annotations

from pathlib import Path

from thesis import paths


def test_repo_root_contains_pyproject() -> None:
    assert (paths.REPO_ROOT / "pyproject.toml").is_file()


def test_find_repo_root_walks_upwards() -> None:
    deep = paths.REPO_ROOT / "src" / "thesis" / "data"
    assert paths.find_repo_root(deep) == paths.REPO_ROOT


def test_find_repo_root_raises_outside_repo(tmp_path: Path) -> None:
    try:
        paths.find_repo_root(tmp_path)
    except RuntimeError:
        return
    msg = "expected RuntimeError outside a repository"
    raise AssertionError(msg)


def test_derived_paths_sit_under_repo_root() -> None:
    for directory in (paths.DATA_DIR, paths.RUNS_DIR, paths.OUTPUTS_DIR, paths.CONFIGS_DIR):
        assert paths.REPO_ROOT in directory.parents or directory.parent == paths.REPO_ROOT


def test_ensure_dirs_is_idempotent() -> None:
    paths.ensure_dirs()
    paths.ensure_dirs()
    assert paths.CACHE_DIR.is_dir()
