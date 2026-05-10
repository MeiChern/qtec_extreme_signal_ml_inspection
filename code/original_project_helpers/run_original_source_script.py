# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/_run_source.py
# Renamed package path: code/original_project_helpers/run_original_source_script.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

"""Shared helper for running bundled submission figure scripts."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SUBMISSION_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SUBMISSION_ROOT.parent
PROJECT_OUTPUT = PROJECT_ROOT / "outputs" / "deformation_rate_gradient_lake_paper"
FIG_DIR = SUBMISSION_ROOT / "results" / "figures"
TABLE_DIR = SUBMISSION_ROOT / "results" / "tables"
PREFERRED_PYTHON = Path("/home/cassian/miniforge/envs/insar_ml/bin/python")


def resolve_python() -> str:
    if PREFERRED_PYTHON.exists():
        return str(PREFERRED_PYTHON)
    return str(Path(sys.executable).resolve())


def run_source(source_name: str, extra_args: list[str] | None = None) -> None:
    """Run a bundled source script against the project root and shared output cache."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    source = SCRIPT_DIR / source_name
    if not source.exists():
        raise FileNotFoundError(f"Bundled source script not found: {source}")

    cmd = [
        resolve_python(),
        str(source),
        "--base-dir",
        str(PROJECT_ROOT),
        "--out-dir",
        str(PROJECT_OUTPUT),
    ]
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
    env.setdefault("XDG_CACHE_HOME", "/tmp")
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = (
        f"{SCRIPT_DIR}:{existing_pythonpath}"
        if existing_pythonpath
        else str(SCRIPT_DIR)
    )

    print(f"[run_source] Running: {source.name}")
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), env=env)
    print(f"[run_source] Completed: {source.name}")


def copy_figure(source_stem: str, submission_name: str, source_subdir: str | None = None) -> bool:
    """Copy a rebuilt figure into the submission results/figures directory."""
    src_dir = PROJECT_OUTPUT / "figures"
    if source_subdir:
        src_dir = src_dir / source_subdir
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    copied = False
    for ext in ("png", "pdf"):
        src = src_dir / f"{source_stem}.{ext}"
        dst = FIG_DIR / f"{submission_name}.{ext}"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"[run_source] Copied figure: {dst}")
            copied = True
        else:
            print(f"[run_source] Warning: figure not found: {src}")
    return copied


def copy_table(source_stem: str, submission_name: str, source_subdir: str | None = None) -> bool:
    """Copy a rebuilt table into the submission results/tables directory."""
    src_dir = PROJECT_OUTPUT / "tables"
    if source_subdir:
        src_dir = src_dir / source_subdir
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    src = src_dir / f"{source_stem}.csv"
    dst = TABLE_DIR / f"{submission_name}.csv"
    if src.exists():
        shutil.copy2(src, dst)
        print(f"[run_source] Copied table: {dst}")
        return True
    print(f"[run_source] Warning: table not found: {src}")
    return False
