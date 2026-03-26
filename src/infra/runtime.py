from __future__ import annotations

import os
from pathlib import Path

from ..types import RuntimePaths


def detect_runtime_environment() -> str:
    try:
        import google.colab  # type: ignore  # noqa: F401

        return "colab"
    except Exception:
        if Path("/kaggle/input").exists():
            return "kaggle"
        return "local"


def find_kaggle_data_dir() -> Path | None:
    for dirname, _, filenames in os.walk("/kaggle/input"):
        if "docs.json" in filenames:
            return Path(dirname)
    return None


def find_colab_project_dir(project_name: str = "retrieval_project") -> Path | None:
    drive_candidates = [
        Path("/content/drive/MyDrive"),
        Path("/content/drive/Shareddrives"),
    ]

    for drive_root in drive_candidates:
        if not drive_root.exists():
            continue
        direct_candidate = drive_root / project_name
        if (direct_candidate / "data" / "docs.json").exists():
            return direct_candidate
        for candidate in drive_root.rglob(project_name):
            if candidate.is_dir() and (candidate / "data" / "docs.json").exists():
                return candidate
    return None


def find_local_project_data_dir() -> Path | None:
    roots = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
    for root in roots:
        data_dir = root / "data"
        if (data_dir / "docs.json").exists():
            return data_dir
    return None


def resolve_runtime_paths(output_filename: str = "solutions_SeaFour.csv") -> RuntimePaths:
    runtime_env = detect_runtime_environment()
    project_dir: Path | None = None
    work_dir = Path.cwd()

    if runtime_env == "colab":
        project_dir = find_colab_project_dir("retrieval_project")
        if project_dir is None:
            raise FileNotFoundError(
                "Google Drive is mounted, but `retrieval_project/data/docs.json` was not found."
            )
        os.chdir(project_dir)
        work_dir = project_dir
        data_dir = project_dir / "data"
    elif runtime_env == "kaggle":
        data_dir = find_kaggle_data_dir()
        if data_dir is None:
            raise FileNotFoundError(
                "Kaggle environment detected, but `docs.json` was not found under /kaggle/input."
            )
        project_dir = Path.cwd()
    else:
        data_dir = find_local_project_data_dir()
        if data_dir is None:
            raise FileNotFoundError("Could not find `data/docs.json` near the current working directory.")
        project_dir = data_dir.parent
        work_dir = project_dir
        os.chdir(work_dir)

    cache_dir = work_dir / "cache"
    output_path = work_dir / output_filename
    return RuntimePaths(
        runtime_env=runtime_env,
        project_dir=project_dir or work_dir,
        work_dir=work_dir,
        data_dir=data_dir,
        cache_dir=cache_dir,
        output_path=output_path,
    )
