from __future__ import annotations

import os
import sys
from pathlib import Path

from .runtime import detect_runtime_environment


def candidate_project_roots(project_name: str = "retrieval_project") -> list[Path]:
    runtime_env = detect_runtime_environment()
    candidates = [Path.cwd(), *Path.cwd().parents]
    if runtime_env == "colab":
        drive_root = Path("/content/drive/MyDrive")
        if not drive_root.exists():
            from google.colab import drive  # type: ignore

            drive.mount("/content/drive", force_remount=False)
        candidates = [
            Path("/content"),
            Path("/content/drive/MyDrive"),
            Path("/content/drive/Shareddrives"),
            *candidates,
        ]
    elif runtime_env == "kaggle":
        candidates = [Path("/kaggle/working"), *candidates]

    seen: set[str] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)
    return unique_candidates


def find_project_root(project_name: str = "retrieval_project") -> Path:
    runtime_env = detect_runtime_environment()
    for base in candidate_project_roots(project_name):
        if (base / "src" / "config.py").exists():
            return base
        if (base / project_name / "src" / "config.py").exists():
            return base / project_name
        if runtime_env == "colab" and base.exists():
            for match in base.rglob(project_name):
                if (match / "src" / "config.py").exists():
                    return match
    raise FileNotFoundError("Could not locate project root containing src/config.py")


def setup_notebook(project_name: str = "retrieval_project") -> tuple[str, Path]:
    runtime_env = detect_runtime_environment()
    project_root = find_project_root(project_name=project_name)
    os.chdir(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return runtime_env, project_root
