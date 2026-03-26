from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def require_columns(frame: pd.DataFrame, required_columns: Sequence[str], frame_name: str) -> None:
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"{frame_name} is missing required columns: {missing_columns}")


def ensure_unique_ids(frame: pd.DataFrame, frame_name: str) -> None:
    require_columns(frame, ["id"], frame_name)
    if not frame["id"].astype(str).is_unique:
        raise ValueError(f"{frame_name} contains duplicate ids, which would break retrieval output mapping.")


def load_json_frame(path: Path, frame_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{frame_name} file not found: {path}")
    return pd.read_json(path)
