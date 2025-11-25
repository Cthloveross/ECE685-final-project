from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2))


def save_table(path: Path, records: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    df = pd.DataFrame(records)
    df.to_parquet(path)


def load_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
