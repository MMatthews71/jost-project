"""Append-only JSON-lines history of delivered picks and settled results."""
from __future__ import annotations

import json

from . import config


def read(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def append(path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def picks() -> list[dict]:
    return read(config.PICKS_LOG)


def results() -> list[dict]:
    return read(config.RESULTS_LOG)


def pending_picks() -> list[dict]:
    settled = {r["pick_id"] for r in results()}
    return [p for p in picks() if p["pick_id"] not in settled]
