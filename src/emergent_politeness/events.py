"""Event log: every act is recorded so that all metrics can be computed afterwards (spec §4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


class EventLog:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def log(self, step: int, kind: str, /, **fields: Any) -> None:
        assert "step" not in fields and "kind" not in fields, "reserved field name"
        self.records.append({"step": step, "kind": kind, **fields})

    def of_kind(self, kind: str) -> Iterator[dict[str, Any]]:
        return (r for r in self.records if r["kind"] == kind)

    def count(self, kind: str) -> int:
        return sum(1 for _ in self.of_kind(kind))

    def to_jsonl(self, path: str | Path) -> None:
        with open(path, "w") as f:
            for r in self.records:
                f.write(json.dumps(r) + "\n")
