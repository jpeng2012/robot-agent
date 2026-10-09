"""JSONL episode traces."""

import json
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _default(o: Any) -> Any:
    if isinstance(o, BaseModel):
        return o.model_dump(mode="json")
    return str(o)


class Tracer:
    def __init__(self, trace_dir: Path | str, name: str):
        self.episode_id = f"{name}_{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.path = Path(trace_dir) / f"{self.episode_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events: list[dict] = []
        self._fh = self.path.open("w")

    def log(self, event: str, **payload: Any) -> None:
        rec = {"seq": len(self.events), "ts": round(time.time(), 3), "event": event, **payload}
        rec = json.loads(json.dumps(rec, default=_default))
        self.events.append(rec)
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
