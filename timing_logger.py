from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _default_run_id() -> str:
    env_run_id = os.environ.get("MISALIGNED_FITS_RUN_ID", "").strip()
    if env_run_id:
        return env_run_id
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{os.getpid()}_{socket.gethostname()}"


def resolve_timing_path(timing_path: Path | None = None, default_dir: Path | None = None) -> Path:
    env_path = os.environ.get("MISALIGNED_FITS_TIMING_PATH", "").strip()
    if timing_path is not None:
        p = Path(timing_path)
    elif env_path:
        p = Path(env_path)
    else:
        root_dir = Path(__file__).resolve().parent if default_dir is None else Path(default_dir)
        p = root_dir / "output" / "timing.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


class TimingLogger:
    def __init__(
        self,
        script: str,
        timing_path: Path | None = None,
        run_id: str | None = None,
        default_dir: Path | None = None,
    ):
        self.script = str(script)
        self.run_id = str(run_id).strip() if run_id is not None and str(run_id).strip() else _default_run_id()
        self.timing_path = resolve_timing_path(timing_path=timing_path, default_dir=default_dir)

    def write_event(self, step: str, duration_ms: float, status: str = "ok", meta: dict[str, Any] | None = None) -> None:
        payload = {
            "ts": _utc_now_iso(),
            "run_id": self.run_id,
            "script": self.script,
            "step": str(step),
            "duration_ms": float(duration_ms),
            "status": str(status),
        }
        if meta:
            payload["meta"] = meta
        line = json.dumps(payload, ensure_ascii=False)
        with self.timing_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    @contextmanager
    def step(self, step: str, meta: dict[str, Any] | None = None):
        t0 = time.perf_counter()
        try:
            yield
        except Exception as exc:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            err_meta: dict[str, Any] = {} if meta is None else dict(meta)
            err_meta["error"] = str(exc)
            self.write_event(step=step, duration_ms=dt_ms, status="error", meta=err_meta)
            raise
        else:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.write_event(step=step, duration_ms=dt_ms, status="ok", meta=meta)


def run_script_with_timing(main_fn, script_name: str, timing_path: Path | None = None, default_dir: Path | None = None):
    logger = TimingLogger(
        script=script_name,
        timing_path=timing_path,
        default_dir=default_dir,
    )
    with logger.step("script_total"):
        main_fn()

