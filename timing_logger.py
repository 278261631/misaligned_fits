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
        # Map monotonic perf_counter timestamps to UTC wall clock for precise spans.
        self._perf0 = time.perf_counter()
        self._wall0_epoch = time.time()

    def _perf_to_utc_iso(self, perf_value: float) -> str:
        epoch = self._wall0_epoch + (float(perf_value) - self._perf0)
        return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def write_event(
        self,
        step: str,
        duration_ms: float,
        status: str = "ok",
        meta: dict[str, Any] | None = None,
        start_perf: float | None = None,
        end_perf: float | None = None,
    ) -> None:
        duration_ms = float(duration_ms)
        if start_perf is not None and end_perf is None:
            end_perf = float(start_perf) + (duration_ms / 1000.0)
        elif end_perf is not None and start_perf is None:
            start_perf = float(end_perf) - (duration_ms / 1000.0)

        event_ts = self._perf_to_utc_iso(end_perf) if end_perf is not None else _utc_now_iso()
        payload = {
            "ts": event_ts,
            "run_id": self.run_id,
            "script": self.script,
            "step": str(step),
            "duration_ms": duration_ms,
            "status": str(status),
        }
        if start_perf is not None:
            payload["start_ts"] = self._perf_to_utc_iso(start_perf)
        if end_perf is not None:
            payload["end_ts"] = self._perf_to_utc_iso(end_perf)
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
            t1 = time.perf_counter()
            dt_ms = (t1 - t0) * 1000.0
            err_meta: dict[str, Any] = {} if meta is None else dict(meta)
            err_meta["error"] = str(exc)
            self.write_event(step=step, duration_ms=dt_ms, status="error", meta=err_meta, start_perf=t0, end_perf=t1)
            raise
        else:
            t1 = time.perf_counter()
            dt_ms = (t1 - t0) * 1000.0
            self.write_event(step=step, duration_ms=dt_ms, status="ok", meta=meta, start_perf=t0, end_perf=t1)


def run_script_with_timing(main_fn, script_name: str, timing_path: Path | None = None, default_dir: Path | None = None):
    logger = TimingLogger(
        script=script_name,
        timing_path=timing_path,
        default_dir=default_dir,
    )
    with logger.step("script_total"):
        main_fn()

