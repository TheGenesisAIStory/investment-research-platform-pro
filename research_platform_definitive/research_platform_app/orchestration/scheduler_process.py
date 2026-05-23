"""Launch/stop helpers for the local scheduler process."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .utils import APP_ROOT, STATE_ROOT, read_json, write_json


SCHEDULER_STATE_PATH = STATE_ROOT / "scheduler_state.json"


def _is_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def scheduler_status() -> dict:
    state = read_json(SCHEDULER_STATE_PATH, {})
    pid = state.get("pid")
    alive = _is_alive(pid)
    state["alive"] = alive
    return state


def start_scheduler_process(
    roots: dict[str, Path],
    jobs: list[str] | None = None,
    interval_seconds: int = 900,
    notebook_offhour_start: int = 20,
    notebook_offhour_end: int = 7,
) -> dict:
    current = scheduler_status()
    if current.get("alive"):
        return {"ok": True, "message": "Scheduler already running", **current}
    jobs = jobs or ["screener_refresh"]
    cmd = [
        sys.executable,
        str(APP_ROOT / "scheduler.py"),
        "--interval-seconds",
        str(interval_seconds),
        "--jobs",
        *jobs,
        "--company-root",
        str(roots["company"]),
        "--portfolio-root",
        str(roots["portfolio"]),
        "--workspace-root",
        str(roots["workspace"]),
        "--financial-db-root",
        str(roots.get("financial_db", "")),
        "--notebook-offhour-start",
        str(notebook_offhour_start),
        "--notebook-offhour-end",
        str(notebook_offhour_end),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{APP_ROOT}:{APP_ROOT.parent}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.Popen(
        cmd,
        cwd=str(APP_ROOT.parent),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    state = {
        "pid": proc.pid,
        "jobs": jobs,
        "interval_seconds": interval_seconds,
        "notebook_offhour_start": notebook_offhour_start,
        "notebook_offhour_end": notebook_offhour_end,
        "command": cmd,
        "alive": True,
    }
    write_json(SCHEDULER_STATE_PATH, state)
    return {"ok": True, "message": "Scheduler started", **state}


def stop_scheduler_process() -> dict:
    state = scheduler_status()
    pid = state.get("pid")
    if not state.get("alive") or not pid:
        return {"ok": True, "message": "Scheduler is not running", **state}
    try:
        os.killpg(pid, signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception as exc:
            return {"ok": False, "message": f"Failed to stop scheduler: {exc}", **state}
    for _ in range(10):
        time.sleep(0.2)
        if not _is_alive(pid):
            break
    if _is_alive(pid):
        try:
            os.killpg(pid, signal.SIGKILL)
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
    for _ in range(10):
        time.sleep(0.1)
        if not _is_alive(pid):
            break
    state["alive"] = _is_alive(pid)
    write_json(SCHEDULER_STATE_PATH, state)
    return {"ok": not state["alive"], "message": "Scheduler stopped" if not state["alive"] else "Scheduler stop requested but process still appears alive", **state}
