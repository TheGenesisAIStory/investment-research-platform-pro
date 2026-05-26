"""Stage-level lock files for long-running data jobs.

The lock is intentionally simple and file based: Streamlit launchers, CLI
scripts and tests can all share it without requiring an external scheduler.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .data_platform import utc_now


@dataclass(frozen=True)
class LockInfo:
    run_id: str
    stage: str
    root: str
    pid: int
    started_at: str
    lock_path: str
    stale_replaced: bool = False


class StageLockError(RuntimeError):
    """Raised when a live stage lock prevents another run from starting."""


def _pid_alive(pid: int | str | None) -> bool:
    if pid in {None, ""}:
        return False
    try:
        os.kill(int(str(pid).strip()), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def _safe_stage(stage: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(stage).strip()) or "stage"


def stage_lock_path(stage: str, root: str | Path) -> Path:
    """Return the canonical lock path for a stage under an output root."""
    output_root = Path(root).expanduser()
    return output_root / "locks" / f"{_safe_stage(stage)}.lock"


def read_stage_lock(stage: str, root: str | Path) -> LockInfo | None:
    path = stage_lock_path(stage, root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return LockInfo(
            run_id=str(payload.get("run_id", "")),
            stage=str(payload.get("stage", stage)),
            root=str(payload.get("root", "")),
            pid=int(payload.get("pid", 0)),
            started_at=str(payload.get("started_at", "")),
            lock_path=str(path),
            stale_replaced=bool(payload.get("stale_replaced", False)),
        )
    except Exception:
        return LockInfo("", stage, "", 0, "", str(path))


def is_stage_locked(stage: str, root: str | Path) -> bool:
    info = read_stage_lock(stage, root)
    return bool(info and _pid_alive(info.pid))


def acquire_stage_lock(
    stage: str,
    root: str | Path,
    *,
    run_id: str | None = None,
    protected_root: str | Path | None = None,
    force_stale: bool = True,
) -> LockInfo | None:
    """Acquire a stage lock.

    Returns `None` when a live lock already exists. Stale locks are replaced by
    default because long provider jobs can be interrupted by shells or reboots.
    """
    output_root = Path(root).expanduser()
    path = stage_lock_path(stage, output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_stage_lock(stage, output_root)
    stale_replaced = False
    if existing is not None:
        if _pid_alive(existing.pid):
            return None
        if not force_stale:
            return None
        try:
            path.unlink()
            stale_replaced = True
        except FileNotFoundError:
            stale_replaced = True

    info = LockInfo(
        run_id=run_id or os.environ.get("RESEARCH_PLATFORM_RUN_ID") or f"{_safe_stage(stage)}_{uuid.uuid4().hex[:12]}",
        stage=_safe_stage(stage),
        root=str(Path(protected_root).expanduser() if protected_root is not None else output_root),
        pid=os.getpid(),
        started_at=utc_now(),
        lock_path=str(path),
        stale_replaced=stale_replaced,
    )
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(asdict(info), handle, indent=2, default=str)
    except FileExistsError:
        existing = read_stage_lock(stage, output_root)
        if existing and _pid_alive(existing.pid):
            return None
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        with path.open("x", encoding="utf-8") as handle:
            json.dump(asdict(info), handle, indent=2, default=str)
    return info


def release_stage_lock(stage: str, root: str | Path, *, lock_info: LockInfo | None = None, force: bool = False) -> None:
    """Release a stage lock if owned by this process or if force=True."""
    path = stage_lock_path(stage, root)
    if not path.exists():
        return
    current = read_stage_lock(stage, root)
    if not force and current is not None:
        owner_matches = lock_info is not None and current.run_id == lock_info.run_id and int(current.pid) == int(lock_info.pid)
        process_matches = int(current.pid or 0) == os.getpid()
        if not (owner_matches or process_matches):
            return
    try:
        path.unlink()
    except FileNotFoundError:
        return


def list_stage_locks(root: str | Path) -> list[LockInfo]:
    locks_dir = Path(root).expanduser() / "locks"
    if not locks_dir.exists():
        return []
    out: list[LockInfo] = []
    for path in sorted(locks_dir.glob("*.lock")):
        info = read_stage_lock(path.stem, root)
        if info is not None:
            out.append(info)
    return out


def lock_payload(info: LockInfo | None) -> dict[str, Any]:
    return asdict(info) if info is not None else {}
