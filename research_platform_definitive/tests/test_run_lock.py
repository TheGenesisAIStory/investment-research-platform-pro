from __future__ import annotations

import json
from pathlib import Path

from research_platform_core.run_lock import acquire_stage_lock, is_stage_locked, read_stage_lock, release_stage_lock, stage_lock_path


def test_stage_lock_allows_single_live_owner(tmp_path: Path) -> None:
    first = acquire_stage_lock("equity_prices", tmp_path, run_id="run_a", protected_root=tmp_path / "ohlcv")
    second = acquire_stage_lock("equity_prices", tmp_path, run_id="run_b", protected_root=tmp_path / "ohlcv")

    assert first is not None
    assert second is None
    assert is_stage_locked("equity_prices", tmp_path)
    assert read_stage_lock("equity_prices", tmp_path).run_id == "run_a"

    release_stage_lock("equity_prices", tmp_path, lock_info=first)
    assert read_stage_lock("equity_prices", tmp_path) is None


def test_stage_lock_replaces_stale_pid(tmp_path: Path) -> None:
    path = stage_lock_path("equity_prices", tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "run_id": "old",
                "stage": "equity_prices",
                "root": str(tmp_path / "ohlcv"),
                "pid": 99999999,
                "started_at": "2026-01-01T00:00:00+00:00",
                "lock_path": str(path),
            }
        ),
        encoding="utf-8",
    )

    info = acquire_stage_lock("equity_prices", tmp_path, run_id="new", protected_root=tmp_path / "ohlcv")

    assert info is not None
    assert info.run_id == "new"
    assert info.stale_replaced
    release_stage_lock("equity_prices", tmp_path, lock_info=info)
