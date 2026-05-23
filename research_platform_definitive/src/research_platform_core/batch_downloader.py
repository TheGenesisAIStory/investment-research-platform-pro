"""Parallel batch downloader with conservative rate limiting."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable


class BatchDownloader:
    """Download/fetch a symbol list with bounded workers and RPM throttle."""

    def __init__(self, max_workers: int = 5, requests_per_minute: int = 100):
        self.max_workers = max(int(max_workers), 1)
        self.requests_per_minute = max(int(requests_per_minute), 1)
        self.request_times: list[float] = []

    def _wait_if_needed(self) -> None:
        now = time.time()
        self.request_times = [t for t in self.request_times if now - t < 60]
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.request_times[0])
            time.sleep(max(0.0, sleep_time))
        self.request_times.append(time.time())

    def batch_download(
        self,
        symbols: Iterable[str],
        fetch_func: Callable[[str], Any],
        progress_callback: Callable[[int, int, str, str], None] | None = None,
    ) -> dict[str, Any]:
        symbols = list(dict.fromkeys(str(s).strip() for s in symbols if str(s).strip()))
        results: dict[str, Any] = {}
        errors: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for symbol in symbols:
                self._wait_if_needed()
                futures[executor.submit(fetch_func, symbol)] = symbol
            total = len(futures)
            for current, future in enumerate(as_completed(futures), start=1):
                symbol = futures[future]
                try:
                    results[symbol] = future.result()
                    status = "complete"
                except Exception as exc:
                    errors[symbol] = str(exc)
                    status = "failed"
                if progress_callback:
                    progress_callback(current, total, symbol, status)
        return {"results": results, "errors": errors}
