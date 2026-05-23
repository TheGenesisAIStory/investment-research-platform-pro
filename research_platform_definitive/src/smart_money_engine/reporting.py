"""HTML/CSV reporting for Smart Money Government Data Engine."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io import ensure_dir


def table_html(df: pd.DataFrame, title: str, max_rows: int = 25) -> str:
    if df.empty:
        return f"<section><h2>{title}</h2><p class='empty'>No data available. Check source coverage and ingestion manifest.</p></section>"
    return f"<section><h2>{title}</h2>{df.head(max_rows).to_html(index=False, classes='table')}</section>"


def build_html_report(outputs: dict[str, pd.DataFrame], path: Path) -> Path:
    ensure_dir(path.parent)
    css = """
    <style>
    body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:28px;color:#172033;background:#f7f9fc}
    h1{margin-bottom:4px}.subtitle{color:#667085}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}
    .card{background:white;border:1px solid #d9e2ec;border-radius:10px;padding:14px}.card b{font-size:24px}
    section{background:white;border:1px solid #d9e2ec;border-radius:10px;padding:16px;margin:16px 0;overflow:auto}
    .table{border-collapse:collapse;width:100%;font-size:13px}.table th,.table td{border-bottom:1px solid #edf2f7;padding:7px;text-align:left}
    .empty{color:#b54708}.note{background:#fff7ed;border-left:5px solid #da7101;padding:12px;border-radius:8px}
    </style>
    """
    scores = outputs.get("smart_money_scores", pd.DataFrame())
    events = outputs.get("event_feed", pd.DataFrame())
    coverage = outputs.get("coverage", pd.DataFrame())
    cards = {
        "Ranked issuers": len(scores),
        "Events": len(events),
        "Datasets": len(coverage),
        "Sources OK": int(coverage.get("status", pd.Series(dtype=str)).astype(str).eq("OK").sum()) if not coverage.empty else 0,
    }
    html = [
        "<html><head><meta charset='utf-8'>",
        css,
        "</head><body>",
        "<h1>Smart Money Government Data Engine</h1>",
        "<p class='subtitle'>Official-source-first ownership, insider, activism, macro flow and public spending intelligence.</p>",
        "<div class='grid'>",
        *[f"<div class='card'><span>{k}</span><br><b>{v}</b></div>" for k, v in cards.items()],
        "</div>",
        "<div class='note'><b>Caveat:</b> 13F is delayed, Form 4 intent needs context, TIC/COT are macro proxies, and EU coverage is fragmented by regulator/country.</div>",
        table_html(scores, "Smart Money Leaderboard"),
        table_html(events, "Event Feed"),
        table_html(outputs.get("sector_monitor", pd.DataFrame()), "Sector / Theme Monitor"),
        table_html(coverage, "Ingestion Coverage"),
        table_html(outputs.get("government_dataset_matrix", pd.DataFrame()), "Government Dataset Matrix", 50),
        "</body></html>",
    ]
    path.write_text("\n".join(html), encoding="utf-8")
    return path
