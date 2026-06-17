"""Lightweight social sentiment connectors and scoring utilities.

Provider calls are best-effort and optional. StockTwits and Reddit can work
without project API keys; X requires `X_BEARER_TOKEN`. All outputs are treated
as noisy research context, not trading signals.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .data_platform import resolve_data_platform_roots, utc_now


POSITIVE_TERMS = {
    "beat",
    "beats",
    "bull",
    "bullish",
    "buy",
    "growth",
    "long",
    "outperform",
    "positive",
    "strong",
    "upgrade",
    "upside",
    "winner",
}
NEGATIVE_TERMS = {
    "bear",
    "bearish",
    "cut",
    "downgrade",
    "fraud",
    "loss",
    "miss",
    "negative",
    "risk",
    "sell",
    "short",
    "weak",
    "downside",
}


def _http_json(url: str, headers: dict[str, str] | None = None, timeout: int = 8) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def score_text(text: str) -> float:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_+-]*", str(text).lower())
    if not tokens:
        return 0.0
    pos = sum(1 for token in tokens if token in POSITIVE_TERMS)
    neg = sum(1 for token in tokens if token in NEGATIVE_TERMS)
    return float((pos - neg) / max(pos + neg, 1))


def score_sentiment_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["provider", "ticker", "created_at", "author", "text", "url", "sentiment_score", "sentiment_label"])
    frame["sentiment_score"] = frame["text"].fillna("").map(score_text)
    frame["sentiment_label"] = pd.cut(
        frame["sentiment_score"],
        bins=[-1.01, -0.15, 0.15, 1.01],
        labels=["negative", "neutral", "positive"],
    ).astype(str)
    frame["collected_at"] = utc_now()
    return frame


def fetch_stocktwits_messages(ticker: str, limit: int = 30) -> pd.DataFrame:
    symbol = str(ticker or "").strip().upper().lstrip("$")
    if not symbol:
        return score_sentiment_frame([])
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{urllib.parse.quote(symbol)}.json"
    try:
        payload = _http_json(url, headers={"User-Agent": "TheGenesisAI-ResearchPlatform/1.0"})
    except Exception as exc:
        return pd.DataFrame([{"provider": "stocktwits", "ticker": symbol, "status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}", "collected_at": utc_now()}])
    rows: list[dict[str, Any]] = []
    for message in payload.get("messages", [])[: int(limit)]:
        rows.append(
            {
                "provider": "stocktwits",
                "ticker": symbol,
                "created_at": message.get("created_at", ""),
                "author": (message.get("user") or {}).get("username", ""),
                "text": message.get("body", ""),
                "url": f"https://stocktwits.com/{(message.get('user') or {}).get('username', '')}/message/{message.get('id', '')}",
                "status": "OK",
            }
        )
    return score_sentiment_frame(rows)


def fetch_reddit_mentions(ticker: str, subreddits: Iterable[str] = ("stocks", "investing", "wallstreetbets"), limit: int = 15) -> pd.DataFrame:
    query = str(ticker or "").strip().upper().lstrip("$")
    if not query:
        return score_sentiment_frame([])
    rows: list[dict[str, Any]] = []
    headers = {"User-Agent": os.getenv("REDDIT_USER_AGENT", "TheGenesisAI-ResearchPlatform/1.0")}
    for subreddit in subreddits:
        params = urllib.parse.urlencode({"q": query, "restrict_sr": "on", "sort": "new", "limit": int(limit)})
        url = f"https://www.reddit.com/r/{urllib.parse.quote(str(subreddit))}/search.json?{params}"
        try:
            payload = _http_json(url, headers=headers)
        except Exception as exc:
            rows.append({"provider": "reddit", "ticker": query, "subreddit": subreddit, "status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}", "text": ""})
            continue
        for child in (payload.get("data") or {}).get("children", [])[: int(limit)]:
            data = child.get("data") or {}
            rows.append(
                {
                    "provider": "reddit",
                    "ticker": query,
                    "subreddit": subreddit,
                    "created_at": pd.to_datetime(data.get("created_utc", 0), unit="s", errors="coerce"),
                    "author": data.get("author", ""),
                    "text": f"{data.get('title', '')}\n{data.get('selftext', '')}".strip(),
                    "url": f"https://www.reddit.com{data.get('permalink', '')}",
                    "score_raw": data.get("score", 0),
                    "status": "OK",
                }
            )
    return score_sentiment_frame(rows)


def fetch_x_recent_mentions(ticker: str, limit: int = 20) -> pd.DataFrame:
    token = os.getenv("X_BEARER_TOKEN", "")
    query = str(ticker or "").strip().upper().lstrip("$")
    if not token:
        return pd.DataFrame([{"provider": "x", "ticker": query, "status": "UNCONFIGURED", "error": "Set X_BEARER_TOKEN to enable X recent search.", "collected_at": utc_now()}])
    params = urllib.parse.urlencode(
        {
            "query": f'("${query}" OR {query}) lang:en -is:retweet',
            "max_results": max(10, min(int(limit), 100)),
            "tweet.fields": "created_at,public_metrics,author_id",
        }
    )
    url = f"https://api.twitter.com/2/tweets/search/recent?{params}"
    try:
        payload = _http_json(url, headers={"Authorization": f"Bearer {token}", "User-Agent": "TheGenesisAI-ResearchPlatform/1.0"})
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        return pd.DataFrame([{"provider": "x", "ticker": query, "status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}", "collected_at": utc_now()}])
    rows = [
        {
            "provider": "x",
            "ticker": query,
            "created_at": item.get("created_at", ""),
            "author": item.get("author_id", ""),
            "text": item.get("text", ""),
            "url": "",
            "status": "OK",
        }
        for item in payload.get("data", [])
    ]
    return score_sentiment_frame(rows)


def collect_ticker_sentiment(
    ticker: str,
    output_root: str | Path | None = None,
    providers: Iterable[str] = ("stocktwits", "reddit"),
    limit: int = 25,
) -> dict[str, pd.DataFrame]:
    provider_set = {str(provider).lower() for provider in providers}
    frames: list[pd.DataFrame] = []
    if "stocktwits" in provider_set:
        frames.append(fetch_stocktwits_messages(ticker, limit=limit))
    if "reddit" in provider_set:
        frames.append(fetch_reddit_mentions(ticker, limit=max(5, min(int(limit), 25))))
    if "x" in provider_set:
        frames.append(fetch_x_recent_mentions(ticker, limit=limit))
    mentions = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True, sort=False) if frames else pd.DataFrame()
    ok = mentions[mentions.get("status", pd.Series("", index=mentions.index)).astype(str).eq("OK")] if not mentions.empty and "status" in mentions.columns else mentions
    summary = pd.DataFrame(
        [
            {
                "ticker": str(ticker or "").upper(),
                "mention_count": int(len(ok)),
                "avg_sentiment": float(pd.to_numeric(ok.get("sentiment_score", pd.Series(dtype=float)), errors="coerce").mean()) if not ok.empty else 0.0,
                "positive_count": int(ok.get("sentiment_label", pd.Series(dtype=str)).astype(str).eq("positive").sum()) if not ok.empty else 0,
                "negative_count": int(ok.get("sentiment_label", pd.Series(dtype=str)).astype(str).eq("negative").sum()) if not ok.empty else 0,
                "providers": ",".join(sorted(provider_set)),
                "collected_at": utc_now(),
            }
        ]
    )
    if output_root:
        roots = resolve_data_platform_roots(repo_output_root=output_root)
        table_root = roots.repo_output / "sentiment" / "tables"
        table_root.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(ticker or "").upper())
        mentions.to_csv(table_root / f"Sentiment_mentions_{safe}.csv", index=False)
        summary.to_csv(table_root / f"Sentiment_summary_{safe}.csv", index=False)
    return {"mentions": mentions, "summary": summary}
