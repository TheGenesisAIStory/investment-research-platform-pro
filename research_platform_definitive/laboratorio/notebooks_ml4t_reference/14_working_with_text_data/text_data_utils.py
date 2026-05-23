"""Modern local NLP helpers for chapter 14 notebooks."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "is", "are", "was", "were", "as", "by", "from", "at"}
POSITIVE = {"gain", "gains", "beat", "beats", "strong", "growth", "upgrade", "profit", "record", "surge", "positive", "bullish", "improve", "improved"}
NEGATIVE = {"loss", "losses", "miss", "weak", "downgrade", "risk", "fall", "falls", "lawsuit", "negative", "bearish", "decline", "default"}


def ensure_output_dir(path: str | Path = "../data/text_data") -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def sample_financial_corpus() -> pd.DataFrame:
    rows = [
        ("markets", "Global equities gained as inflation cooled and central banks signaled patience."),
        ("markets", "Technology shares fell after weak guidance and rising bond yields pressured valuations."),
        ("earnings", "The company beat revenue expectations and reported record operating profit."),
        ("earnings", "Management missed earnings estimates and warned about slower demand."),
        ("macro", "Payroll growth improved while unemployment remained stable across major sectors."),
        ("macro", "Manufacturing surveys declined and recession risk increased among exporters."),
        ("credit", "Credit spreads tightened as default fears eased and liquidity improved."),
        ("credit", "The issuer faced downgrade risk after losses and covenant pressure."),
        ("markets", "Energy stocks surged with oil prices while defensive sectors lagged."),
        ("earnings", "Analysts upgraded the stock after stronger margins and positive cash flow."),
        ("macro", "Consumer confidence weakened despite resilient labor income."),
        ("credit", "Bondholders welcomed refinancing news and a stronger balance sheet."),
    ]
    return pd.DataFrame(rows, columns=["label", "text"])


def sample_social_posts() -> pd.DataFrame:
    rows = [
        ("bullish", "Strong breakout today, volume confirms the rally and guidance looks positive"),
        ("bearish", "Weak chart, downgrade risk and demand falling faster than expected"),
        ("bullish", "Record cash flow and margin improvement, buyers are back"),
        ("bearish", "Lawsuit headline plus missed earnings, I expect more losses"),
        ("bullish", "Inflation cooling should support growth stocks"),
        ("bearish", "Credit stress rising and liquidity looks poor"),
    ]
    return pd.DataFrame(rows, columns=["sentiment", "text"])


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z][A-Za-z'-]+", text.lower()) if t not in STOPWORDS]


def nlp_pipeline(text: str) -> pd.DataFrame:
    tokens = tokenize(text)
    counts = Counter(tokens)
    return pd.DataFrame({"token": list(counts.keys()), "count": list(counts.values())}).sort_values("count", ascending=False)


def lexical_sentiment(text: str) -> dict[str, float | str]:
    tokens = tokenize(text)
    pos = sum(t in POSITIVE for t in tokens)
    neg = sum(t in NEGATIVE for t in tokens)
    score = (pos - neg) / max(len(tokens), 1)
    label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
    return {"positive_hits": pos, "negative_hits": neg, "score": score, "label": label}


def document_term_matrix(corpus: pd.Series, tfidf: bool = False):
    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, min_df=1) if tfidf else CountVectorizer(tokenizer=tokenize, token_pattern=None, min_df=1)
    matrix = vectorizer.fit_transform(corpus)
    return pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names_out())


def train_news_classifier(data: pd.DataFrame) -> tuple[object, pd.DataFrame, str]:
    train, test = train_test_split(data, test_size=0.35, random_state=42, stratify=data["label"])
    model = make_pipeline(TfidfVectorizer(tokenizer=tokenize, token_pattern=None), MultinomialNB())
    model.fit(train["text"], train["label"])
    pred = model.predict(test["text"])
    results = test.assign(prediction=pred)
    return model, results, classification_report(test["label"], pred, zero_division=0)


def train_sentiment_classifier(data: pd.DataFrame) -> tuple[object, pd.DataFrame, float]:
    train, test = train_test_split(data, test_size=0.34, random_state=42, stratify=data["sentiment"])
    model = make_pipeline(TfidfVectorizer(tokenizer=tokenize, token_pattern=None), LogisticRegression(max_iter=500))
    model.fit(train["text"], train["sentiment"])
    pred = model.predict(test["text"])
    return model, test.assign(prediction=pred), accuracy_score(test["sentiment"], pred)
