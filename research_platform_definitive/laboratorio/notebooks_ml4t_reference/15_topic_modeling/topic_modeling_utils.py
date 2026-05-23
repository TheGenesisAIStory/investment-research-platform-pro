"""Modern topic modeling helpers for chapter 15 notebooks."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

STOPWORDS = "english"


def ensure_output_dir(path: str | Path = "../data/topic_modeling") -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def financial_documents() -> pd.DataFrame:
    docs = [
        ("earnings", "Revenue growth accelerated as cloud demand improved and operating margins expanded."),
        ("earnings", "Management discussed pricing power, cost discipline, and a stronger backlog."),
        ("credit", "Credit spreads tightened after refinancing reduced near term default risk."),
        ("credit", "Bond investors focused on leverage, liquidity, covenants, and rating downgrade risk."),
        ("macro", "Inflation cooled while labor markets remained resilient and central banks paused."),
        ("macro", "Yield curves steepened as investors repriced growth and policy expectations."),
        ("energy", "Oil prices rose after supply cuts and stronger refinery demand."),
        ("energy", "Utilities highlighted renewable investment, grid reliability, and fuel cost recovery."),
        ("technology", "Semiconductor orders recovered as AI infrastructure spending increased."),
        ("technology", "Software companies emphasized retention, subscription growth, and cybersecurity demand."),
        ("consumer", "Retail traffic weakened as households traded down and promotional intensity increased."),
        ("consumer", "Luxury sales improved in Europe while China demand remained mixed."),
    ]
    return pd.DataFrame(docs, columns=["label", "text"])


def top_terms(model, vectorizer, n_terms: int = 8) -> pd.DataFrame:
    terms = vectorizer.get_feature_names_out()
    rows = []
    for topic_idx, topic in enumerate(model.components_):
        for rank, idx in enumerate(topic.argsort()[::-1][:n_terms], start=1):
            rows.append({"topic": topic_idx, "rank": rank, "term": terms[idx], "weight": topic[idx]})
    return pd.DataFrame(rows)


def fit_lsi(docs: pd.Series, n_topics: int = 4):
    vectorizer = TfidfVectorizer(stop_words=STOPWORDS, min_df=1)
    X = vectorizer.fit_transform(docs)
    model = TruncatedSVD(n_components=n_topics, random_state=42)
    doc_topics = model.fit_transform(X)
    return model, vectorizer, pd.DataFrame(doc_topics, columns=[f"topic_{i}" for i in range(n_topics)])


def fit_nmf(docs: pd.Series, n_topics: int = 4):
    vectorizer = TfidfVectorizer(stop_words=STOPWORDS, min_df=1)
    X = vectorizer.fit_transform(docs)
    model = NMF(n_components=n_topics, init="nndsvda", random_state=42, max_iter=500)
    doc_topics = model.fit_transform(X)
    return model, vectorizer, pd.DataFrame(doc_topics, columns=[f"topic_{i}" for i in range(n_topics)])


def fit_lda(docs: pd.Series, n_topics: int = 4):
    vectorizer = CountVectorizer(stop_words=STOPWORDS, min_df=1)
    X = vectorizer.fit_transform(docs)
    model = LatentDirichletAllocation(n_components=n_topics, learning_method="batch", random_state=42, max_iter=20)
    doc_topics = model.fit_transform(X)
    return model, vectorizer, pd.DataFrame(doc_topics, columns=[f"topic_{i}" for i in range(n_topics)]), X


def dirichlet_samples(alpha: list[float] | np.ndarray, samples: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    draws = rng.dirichlet(np.asarray(alpha, dtype=float), size=samples)
    return pd.DataFrame(draws, columns=[f"topic_{i}" for i in range(len(alpha))])


def assign_dominant_topic(doc_topics: pd.DataFrame, docs: pd.DataFrame) -> pd.DataFrame:
    result = docs.copy()
    result["dominant_topic"] = doc_topics.idxmax(axis=1)
    result["topic_confidence"] = doc_topics.max(axis=1)
    return result
