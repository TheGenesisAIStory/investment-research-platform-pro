"""Modern local embedding helpers for chapter 16 notebooks."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def ensure_output_dir(path: str | Path = "../data/word_embeddings") -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def corpus() -> pd.DataFrame:
    rows = [
        ("positive", "profit growth improved as cloud revenue and margins beat expectations"),
        ("negative", "losses increased after weak demand and downgrade risk hurt liquidity"),
        ("positive", "cash flow strengthened and management raised guidance for revenue"),
        ("negative", "default concerns rose as debt costs and covenant pressure increased"),
        ("positive", "semiconductor orders recovered with artificial intelligence infrastructure demand"),
        ("negative", "retail sales declined and inventory markdowns pressured gross margin"),
        ("positive", "credit spreads tightened after refinancing improved balance sheet flexibility"),
        ("negative", "litigation risk and regulatory investigation weighed on investor sentiment"),
    ]
    return pd.DataFrame(rows, columns=["label", "text"])


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z]+", text.lower())


def sentence_tokens(docs: pd.Series) -> list[list[str]]:
    return [tokenize(doc) for doc in docs]


def cooccurrence_embeddings(docs: pd.Series, dimensions: int = 5, window: int = 2) -> pd.DataFrame:
    tokens = sentence_tokens(docs)
    vocab = sorted({token for sent in tokens for token in sent})
    idx = {token: i for i, token in enumerate(vocab)}
    mat = np.zeros((len(vocab), len(vocab)))
    for sent in tokens:
        for i, token in enumerate(sent):
            left, right = max(0, i - window), min(len(sent), i + window + 1)
            for ctx in sent[left:i] + sent[i + 1:right]:
                mat[idx[token], idx[ctx]] += 1
    svd = TruncatedSVD(n_components=min(dimensions, len(vocab) - 1), random_state=42)
    emb = svd.fit_transform(mat + 1e-6)
    return pd.DataFrame(emb, index=vocab, columns=[f"dim_{i}" for i in range(emb.shape[1])])


def nearest_words(embeddings: pd.DataFrame, word: str, top_n: int = 5) -> pd.DataFrame:
    if word not in embeddings.index:
        raise KeyError(f"{word} not in embeddings")
    sims = cosine_similarity(embeddings.loc[[word]], embeddings)[0]
    out = pd.DataFrame({"word": embeddings.index, "similarity": sims}).sort_values("similarity", ascending=False)
    return out[out.word != word].head(top_n)


def evaluate_analogies(embeddings: pd.DataFrame) -> pd.DataFrame:
    pairs = [("profit", "growth"), ("debt", "risk"), ("cloud", "revenue"), ("default", "debt")]
    rows = []
    for left, right in pairs:
        if left in embeddings.index and right in embeddings.index:
            sim = cosine_similarity(embeddings.loc[[left]], embeddings.loc[[right]])[0, 0]
            rows.append({"left": left, "right": right, "cosine_similarity": sim})
    return pd.DataFrame(rows).sort_values("cosine_similarity", ascending=False)


def doc_vectors(docs: pd.Series, dimensions: int = 5) -> tuple[pd.DataFrame, object]:
    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
    X = vectorizer.fit_transform(docs)
    svd = TruncatedSVD(n_components=min(dimensions, X.shape[1] - 1), random_state=42)
    return pd.DataFrame(svd.fit_transform(X), columns=[f"doc_dim_{i}" for i in range(min(dimensions, X.shape[1] - 1))]), make_pipeline(vectorizer, svd)


def sentiment_model(data: pd.DataFrame) -> tuple[object, pd.DataFrame, float]:
    train, test = train_test_split(data, test_size=0.35, random_state=42, stratify=data["label"])
    model = make_pipeline(TfidfVectorizer(tokenizer=tokenize, token_pattern=None), LogisticRegression(max_iter=500))
    model.fit(train.text, train.label)
    pred = model.predict(test.text)
    return model, test.assign(prediction=pred), accuracy_score(test.label, pred)
