"""Modern sequence-model fallbacks for chapter 19 notebooks."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def ensure_output_dir(path: str | Path = "../data/recurrent_neural_nets") -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def make_sequence_dataset(samples: int = 1000, lookback: int = 20, features: int = 3, seed: int = 19):
    rng = np.random.default_rng(seed)
    base = np.sin(np.linspace(0, 40, samples)) + rng.normal(0, 0.12, samples)
    extra = np.column_stack([base, np.roll(base, 3), rng.normal(0, 1, samples)])[:, :features]
    X, y_reg, y_cls = [], [], []
    for i in range(lookback, samples - 1):
        window = extra[i - lookback:i].reshape(-1)
        target = base[i + 1]
        X.append(window)
        y_reg.append(target)
        y_cls.append(int(target > base[i]))
    return pd.DataFrame(X), pd.Series(y_reg, name="target"), pd.Series(y_cls, name="direction")


def train_sequence_regression(features: int = 1) -> tuple[MLPRegressor, float]:
    X, y, _ = make_sequence_dataset(features=features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=False)
    model = MLPRegressor(hidden_layer_sizes=(48,), max_iter=300, random_state=42, early_stopping=True)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, float(np.sqrt(mean_squared_error(y_test, pred)))


def train_sequence_classifier(features: int = 3) -> tuple[MLPClassifier, float]:
    X, _, y = make_sequence_dataset(features=features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=False)
    model = MLPClassifier(hidden_layer_sizes=(48,), max_iter=300, random_state=42, early_stopping=True)
    model.fit(X_train, y_train)
    return model, accuracy_score(y_test, model.predict(X_test))


def text_corpus() -> pd.DataFrame:
    return pd.DataFrame({"label": ["positive", "negative", "positive", "negative", "positive", "negative"], "text": ["strong growth and profit beat expectations", "weak losses and downgrade risk", "cash flow improved with positive guidance", "default risk and falling demand", "revenue recovered and margins expanded", "litigation pressure and liquidity stress"]})


def train_text_classifier() -> tuple[object, pd.DataFrame, float]:
    data = text_corpus()
    train, test = train_test_split(data, test_size=0.34, random_state=42, stratify=data.label)
    model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=500))
    model.fit(train.text, train.label)
    pred = model.predict(test.text)
    return model, test.assign(prediction=pred), accuracy_score(test.label, pred)
