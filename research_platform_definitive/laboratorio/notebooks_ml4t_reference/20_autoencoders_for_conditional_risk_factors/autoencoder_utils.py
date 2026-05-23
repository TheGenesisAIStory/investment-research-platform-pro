"""Modern autoencoder fallbacks for chapter 20 notebooks."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def ensure_output_dir(path: str | Path = "../data/autoencoders") -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def make_panel(samples: int = 800, features: int = 20, seed: int = 20):
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(samples, 4))
    loadings = rng.normal(size=(4, features))
    X = factors @ loadings + rng.normal(0, 0.3, size=(samples, features))
    y = factors[:, 0] * 0.01 + rng.normal(0, 0.02, samples)
    cols = [f"characteristic_{i:02d}" for i in range(features)]
    return pd.DataFrame(StandardScaler().fit_transform(X), columns=cols), pd.Series(y, name="forward_return")


def pca_autoencoder(X: pd.DataFrame, components: int = 5):
    pca = PCA(n_components=components, random_state=42)
    encoded = pca.fit_transform(X)
    reconstructed = pca.inverse_transform(encoded)
    rmse = float(np.sqrt(mean_squared_error(X, reconstructed)))
    return pca, pd.DataFrame(encoded, columns=[f"latent_{i}" for i in range(components)]), rmse


def mlp_autoencoder(X: pd.DataFrame, hidden=(12, 4, 12)):
    train, test = train_test_split(X, test_size=0.30, random_state=42)
    model = MLPRegressor(hidden_layer_sizes=hidden, max_iter=400, random_state=42, early_stopping=True)
    model.fit(train, train)
    recon = model.predict(test)
    return model, float(np.sqrt(mean_squared_error(test, recon)))


def denoise_example(X: pd.DataFrame, seed: int = 42):
    rng = np.random.default_rng(seed)
    noisy = X + rng.normal(0, 0.25, X.shape)
    _, encoded, rmse = pca_autoencoder(noisy, components=5)
    return noisy, encoded, rmse


def conditional_factor_model():
    X, y = make_panel(seed=202)
    _, latent, _ = pca_autoencoder(X, components=5)
    train, test, y_train, y_test = train_test_split(latent, y, test_size=0.30, random_state=42)
    model = MLPRegressor(hidden_layer_sizes=(16,), max_iter=400, random_state=42, early_stopping=True)
    model.fit(train, y_train)
    pred = model.predict(test)
    result = test.copy()
    result["target"] = y_test.to_numpy()
    result["prediction"] = pred
    return model, result, float(np.sqrt(mean_squared_error(y_test, pred)))
