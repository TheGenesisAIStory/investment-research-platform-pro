"""Modern unsupervised learning helpers for chapter 13 notebooks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.datasets import make_blobs, make_moons, make_swiss_roll
from sklearn.decomposition import FastICA, PCA
from sklearn.manifold import LocallyLinearEmbedding, TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def ensure_output_dir(path: str | Path = "../data/unsupervised_learning") -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def make_feature_matrix(samples: int = 500, features: int = 12, seed: int = 13) -> tuple[pd.DataFrame, np.ndarray]:
    X, labels = make_blobs(n_samples=samples, centers=4, n_features=features, cluster_std=1.5, random_state=seed)
    cols = [f"feature_{i:02d}" for i in range(features)]
    return pd.DataFrame(StandardScaler().fit_transform(X), columns=cols), labels


def make_asset_returns(n_assets: int = 60, periods: int = 756, seed: int = 1313) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp("2026-05-08"), periods=periods)
    factors = rng.normal(0, [0.009, 0.006, 0.004], size=(periods, 3))
    exposures = rng.normal(0, 1, size=(n_assets, 3))
    noise = rng.normal(0, 0.008, size=(periods, n_assets))
    returns = factors @ exposures.T + noise + 0.0001
    return pd.DataFrame(returns, index=dates, columns=[f"Asset_{i:03d}" for i in range(n_assets)])


def curse_of_dimensionality(max_dim: int = 50, samples: int = 250, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for dim in range(1, max_dim + 1):
        X = rng.normal(size=(samples, dim))
        d = np.sqrt(((X[:50, None, :] - X[None, :50, :]) ** 2).sum(axis=2))
        d = d[np.triu_indices_from(d, k=1)]
        rows.append({"dimensions": dim, "mean_distance": d.mean(), "min_distance": d.min()})
    return pd.DataFrame(rows)


def pca_summary(X: pd.DataFrame, n_components: int = 5) -> tuple[PCA, pd.DataFrame, pd.DataFrame]:
    pca = PCA(n_components=n_components, random_state=42)
    transformed = pca.fit_transform(X)
    explained = pd.DataFrame({"component": range(1, n_components + 1), "explained_variance": pca.explained_variance_ratio_, "cumulative": np.cumsum(pca.explained_variance_ratio_)})
    scores = pd.DataFrame(transformed, columns=[f"PC{i}" for i in range(1, n_components + 1)])
    return pca, explained, scores


def risk_factor_pca(returns: pd.DataFrame, n_components: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaled = StandardScaler().fit_transform(returns)
    pca = PCA(n_components=n_components, random_state=42).fit(scaled)
    factors = pd.DataFrame(pca.transform(scaled), index=returns.index, columns=[f"factor_{i}" for i in range(1, n_components + 1)])
    loadings = pd.DataFrame(pca.components_.T, index=returns.columns, columns=factors.columns)
    return factors, loadings


def eigen_portfolio_weights(returns: pd.DataFrame, component: int = 1) -> pd.Series:
    _, loadings = risk_factor_pca(returns, n_components=max(component, 3))
    w = loadings[f"factor_{component}"].copy()
    return w / w.abs().sum()


def swiss_roll_embedding(samples: int = 600, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    X, color = make_swiss_roll(n_samples=samples, noise=0.04, random_state=seed)
    return pd.DataFrame(X, columns=["x", "y", "z"]), color


def lle_embedding(X: pd.DataFrame | np.ndarray, n_neighbors: int = 12) -> pd.DataFrame:
    model = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2, method="standard", random_state=42)
    emb = model.fit_transform(np.asarray(X))
    return pd.DataFrame(emb, columns=["dim1", "dim2"])


def tsne_embedding(X: pd.DataFrame | np.ndarray, perplexity: int = 30) -> pd.DataFrame:
    model = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42)
    emb = model.fit_transform(np.asarray(X))
    return pd.DataFrame(emb, columns=["dim1", "dim2"])


def clustering_dataset(seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    X, labels = make_moons(n_samples=500, noise=0.06, random_state=seed)
    return pd.DataFrame(StandardScaler().fit_transform(X), columns=["x", "y"]), labels


def compare_clusterers(X: pd.DataFrame) -> pd.DataFrame:
    models = {
        "kmeans": KMeans(n_clusters=2, n_init=10, random_state=42),
        "agglomerative": AgglomerativeClustering(n_clusters=2),
        "dbscan": DBSCAN(eps=0.22, min_samples=6),
        "gmm": GaussianMixture(n_components=2, random_state=42),
    }
    rows = []
    for name, model in models.items():
        labels = model.fit_predict(X) if hasattr(model, "fit_predict") else model.fit(X).predict(X)
        valid = len(set(labels)) > 1 and len(set(labels)) < len(labels)
        rows.append({"model": name, "clusters": len(set(labels)), "silhouette": silhouette_score(X, labels) if valid else np.nan})
    return pd.DataFrame(rows).sort_values("silhouette", ascending=False)


def kmeans_diagnostics(X: pd.DataFrame, max_k: int = 8) -> pd.DataFrame:
    rows = []
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        rows.append({"k": k, "inertia": KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_, "silhouette": silhouette_score(X, labels)})
    return pd.DataFrame(rows)


def hierarchical_labels(X: pd.DataFrame, clusters: int = 4) -> pd.Series:
    labels = AgglomerativeClustering(n_clusters=clusters).fit_predict(X)
    return pd.Series(labels, name="cluster")


def dbscan_labels(X: pd.DataFrame) -> pd.Series:
    return pd.Series(DBSCAN(eps=0.22, min_samples=6).fit_predict(X), name="cluster")


def gmm_labels(X: pd.DataFrame, components: int = 4) -> pd.Series:
    return pd.Series(GaussianMixture(n_components=components, random_state=42).fit_predict(X), name="cluster")


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    corr = returns.corr().clip(-0.999, 0.999)
    dist = np.sqrt((1 - corr) / 2)
    order = leaves_list(linkage(squareform(dist, checks=False), method="single"))
    ordered = returns.columns[order]
    variances = returns[ordered].var()
    inv_var = 1 / variances.replace(0, np.nan)
    weights = inv_var / inv_var.sum()
    return weights.reindex(returns.columns).fillna(0)


def portfolio_backtest(returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    strategy = returns[weights.index].mul(weights, axis=1).sum(axis=1)
    out = pd.DataFrame({"return": strategy})
    out["equity_curve"] = (1 + out["return"]).cumprod()
    return out
