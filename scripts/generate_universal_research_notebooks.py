"""Generate AGENTS.md-compliant definitive research notebooks.

The notebooks created here are executable scaffolds for chapter folders. They
prefer the curated local Google Drive database when present, then fall back to
repo-local files, and only then synthesize realistic market-like data.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_ROOT = Path(
    "/Users/itsgennymac/Library/CloudStorage/"
    "GoogleDrive-sfn.gns@gmail.com/Il mio Drive/Database Finanziario"
)

SECTION_HEADERS = [
    "## 0. Setup & Config",
    "## 1. Data Ingestion",
    "## 2. Cleaning & Alignment",
    "## 3. Feature Engineering",
    "## 4. Targets & Labels",
    "## 5. Descriptive Stats",
    "## 6. Exploratory / Event Study",
    "## 7. Single-Factor Diagnostics",
    "## 8. Statistical Models (regressions / econometrics)",
    "## 9. ML Walk-Forward",
    "## 10. Feature Ablation",
    "## 11. Backtest / Strategy Evaluation",
    "## 12. Interpretability",
    "## 13. Robustness Checks",
    "## 14. Final Summary",
]

SECTION_EMOJI = {
    0: "⚙️",
    1: "📥",
    2: "🧹",
    3: "🔧",
    4: "🎯",
    5: "📊",
    6: "📈",
    7: "🔬",
    8: "📐",
    9: "🤖",
    10: "🧪",
    11: "💼",
    12: "🧠",
    13: "🔁",
    14: "✅",
}

SECTION_DESCRIPTIONS = {
    0: "Configure dependencies, reproducibility, outputs, logging, plotting, and every tunable experiment parameter.",
    1: "Load real data first from the curated database or repo cache, record provenance, and create a realistic synthetic fallback only if needed.",
    2: "Standardize timestamps, align entities, remove invalid observations, and preserve a clear audit trail.",
    3: "Build lagged, leakage-safe features in named blocks that match EXPERIMENT['feature_blocks'].",
    4: "Create forward-looking labels from future returns while keeping every predictor known at t-1.",
    5: "Summarize sample size, distributions, missingness, and entity coverage before modeling.",
    6: "Explore time variation and event-style behavior in the target and feature groups.",
    7: "Measure one-feature-at-a-time relationships against the target using rank correlations and quantile spreads.",
    8: "Estimate transparent econometric baselines before more flexible machine-learning models.",
    9: "Run chronological walk-forward models with embargo and train-only scaling.",
    10: "Compare feature blocks against a controls-only baseline and rank incremental value.",
    11: "Translate predictions into a simple long-short strategy with transaction costs.",
    12: "Explain model behavior with coefficient, permutation, or proxy importance diagnostics.",
    13: "Stress-test conclusions across subperiods, placebo labels, and parameter sensitivity.",
    14: "List generated artifacts and close the experiment with a compact reproducibility summary.",
}


def slug_to_title(slug: str) -> str:
    slug = re.sub(r"^\d+_", "", slug)
    return slug.replace("_", " ").replace("-", " ").title()


def notebook_name(folder: Path) -> str:
    slug = folder.name
    normalized = re.sub(r"^\d+_", "", slug)
    return f"00_{normalized}_definitive.ipynb"


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def section_markdown(section: int) -> str:
    header = SECTION_HEADERS[section]
    emoji = SECTION_EMOJI[section]
    description = SECTION_DESCRIPTIONS[section]
    formulas = {
        3: r"$$x^{(w)}_{i,t}=\\operatorname{clip}(x_{i,t-1}, q_{0.01}, q_{0.99})$$",
        4: r"$$y_{i,t,h}=\\frac{P_{i,t+h}}{P_{i,t}}-1$$",
        7: r"$$IC_t=\\rho_{Spearman}(x_{i,t}, y_{i,t+1})$$",
        8: r"$$y_{i,t+h}=\\alpha+\\beta'X_{i,t-1}+\\epsilon_{i,t+h}$$",
        9: r"$$\\mathcal{D}_{train,t}=\\{s: s < t-\\text{embargo}\\},\\quad \\mathcal{D}_{test,t}=\\{s:t\\le s<t+\\Delta\\}$$",
        11: r"$$r^{LS}_t=w_t' r_{t+1}-c\\,|\\Delta w_t|$$",
        13: r"$$\\Delta metric = metric_{real}-metric_{placebo}$$",
    }
    formula = formulas.get(section, r"$$\\text{All estimates use information available no later than } t-1.$$")
    intuition = (
        "The economic intuition is to separate signal from timing luck by preserving chronology, "
        "penalizing turnover, and comparing every result with simple baselines."
    )
    return f"""{header}

{emoji} {description}

| Cell | What it does | Output |
|---|---|---|
| {section}.1 | Executes the section workflow | Saved table and/or figure |

{formula}

{intuition}
"""


def setup_code(title: str, folder_name: str, database_root: Path) -> str:
    return f'''# Silent dependency installation.
import subprocess
import sys

DEPENDENCIES = {{
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "scikit-learn": "sklearn",
    "statsmodels": "statsmodels",
    "pyarrow": "pyarrow",
}}
for package, import_name in DEPENDENCIES.items():
    try:
        __import__(import_name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

import logging
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

EXPERIMENT = {{
    "project_name": "{title}",
    "database_root": r"{database_root}",
    "repo_data_root": "../data",
    "start_date": "2018-01-01",
    "end_date": "2026-05-13",
    "target": "forward_return_21d",
    "horizons": [5, 21, 63],
    "test_start": "2023-01-01",
    "embargo": 21,
    "n_quantiles": 5,
    "cost_bps": 10.0,
    "models": ["linear", "elastic_net", "random_forest"],
    "run_ablation": True,
    "run_backtest": True,
    "save_figures": True,
    "feature_blocks": ["controls", "momentum", "risk", "quality"],
    "min_train_rows": 126,
    "min_test_rows": 21,
    "rolling_windows": [21, 63],
    "synthetic_periods": 756,
    "random_seed": 42,
    "figure_dpi": 180,
    "single_figsize": (10, 5),
    "two_panel_figsize": (14, 5),
    "grid_figsize": (14, 10),
}}

UNIVERSE = {{
    "entities": ["ASML.AS", "SAP.DE", "MC.PA", "ENEL.MI", "SAN.MC", "STOXX50E"],
    "project_folder": "{folder_name}",
    "primary_frequency": "business_day",
}}

OUTPUT_ROOT = Path("output")
FIGURE_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"
LOG_DIR = OUTPUT_ROOT / "logs"
for directory in [FIGURE_DIR, TABLE_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(EXPERIMENT["project_name"])
logger.setLevel(logging.INFO)
logger.handlers.clear()
file_handler = logging.FileHandler(LOG_DIR / "experiment.log")
stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

np.random.seed(42)

COLORS = {{
    "primary": "#01696f",
    "accent": "#da7101",
    "q1": "#c0392b",
    "neutral": "#7a7974",
    "bg": "#f7f6f2",
    "blue": "#006494",
    "gold": "#d19900",
    "purple": "#7a39bb",
}}

MODEL_COLORS = {{
    "linear": COLORS["blue"],
    "elastic_net": COLORS["purple"],
    "random_forest": COLORS["primary"],
    "baseline": COLORS["neutral"],
}}

plt.rcParams.update({{
    "figure.facecolor": COLORS["bg"],
    "axes.facecolor": COLORS["bg"],
    "axes.titleweight": "bold",
    "savefig.dpi": EXPERIMENT["figure_dpi"],
}})
sns.set_theme(style="whitegrid", font_scale=1.05)
print(f"Configured {{EXPERIMENT['project_name']}} with database root: {{EXPERIMENT['database_root']}}")
'''


SECTION_CODE = {
    1: r'''def _read_any_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


def _candidate_files(root: Path) -> list[Path]:
    if not root.exists():
        logger.warning("Data root does not exist: %s", root)
        return []
    allowed = {".csv", ".parquet", ".xlsx", ".xls"}
    return [path for path in root.rglob("*") if path.suffix.lower() in allowed and not path.name.startswith(".")]


def synthetic_market_panel() -> pd.DataFrame:
    rng = np.random.default_rng(EXPERIMENT["random_seed"])
    dates = pd.bdate_range(EXPERIMENT["start_date"], periods=EXPERIMENT["synthetic_periods"])
    frames = []
    for entity in UNIVERSE["entities"]:
        shocks = rng.normal(0.0003, 0.012, len(dates))
        close = 100 * np.cumprod(1 + shocks)
        frames.append(pd.DataFrame({
            "date": dates,
            "entity": entity,
            "close_synthetic": close,
            "return_synthetic": shocks,
            "volume_synthetic": rng.lognormal(14.0, 0.35, len(dates)),
        }))
    logger.warning("Real data unavailable; using realistic synthetic market panel.")
    return pd.concat(frames, ignore_index=True)


source_records = []
database_root = Path(EXPERIMENT["database_root"])
repo_data_root = Path(EXPERIMENT["repo_data_root"])
candidate_files = _candidate_files(database_root / UNIVERSE["project_folder"]) + _candidate_files(database_root)[:25] + _candidate_files(repo_data_root)

raw_frames = []
for path in candidate_files[:10]:
    try:
        frame = _read_any_table(path)
        if len(frame) > 0:
            frame = frame.copy()
            frame["source_file"] = str(path)
            raw_frames.append(frame)
            source_records.append({"series": path.stem, "source": "real", "path": str(path), "N": len(frame)})
    except Exception as exc:
        logger.warning("Failed to load %s: %s", path, exc)

if raw_frames:
    raw_data = raw_frames[0]
    logger.info("Loaded primary real dataset: %s", source_records[0]["path"])
else:
    raw_data = synthetic_market_panel()
    source_records.append({"series": "synthetic_market_panel", "source": "synthetic", "path": "", "N": len(raw_data)})

source_summary = pd.DataFrame(source_records)
source_summary.to_csv(TABLE_DIR / "Table_1_data_source_summary.csv", index=False)
print("DATA SOURCE SUMMARY")
print(source_summary.to_string(index=False))
''',
    2: r'''data = raw_data.copy()
lower_map = {column: str(column).strip().lower() for column in data.columns}
data = data.rename(columns=lower_map)

date_candidates = [column for column in data.columns if column in {"date", "datetime", "timestamp", "time"}]
if date_candidates:
    data["date"] = pd.to_datetime(data[date_candidates[0]], errors="coerce")
else:
    logger.warning("No date column found; assigning synthetic business-date index.")
    data["date"] = pd.bdate_range(EXPERIMENT["start_date"], periods=len(data))

if "entity" not in data.columns:
    entity_candidates = [column for column in data.columns if column in {"ticker", "symbol", "asset", "name"}]
    data["entity"] = data[entity_candidates[0]].astype(str) if entity_candidates else UNIVERSE["entities"][0]

numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
if "close" not in data.columns:
    close_candidates = [column for column in numeric_columns if column in {"adj close", "adj_close", "price", "last", "close_synthetic"}]
    if close_candidates:
        data["close"] = pd.to_numeric(data[close_candidates[0]], errors="coerce")
    elif numeric_columns:
        data["close"] = pd.to_numeric(data[numeric_columns[0]], errors="coerce")
        logger.warning("Using %s as close proxy.", numeric_columns[0])
    else:
        fallback = synthetic_market_panel()
        data = fallback.copy()
        data["close"] = data["close_synthetic"]
        logger.warning("No numeric price proxy found; replaced with synthetic panel.")

data = data.dropna(subset=["date", "entity", "close"]).sort_values(["entity", "date"])
data["return_1d"] = data.groupby("entity")["close"].pct_change()
cleaning_summary = pd.DataFrame({
    "N": [len(data)],
    "entities": [data["entity"].nunique()],
    "start_date": [data["date"].min()],
    "end_date": [data["date"].max()],
    "missing_close_pct": [data["close"].isna().mean()],
})
cleaning_summary.round(4).to_csv(TABLE_DIR / "Table_2_cleaning_alignment.csv", index=False)
print(cleaning_summary.to_string(index=False))
''',
    3: r'''FEATURE_BLOCKS = {}


def winsorize_series(series: pd.Series) -> pd.Series:
    lower = series.quantile(0.01)
    upper = series.quantile(0.99)
    return series.clip(lower, upper)


def build_controls_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_price_raw"] = np.log(out["close"]).groupby(out["entity"]).shift(1)
    out["log_price"] = winsorize_series(out["log_price_raw"])
    FEATURE_BLOCKS["controls"] = ["log_price"]
    return out


def build_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for window in EXPERIMENT["rolling_windows"]:
        raw = out.groupby("entity")["close"].transform(lambda series: series.pct_change(window).shift(1))
        out[f"momentum_{window}d_raw"] = raw
        out[f"momentum_{window}d"] = winsorize_series(raw)
    FEATURE_BLOCKS["momentum"] = [f"momentum_{window}d" for window in EXPERIMENT["rolling_windows"]]
    return out


def build_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for window in EXPERIMENT["rolling_windows"]:
        raw = out.groupby("entity")["return_1d"].transform(lambda series: series.rolling(window).std().shift(1))
        out[f"volatility_{window}d_raw"] = raw
        out[f"volatility_{window}d"] = winsorize_series(raw)
    FEATURE_BLOCKS["risk"] = [f"volatility_{window}d" for window in EXPERIMENT["rolling_windows"]]
    return out


def build_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    raw = out.groupby("entity")["return_1d"].transform(lambda series: series.rolling(EXPERIMENT["rolling_windows"][1]).mean().shift(1))
    out["quality_return_raw"] = raw
    out["quality_return"] = winsorize_series(raw)
    FEATURE_BLOCKS["quality"] = ["quality_return"]
    return out


features = data.pipe(build_controls_features).pipe(build_momentum_features).pipe(build_risk_features).pipe(build_quality_features)
feature_columns = [column for block in FEATURE_BLOCKS.values() for column in block]
missing_rows = []
for block, columns in FEATURE_BLOCKS.items():
    for column in columns:
        missing_rows.append({"block": block, "feature": column, "missing_pct": features[column].isna().mean(), "N": features[column].notna().sum()})
feature_missingness = pd.DataFrame(missing_rows)
feature_missingness.round(4).to_csv(TABLE_DIR / "Table_feature_missingness.csv", index=False)
print(feature_missingness.round(3).to_string(index=False))
''',
    4: r'''labels = features.copy()
for horizon in EXPERIMENT["horizons"]:
    labels[f"forward_return_{horizon}d"] = labels.groupby("entity")["close"].shift(-horizon) / labels["close"] - 1
target = EXPERIMENT["target"]
labels["target_quantile"] = pd.qcut(labels[target].rank(method="first"), EXPERIMENT["n_quantiles"], labels=False, duplicates="drop")
target_summary = labels[target].describe().to_frame("value").reset_index().rename(columns={"index": "stat"})
target_summary["N"] = labels[target].notna().sum()
target_summary.round(4).to_csv(TABLE_DIR / "Table_4_targets_labels.csv", index=False)
print(target_summary.round(3).to_string(index=False))
''',
    5: r'''model_data = labels.dropna(subset=feature_columns + [target]).copy()
desc = model_data[feature_columns + [target]].describe().T.reset_index().rename(columns={"index": "variable", "count": "N"})
desc.round(4).to_csv(TABLE_DIR / "Table_5_descriptive_stats.csv", index=False)
fig, ax = plt.subplots(figsize=EXPERIMENT["single_figsize"])
model_data[target].hist(ax=ax, bins=30, color=COLORS["primary"])
ax.axvline(0, color="black", linestyle="--", lw=0.8)
ax.set_title("Target Distribution", fontweight="bold")
ax.set_xlabel("Forward return (decimal)")
ax.set_ylabel("Frequency (count)")
plt.tight_layout()
fig.savefig(FIGURE_DIR / "Figure_5_target_distribution.png", dpi=EXPERIMENT["figure_dpi"])
plt.show()
print(desc.round(3).to_string(index=False))
''',
    6: r'''event_study = model_data.groupby("date")[target].mean().reset_index()
event_table = event_study.assign(N=model_data.groupby("date")[target].size().to_numpy())
event_table.round(4).to_csv(TABLE_DIR / "Table_6_event_study.csv", index=False)
fig, ax = plt.subplots(figsize=EXPERIMENT["single_figsize"])
ax.plot(event_study["date"], event_study[target], color=COLORS["blue"], label="Average target")
ax.axhline(0, color="black", linestyle="--", lw=0.8)
ax.set_title("Average Forward Return Through Time", fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Forward return (decimal)")
ax.legend()
plt.tight_layout()
fig.savefig(FIGURE_DIR / "Figure_6_event_study.png", dpi=EXPERIMENT["figure_dpi"])
plt.show()
print(event_table.tail().round(3).to_string(index=False))
''',
    7: r'''diagnostics = []
for column in feature_columns:
    subset = model_data[[column, target]].dropna()
    ic = subset[column].corr(subset[target], method="spearman")
    diagnostics.append({"feature": column, "spearman_ic": ic, "N": len(subset)})
single_factor = pd.DataFrame(diagnostics).sort_values("spearman_ic", ascending=False)
single_factor.round(4).to_csv(TABLE_DIR / "Table_7_single_factor_diagnostics.csv", index=False)
fig, ax = plt.subplots(figsize=EXPERIMENT["single_figsize"])
sns.barplot(single_factor, x="spearman_ic", y="feature", ax=ax, color=COLORS["primary"])
ax.axvline(0, color="black", linestyle="--", lw=0.8)
ax.set_title("Single-Factor Rank IC", fontweight="bold")
ax.set_xlabel("Spearman IC (correlation)")
ax.set_ylabel("Feature")
plt.tight_layout()
fig.savefig(FIGURE_DIR / "Figure_7_single_factor_ic.png", dpi=EXPERIMENT["figure_dpi"])
plt.show()
print(single_factor.round(3).to_string(index=False))
''',
    8: r'''X = sm.add_constant(model_data[feature_columns])
y = model_data[target]
ols = sm.OLS(y, X).fit()
regression_table = pd.DataFrame({
    "feature": ols.params.index,
    "coef": ols.params.values,
    "t_stat": ols.tvalues.values,
    "p_value": ols.pvalues.values,
    "N": int(ols.nobs),
})
regression_table.round(4).to_csv(TABLE_DIR / "Table_8_statistical_models.csv", index=False)
print(regression_table.round(3).to_string(index=False))
''',
    9: r'''def walk_forward_run(active_features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(model_data.loc[model_data["date"] >= pd.Timestamp(EXPERIMENT["test_start"]), "date"].unique())
    predictions = []
    metrics = []
    for test_date in dates[::EXPERIMENT["min_test_rows"]]:
        test_start = pd.Timestamp(test_date)
        test_end = test_start + pd.offsets.BDay(EXPERIMENT["min_test_rows"] - 1)
        embargo_start = test_start - pd.offsets.BDay(EXPERIMENT["embargo"])
        train = model_data[model_data["date"] < embargo_start]
        test = model_data[(model_data["date"] >= test_start) & (model_data["date"] <= test_end)]
        if len(train) < EXPERIMENT["min_train_rows"] or len(test) < EXPERIMENT["min_test_rows"]:
            continue
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[active_features])
        X_test = scaler.transform(test[active_features])
        y_train = train[target]
        y_test = test[target]
        fitted = {
            "linear": LinearRegression(),
            "elastic_net": ElasticNet(alpha=0.01, l1_ratio=0.25, random_state=EXPERIMENT["random_seed"]),
            "random_forest": RandomForestRegressor(n_estimators=100, max_depth=4, random_state=EXPERIMENT["random_seed"]),
        }
        for model_name in EXPERIMENT["models"]:
            estimator = fitted[model_name]
            estimator.fit(X_train, y_train)
            pred = estimator.predict(X_test)
            fold = pd.DataFrame({"date": test["date"].values, "entity": test["entity"].values, "model": model_name, "prediction": pred, "actual": y_test.values})
            predictions.append(fold)
            metrics.append({
                "fold_start": test_start,
                "model": model_name,
                "rmse": mean_squared_error(y_test, pred, squared=False),
                "mae": mean_absolute_error(y_test, pred),
                "r2": r2_score(y_test, pred),
                "N": len(test),
            })
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics)


wf_predictions, wf_metrics = walk_forward_run(feature_columns)
wf_metrics.round(4).to_csv(TABLE_DIR / "Table_9_walk_forward_metrics.csv", index=False)
wf_predictions.round(4).to_csv(TABLE_DIR / "Table_9_walk_forward_predictions.csv", index=False)
print(wf_metrics.groupby("model")[["rmse", "mae", "r2", "N"]].mean().round(3).to_string())
''',
    10: r'''baseline_features = FEATURE_BLOCKS["controls"]
ablation_rows = []
_, baseline_metrics = walk_forward_run(baseline_features)
baseline_rmse = baseline_metrics.groupby("model")["rmse"].mean().min()
for block, columns in FEATURE_BLOCKS.items():
    active = sorted(set(baseline_features + columns))
    _, block_metrics = walk_forward_run(active)
    block_rmse = block_metrics.groupby("model")["rmse"].mean().min()
    ablation_rows.append({"block": block, "rmse": block_rmse, "delta_metric": baseline_rmse - block_rmse, "N": int(block_metrics["N"].sum())})
ablation = pd.DataFrame(ablation_rows).sort_values("delta_metric", ascending=True)
ablation.round(4).to_csv(TABLE_DIR / "Table_ablation.csv", index=False)
fig, ax = plt.subplots(figsize=EXPERIMENT["single_figsize"])
ax.barh(ablation["block"], ablation["delta_metric"], color=COLORS["primary"])
ax.axvline(0, color="black", linestyle="--", lw=0.8)
ax.set_title("Feature Block Ablation vs Controls", fontweight="bold")
ax.set_xlabel("Delta RMSE improvement (decimal)")
ax.set_ylabel("Feature block")
plt.tight_layout()
fig.savefig(FIGURE_DIR / "Figure_10_ablation_delta.png", dpi=EXPERIMENT["figure_dpi"])
plt.show()
print(ablation.round(3).to_string(index=False))
''',
    11: r'''best_model = wf_metrics.groupby("model")["rmse"].mean().idxmin()
signals = wf_predictions[wf_predictions["model"] == best_model].copy()
signals["rank"] = signals.groupby("date")["prediction"].rank(pct=True)
signals["position"] = np.where(signals["rank"] >= 1 - 1 / EXPERIMENT["n_quantiles"], 1, np.where(signals["rank"] <= 1 / EXPERIMENT["n_quantiles"], -1, 0))
gross = signals.groupby("date").apply(lambda x: (x["position"] * x["actual"]).mean()).rename("gross_return")
turnover = signals.pivot_table(index="date", columns="entity", values="position", fill_value=0).diff().abs().mean(axis=1).fillna(0)
strategy = pd.DataFrame({"gross_return": gross, "turnover": turnover})
strategy["net_return"] = strategy["gross_return"] - strategy["turnover"] * EXPERIMENT["cost_bps"] / 10000
strategy["cum_net_return"] = (1 + strategy["net_return"]).cumprod() - 1
strategy["N"] = signals.groupby("date").size()
strategy.round(4).to_csv(TABLE_DIR / "Table_11_backtest_strategy.csv")
fig, ax = plt.subplots(figsize=EXPERIMENT["single_figsize"])
ax.plot(strategy.index, strategy["cum_net_return"], color=COLORS["primary"], label="Net strategy")
ax.axhline(0, color="black", linestyle="--", lw=0.8)
ax.set_title("Walk-Forward Long-Short Backtest", fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative return (decimal)")
ax.legend()
plt.tight_layout()
fig.savefig(FIGURE_DIR / "Figure_11_backtest_strategy.png", dpi=EXPERIMENT["figure_dpi"])
plt.show()
print(strategy.tail().round(3).to_string())
''',
    12: r'''train_cutoff = pd.Timestamp(EXPERIMENT["test_start"])
train = model_data[model_data["date"] < train_cutoff]
test = model_data[model_data["date"] >= train_cutoff]
scaler = StandardScaler()
X_train = scaler.fit_transform(train[feature_columns])
X_test = scaler.transform(test[feature_columns])
model = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=EXPERIMENT["random_seed"])
model.fit(X_train, train[target])
importance = permutation_importance(model, X_test, test[target], n_repeats=5, random_state=EXPERIMENT["random_seed"])
interpretability = pd.DataFrame({"feature": feature_columns, "importance": importance.importances_mean, "N": len(test)}).sort_values("importance", ascending=False)
interpretability.round(4).to_csv(TABLE_DIR / "Table_12_interpretability.csv", index=False)
fig, ax = plt.subplots(figsize=EXPERIMENT["single_figsize"])
sns.barplot(interpretability, x="importance", y="feature", ax=ax, color=COLORS["accent"])
ax.axvline(0, color="black", linestyle="--", lw=0.8)
ax.set_title("Permutation Importance", fontweight="bold")
ax.set_xlabel("Importance (R2 decrease)")
ax.set_ylabel("Feature")
plt.tight_layout()
fig.savefig(FIGURE_DIR / "Figure_12_interpretability.png", dpi=EXPERIMENT["figure_dpi"])
plt.show()
print(interpretability.round(3).to_string(index=False))
''',
    13: r'''robust_rows = []
median_date = model_data["date"].median()
for label, subset in [("early", model_data[model_data["date"] <= median_date]), ("late", model_data[model_data["date"] > median_date])]:
    corr = subset[feature_columns[0]].corr(subset[target], method="spearman")
    robust_rows.append({"test": "subperiod_time", "bucket": label, "metric": corr, "N": len(subset)})

placebo = model_data.copy()
placebo[target] = np.random.default_rng(EXPERIMENT["random_seed"]).permutation(placebo[target].to_numpy())
placebo_metric = placebo[feature_columns[0]].corr(placebo[target], method="spearman")
robust_rows.append({"test": "placebo_target", "bucket": "permuted", "metric": placebo_metric, "N": len(placebo)})

for cost in [0.0, EXPERIMENT["cost_bps"], EXPERIMENT["cost_bps"] * 2]:
    net = strategy["gross_return"] - strategy["turnover"] * cost / 10000
    robust_rows.append({"test": "cost_sensitivity", "bucket": f"{cost:.1f}_bps", "metric": net.mean(), "N": len(net)})

robustness = pd.DataFrame(robust_rows)
robustness.round(4).to_csv(TABLE_DIR / "Table_VII_robustness.csv", index=False)
heat = robustness[robustness["test"] == "cost_sensitivity"].pivot_table(index="test", columns="bucket", values="metric")
fig, ax = plt.subplots(figsize=EXPERIMENT["single_figsize"])
sns.heatmap(heat, annot=True, fmt=".4f", cmap="viridis", ax=ax)
ax.set_title("Robustness Cost Sensitivity", fontweight="bold")
ax.set_xlabel("Transaction cost (bps)")
ax.set_ylabel("Test")
plt.tight_layout()
fig.savefig(FIGURE_DIR / "Figure_13_robustness_heatmap.png", dpi=EXPERIMENT["figure_dpi"])
plt.show()
print(robustness.round(3).to_string(index=False))
''',
    14: r'''import os
from pathlib import Path
tables  = sorted(Path("output/tables").glob("*.csv"))
figures = sorted(Path("output/figures").glob("*.png"))
print(f"✅ EXPERIMENT COMPLETE")
print(f"   Tables : {len(tables)}")
print(f"   Figures: {len(figures)}")
for f in tables:  print(f"   📋 {f.name}")
for f in figures: print(f"   📊 {f.name}")
''',
}


def build_notebook(folder: Path, database_root: Path) -> dict:
    title = slug_to_title(folder.name)
    cells = [markdown_cell(f"# {title} Definitive Research Notebook\n\nAGENTS.md-compliant scaffold generated for `{folder.name}`.")]
    for section in range(len(SECTION_HEADERS)):
        cells.append(markdown_cell(section_markdown(section)))
        if section == 0:
            cells.append(code_cell(setup_code(title, folder.name, database_root)))
        else:
            cells.append(code_cell(SECTION_CODE[section]))
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def current_headers(path: Path) -> list[str]:
    try:
        notebook = json.loads(path.read_text())
    except Exception:
        return []
    headers: list[str] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = "".join(cell.get("source", []))
        headers.extend(line.strip() for line in source.splitlines() if line.startswith("## "))
    return headers


def has_compliant_definitive_notebook(folder: Path) -> bool:
    return any(current_headers(path) == SECTION_HEADERS for path in folder.glob("*definitive.ipynb"))


def discover_chapter_folders(root: Path) -> list[Path]:
    folders = [path for path in root.iterdir() if path.is_dir() and re.match(r"^\d{2}[_-]", path.name)]
    return sorted(folders, key=lambda path: path.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate universal research notebooks for chapter folders.")
    parser.add_argument("--database-root", type=Path, default=DEFAULT_DATABASE_ROOT)
    parser.add_argument("--overwrite-noncompliant", action="store_true")
    parser.add_argument("--force-overwrite", action="store_true", help="Regenerate target notebooks even when already compliant.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("folders", nargs="*", type=Path)
    args = parser.parse_args()

    folders = args.folders or discover_chapter_folders(REPO_ROOT)
    written: list[Path] = []
    skipped: list[Path] = []
    for folder in folders:
        folder = folder if folder.is_absolute() else REPO_ROOT / folder
        target = folder / notebook_name(folder)
        if not args.force_overwrite and not target.exists() and has_compliant_definitive_notebook(folder):
            skipped.extend(sorted(folder.glob("*definitive.ipynb")))
            continue
        headers = current_headers(target) if target.exists() else []
        compliant = headers == SECTION_HEADERS
        if target.exists() and compliant and not args.force_overwrite:
            skipped.append(target)
            continue
        if target.exists() and not args.overwrite_noncompliant and not args.force_overwrite:
            skipped.append(target)
            continue
        if not args.dry_run:
            notebook = build_notebook(folder, args.database_root)
            target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
        written.append(target)

    print(f"Generated {len(written)} notebooks")
    for path in written:
        print(f"   {path.relative_to(REPO_ROOT)}")
    print(f"Skipped {len(skipped)} notebooks")
    for path in skipped:
        print(f"   {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
