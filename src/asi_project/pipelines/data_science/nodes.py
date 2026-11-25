import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.pipeline import Pipeline

from asi_project.config import DatasetContract, ProjectConfig
from asi_project.data import make_xy, split_train_test
from asi_project.metrics import compute_metrics
from asi_project.model import build_baseline_pipeline
from asi_project.model_package import benchmark_model_package, save_model_package


def split_data(
    processed_data: pd.DataFrame,
    contract: DatasetContract,
    config: ProjectConfig,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Buduje X/y i dzieli dane zgodnie ze strategią kontraktu."""
    X, y = make_xy(processed_data, contract, config.model_features)
    return split_train_test(X, y, contract, test_size, random_state)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ProjectConfig,
) -> Pipeline:
    """Dopasowuje preprocessing i model wyłącznie na zbiorze treningowym."""
    model = build_baseline_pipeline(config.numeric_features, config.categorical_features)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Oblicza metryki modelu na zbiorze testowym."""
    return compute_metrics(y_test, model.predict(X_test))


def train_autogluon_model(
    X_train: pd.DataFrame, y_train: pd.Series, target_col: str, time_limit: int, presets: str
) -> TabularPredictor:
    """
    Trenuje model za pomocą AutoGluona z zadanym limitem czasu i konfiguracją.
    """
    train_data = X_train.copy()
    train_data[target_col] = y_train

    predictor = TabularPredictor(label=target_col, eval_metric="f1").fit(
        train_data, time_limit=time_limit, presets=presets, verbosity=2
    )
    return predictor


def benchmark_candidates(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_col: str,
    baseline_model: Pipeline,
    baseline_metrics: dict[str, Any],
    ag_fast_model: TabularPredictor,
    ag_fast_metrics: dict[str, Any],
    ag_hq_model: TabularPredictor,
    ag_hq_metrics: dict[str, Any],
) -> pd.DataFrame:
    """Porównuje kandydatów pod kątem metryk jakości oraz kosztów technicznych."""
    candidates = [
        ("baseline", baseline_model, "sklearn", baseline_metrics),
        ("ag_fast", ag_fast_model, "autogluon", ag_fast_metrics),
        ("ag_hq", ag_hq_model, "autogluon", ag_hq_metrics),
    ]

    rows: list[dict[str, Any]] = []
    features = list(X_test.columns)
    positive_class = 1

    with tempfile.TemporaryDirectory(prefix="asi_bench_") as tmp_root:
        for name, mdl, mtype, met in candidates:
            pkg_dir = Path(tmp_root) / name
            save_model_package(
                model=mdl,
                package_dir=pkg_dir,
                model_type=mtype,
                features=features,
                target=target_col,
                positive_class=positive_class,
                metrics=met,
                fingerprint_df=X_test,
            )
            bench = benchmark_model_package(pkg_dir, X_test)
            rows.append(
                {
                    "model": name,
                    **met,
                    **bench,
                }
            )

    df_summary = pd.DataFrame(rows)
    df_summary = df_summary.drop(columns=["confusion_matrix"], errors="ignore")
    df_summary = df_summary.round(4)
    return df_summary


def export_production_package(
    model: TabularPredictor,
    metrics: dict[str, Any],
    X_test: pd.DataFrame,
    raw_data: pd.DataFrame,
    target_col: str,
    package_dir: str = "data/06_models/production_package",
) -> None:
    """Zapisuje wybrany model produkcyjny (ag_hq) wraz z metadata.json."""
    save_model_package(
        model=model,
        package_dir=package_dir,
        model_type="autogluon",
        features=list(X_test.columns),
        target=target_col,
        positive_class=1,
        metrics=metrics,
        fingerprint_df=raw_data,
    )
