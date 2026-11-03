from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from asi_project.config import DatasetContract, ProjectConfig
from asi_project.data import make_xy, split_train_test
from asi_project.metrics import compute_metrics
from asi_project.model import build_baseline_pipeline


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
