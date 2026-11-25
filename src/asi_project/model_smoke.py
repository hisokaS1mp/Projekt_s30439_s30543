import os
from tempfile import TemporaryDirectory
from typing import Any, TypedDict

import numpy as np
import pandas as pd

from asi_project.config import get_contract, load_project_config
from asi_project.data import load_dataset, normalize_input
from asi_project.model_artifact import ARTIFACT_NAME, PRODUCTION_ALIAS, WANDB_PROJECT
from asi_project.model_package import load_model_package

SMOKE_ROWS = 5


class Prediction(TypedDict):
    label: int
    positive_probability: float
    model_version: str


class SmokeResult(TypedDict):
    artifact: str
    digest: str
    predictions: list[Prediction]


def _predict_rows(
    model: Any, sample: pd.DataFrame, metadata: dict[str, Any], version: str
) -> list[Prediction]:
    if len(sample) < SMOKE_ROWS:
        raise ValueError(f"Próbka do smoke testu musi mieć co najmniej {SMOKE_ROWS} wierszy.")
    features = metadata["features"]
    missing = set(features) - set(sample.columns)
    if missing:
        raise ValueError(f"W próbce brakuje cech: {', '.join(sorted(missing))}.")
    if not version:
        raise ValueError("W odpowiedzi musi znaleźć się wersja modelu.")

    X = normalize_input(sample.head(SMOKE_ROWS).loc[:, features])
    positive_label = metadata["positive_class_label"]
    labels = np.asarray(model.predict(X))
    probabilities = model.predict_proba(X)
    classes = (
        list(probabilities.columns)
        if isinstance(probabilities, pd.DataFrame)
        else list(model.classes_)
    )
    probabilities = np.asarray(probabilities)
    if (
        len(classes) != 2
        or set(classes) != {0, 1}
        or positive_label not in classes
        or probabilities.shape != (SMOKE_ROWS, 2)
    ):
        raise ValueError("Nieprawidłowy schemat macierzy prawdopodobieństw modelu.")
    positive_probabilities = probabilities[:, classes.index(positive_label)]

    if labels.shape != (SMOKE_ROWS,) or not np.isin(labels, (0, 1)).all():
        raise ValueError("Model nie zwrócił pięciu etykiet binarnych.")
    if (
        positive_probabilities.dtype.kind not in "fiu"
        or not ((0 <= positive_probabilities) & (positive_probabilities <= 1)).all()
    ):
        raise ValueError("Prawdopodobieństwo klasy pozytywnej musi być liczbą z zakresu [0, 1].")

    return [
        {
            "label": int(label),
            "positive_probability": float(probability),
            "model_version": version,
        }
        for label, probability in zip(labels, positive_probabilities, strict=True)
    ]


def smoke_production_model(*, entity: str) -> SmokeResult:
    if os.environ.get("WANDB_MODE") not in (None, "", "online"):
        raise RuntimeError("Smoke test artefaktu wymaga W&B w trybie online.")

    import wandb

    config = load_project_config()
    sample = load_dataset(config, get_contract(config), mode="sample")
    collection = f"{entity}/{WANDB_PROJECT}/{ARTIFACT_NAME}"
    with TemporaryDirectory(prefix="asi-model-smoke-") as root:
        try:
            artifact = wandb.Api().artifact(f"{collection}:{PRODUCTION_ALIAS}", type="model")
            package = artifact.download(root=root, skip_cache=True)
        except wandb.errors.Error as error:
            raise RuntimeError(str(error)) from error
        model, metadata = load_model_package(package)
        predictions = _predict_rows(model, sample, metadata, artifact.version)

    return {
        "artifact": f"{collection}:{artifact.version}",
        "digest": artifact.digest,
        "predictions": predictions,
    }
