import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from asi_project.model import build_baseline_pipeline
from asi_project.model_package import load_model_package, save_model_package


@pytest.fixture
def candidate():
    X = pd.DataFrame({"Duration": [2, 5, 8, 12], "VisitorType": ["New", "Return", "New", "Return"]})
    y = [0, 0, 1, 1]
    model = build_baseline_pipeline(("Duration",), ("VisitorType",)).fit(X, y)
    metadata = {
        "model_type": "sklearn",
        "features": list(X.columns),
        "target": "Revenue",
        "positive_class": "true",
        "metrics": {"f1": 1.0},
        "git_sha": "a" * 40,
        "data_fingerprint": "b" * 32,
    }
    return X, model, metadata


def test_sklearn_package_roundtrip(tmp_path, candidate):
    X, model, metadata = candidate
    package = save_model_package(model, tmp_path / "package", **metadata)

    assert (package / "model" / "model.joblib").is_file()
    loaded_model, loaded_metadata = load_model_package(package)
    np.testing.assert_array_equal(loaded_model.predict(X), model.predict(X))
    np.testing.assert_allclose(loaded_model.predict_proba(X), model.predict_proba(X))
    assert loaded_metadata == {
        **metadata,
        "positive_class_label": 1,
        "library_versions": loaded_metadata["library_versions"],
    }
    assert loaded_metadata["library_versions"]["scikit-learn"]


def test_autogluon_package_uses_saved_directory_after_move(tmp_path, candidate, monkeypatch):
    X, model, metadata = candidate
    source = tmp_path / "trainer-output"
    source.mkdir()
    joblib.dump(model, source / "fake-predictor.joblib")

    # Stub biblioteki: testujemy kopiowanie i odczyt pakietu, nie trening AutoGluona.
    autogluon = ModuleType("autogluon")
    tabular = ModuleType("autogluon.tabular")
    tabular.TabularPredictor = SimpleNamespace(
        load=lambda path: joblib.load(Path(path) / "fake-predictor.joblib")
    )
    monkeypatch.setitem(sys.modules, "autogluon", autogluon)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", tabular)

    package = save_model_package(
        SimpleNamespace(path=source),
        tmp_path / "package",
        **{**metadata, "model_type": "autogluon"},
    )
    moved = package.rename(tmp_path / "downloaded")
    (source / "fake-predictor.joblib").unlink()

    loaded_model, loaded_metadata = load_model_package(moved)
    np.testing.assert_array_equal(loaded_model.predict(X), model.predict(X))
    assert loaded_metadata["model_type"] == "autogluon"


def test_package_requires_metadata_before_loading_model(tmp_path, candidate):
    _, model, metadata = candidate
    package = save_model_package(model, tmp_path / "package", **metadata)
    metadata_path = package / "metadata.json"
    incomplete = json.loads(metadata_path.read_text(encoding="utf-8"))
    del incomplete["data_fingerprint"]
    metadata_path.write_text(json.dumps(incomplete), encoding="utf-8")

    with pytest.raises(ValueError, match="data_fingerprint"):
        load_model_package(package)


def test_package_overwrites_only_existing_package_on_request(tmp_path, candidate):
    _, model, metadata = candidate
    package = save_model_package(model, tmp_path / "package", **metadata)
    new_metadata = {**metadata, "metrics": {"f1": 0.5}}
    with pytest.raises(FileExistsError):
        save_model_package(model, package, **new_metadata)
    assert load_model_package(package)[1]["metrics"] == metadata["metrics"]

    save_model_package(model, package, overwrite=True, **new_metadata)
    assert load_model_package(package)[1]["metrics"] == new_metadata["metrics"]

    other_folder = tmp_path / "other"
    other_folder.mkdir()
    sentinel = other_folder / "untouched.txt"
    sentinel.write_text("user data", encoding="utf-8")
    with pytest.raises(FileExistsError):
        save_model_package(model, other_folder, overwrite=True, **metadata)
    assert sentinel.read_text(encoding="utf-8") == "user data"
