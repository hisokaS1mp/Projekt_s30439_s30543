import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from asi_project.model_package import save_model_package
from asi_project.model_smoke import _predict_rows, smoke_production_model


def test_smoke_downloads_production_and_predicts_from_only_the_downloaded_package(
    tmp_path, monkeypatch, project
):
    features = list(project.config.model_features)
    model = DummyClassifier(strategy="constant", constant=1).fit(
        project.sample.loc[:, features].head(2), [0, 1]
    )
    package = save_model_package(
        model,
        tmp_path / "published",
        model_type="sklearn",
        features=features,
        target=project.contract.target,
        positive_class=project.contract.positive_class,
        metrics={"f1": 0.1},
        git_sha="a" * 40,
        data_fingerprint="b" * 32,
    )
    download = Mock(
        side_effect=lambda *, root, skip_cache: shutil.copytree(package, root, dirs_exist_ok=True)
    )
    artifact = SimpleNamespace(download=download, version="v0", digest="local-test-digest")
    api = Mock()
    api.artifact.return_value = artifact
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=lambda: api))
    monkeypatch.delenv("WANDB_MODE", raising=False)

    result = smoke_production_model(entity="course-team")

    api.artifact.assert_called_once_with(
        "course-team/online-shoppers/online-shoppers-model:production", type="model"
    )
    download.assert_called_once()
    assert download.call_args.kwargs["skip_cache"] is True
    assert not Path(download.call_args.kwargs["root"]).exists()
    assert result == {
        "artifact": "course-team/online-shoppers/online-shoppers-model:v0",
        "digest": "local-test-digest",
        "predictions": [{"label": 1, "positive_probability": 1.0, "model_version": "v0"}] * 5,
    }


def test_smoke_normalizes_only_required_features_and_selects_positive_probability_by_label():
    sample = pd.DataFrame(
        {"unneeded": [0] * 6, "VisitorType": [" Return "] * 6, "Revenue": [1] * 6}
    )
    model = Mock()
    model.predict.return_value = pd.Series([1] * 5)
    model.predict_proba.return_value = pd.DataFrame({1: [0.3] * 5, 0: [0.7] * 5})

    result = _predict_rows(
        model, sample, {"features": ["VisitorType"], "positive_class_label": 1}, "v0"
    )

    expected_X = pd.DataFrame({"VisitorType": ["Return"] * 5})
    pd.testing.assert_frame_equal(model.predict.call_args.args[0], expected_X)
    pd.testing.assert_frame_equal(model.predict_proba.call_args.args[0], expected_X)
    assert result == [{"label": 1, "positive_probability": 0.3, "model_version": "v0"}] * 5


@pytest.mark.parametrize(
    ("labels", "positive_probability", "expected_error"),
    [
        ([1] * 4, 0.3, "etykiet"),
        ([2] * 5, 0.3, "etykiet"),
        ([1] * 5, np.nan, "prawdopodobieństwo"),
        ([1] * 5, 1.1, "prawdopodobieństwo"),
    ],
)
def test_smoke_rejects_invalid_model_output(labels, positive_probability, expected_error):
    sample = pd.DataFrame({"VisitorType": ["Return"] * 5})
    model = Mock()
    model.predict.return_value = np.array(labels)
    model.predict_proba.return_value = pd.DataFrame({0: [0.7] * 5, 1: [positive_probability] * 5})

    with pytest.raises(ValueError, match=f"(?i){expected_error}"):
        _predict_rows(model, sample, {"features": ["VisitorType"], "positive_class_label": 1}, "v0")
