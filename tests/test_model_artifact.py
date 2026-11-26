import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from asi_project import model_artifact
from asi_project.model_artifact import publish_production_model
from asi_project.model_package import save_model_package


class WandbUsageError(Exception):
    pass


@pytest.fixture
def publication(tmp_path, monkeypatch, project):
    artifact = Mock()
    run_context = MagicMock()
    run = run_context.__enter__.return_value
    run.log_artifact.return_value.wait.return_value.version = "v3"
    api = Mock()
    api.artifact.return_value.version = "v3"
    wandb = SimpleNamespace(
        Artifact=Mock(return_value=artifact),
        init=Mock(return_value=run_context),
        Api=Mock(return_value=api),
        errors=SimpleNamespace(UsageError=WandbUsageError),
    )
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    monkeypatch.delenv("WANDB_MODE", raising=False)

    from asi_project.tracking import get_data_fingerprint

    raw_data = pd.DataFrame({"raw": [1, 2]})
    load_dataset = Mock(return_value=raw_data)
    monkeypatch.setattr(model_artifact, "load_dataset", load_dataset)
    monkeypatch.setattr(model_artifact, "load_project_config", lambda: project.config)
    features = list(project.config.model_features)
    model = DummyClassifier().fit(pd.DataFrame([{name: 0 for name in features}] * 2), [0, 1])
    package = save_model_package(
        model,
        tmp_path / "package",
        model_type="sklearn",
        features=features,
        target=project.contract.target,
        positive_class=project.contract.positive_class,
        metrics={"f1": 0.1},
        git_sha="a" * 40,
        data_fingerprint=get_data_fingerprint(raw_data),
    )
    return SimpleNamespace(
        package=package, wandb=wandb, artifact=artifact, run=run, api=api, load_dataset=load_dataset
    )


def test_publish_package_checks_production_alias(publication):
    reference = publish_production_model(publication.package, entity="course-team")

    assert reference == "course-team/online-shoppers/online-shoppers-model:v3"
    assert publication.load_dataset.call_args.kwargs["mode"] == "full"
    publication.artifact.add_dir.assert_called_once_with(
        str(publication.package / "model"), name="model"
    )
    publication.artifact.add_file.assert_called_once_with(
        str(publication.package / "metadata.json"), name="metadata.json"
    )
    publication.wandb.init.assert_called_once_with(
        entity="course-team", project="online-shoppers", job_type="publish-model", mode="online"
    )
    publication.run.log_artifact.assert_called_once_with(
        publication.artifact, aliases=["production"]
    )
    publication.run.log_artifact.return_value.wait.assert_called_once_with()
    publication.api.artifact.assert_called_once_with(
        "course-team/online-shoppers/online-shoppers-model:production", type="model"
    )


def test_publish_rejects_package_from_other_data(publication):
    publication.load_dataset.return_value = pd.DataFrame({"raw": [3, 4]})

    with pytest.raises(ValueError, match="Fingerprint"):
        publish_production_model(publication.package, entity="course-team")

    publication.wandb.Artifact.assert_not_called()
    publication.wandb.init.assert_not_called()


def test_publish_does_not_override_disabled_wandb(publication, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")

    with pytest.raises(RuntimeError, match="online"):
        publish_production_model(publication.package, entity="course-team")

    publication.wandb.init.assert_not_called()


def test_publish_stops_before_upload_if_wandb_login_is_required(publication):
    publication.wandb.Api.side_effect = WandbUsageError("Use `wandb login` to log in.")

    with pytest.raises(RuntimeError, match="wandb login"):
        publish_production_model(publication.package, entity="course-team")

    publication.wandb.Artifact.assert_not_called()
    publication.wandb.init.assert_not_called()


def test_publish_does_not_report_success_if_production_points_elsewhere(publication):
    publication.api.artifact.return_value.version = "v2"

    with pytest.raises(RuntimeError, match="wskazuje v2, zamiast v3"):
        publish_production_model(publication.package, entity="course-team")
