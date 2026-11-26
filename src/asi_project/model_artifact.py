import os
import re
from pathlib import Path

from asi_project.config import get_contract, load_project_config
from asi_project.data import load_dataset
from asi_project.model_package import load_model_package

WANDB_PROJECT = "online-shoppers"
ARTIFACT_NAME = "online-shoppers-model"
PRODUCTION_ALIAS = "production"


def publish_production_model(package_dir: Path, *, entity: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+", entity):
        raise ValueError("Podaj nazwę zespołu w W&B, bez projektu ani adresu URL.")
    if os.environ.get("WANDB_MODE") not in (None, "", "online"):
        raise RuntimeError("Publikacja wymaga W&B w trybie online.")

    import wandb

    from asi_project.tracking import get_data_fingerprint

    _, metadata = load_model_package(package_dir)
    config = load_project_config()
    contract = get_contract(config)
    if (
        metadata["features"] != list(config.model_features)
        or metadata["target"] != contract.target
        or metadata["positive_class"] != contract.positive_class
    ):
        raise ValueError("Pakiet nie jest zgodny z kontraktem modelu w projekcie.")

    fingerprint = get_data_fingerprint(load_dataset(config, contract, mode="full"))
    if metadata["data_fingerprint"] != fingerprint:
        raise ValueError("Fingerprint pakietu nie odpowiada pełnym danym projektu.")

    try:
        api = wandb.Api()
    except wandb.errors.UsageError as error:
        raise RuntimeError(str(error)) from error

    artifact = wandb.Artifact(name=ARTIFACT_NAME, type="model", metadata=metadata)
    artifact.add_dir(str(package_dir / "model"), name="model")
    artifact.add_file(str(package_dir / "metadata.json"), name="metadata.json")

    with wandb.init(
        entity=entity, project=WANDB_PROJECT, job_type="publish-model", mode="online"
    ) as run:
        published = run.log_artifact(artifact, aliases=[PRODUCTION_ALIAS]).wait()
        version = published.version

    collection = f"{entity}/{WANDB_PROJECT}/{ARTIFACT_NAME}"
    production = api.artifact(f"{collection}:{PRODUCTION_ALIAS}", type="model")
    if production.version != version:
        raise RuntimeError(
            f"Alias {PRODUCTION_ALIAS} wskazuje {production.version}, zamiast {version}."
        )
    return f"{collection}:{version}"
