import hashlib
import subprocess
from typing import Any

import pandas as pd
import wandb


def get_git_sha() -> str:
    """Pobiera aktualny hash commitu z Git."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"


def get_data_fingerprint(df: pd.DataFrame) -> str:
    """Generuje unikalny hash (fingerprint) na podstawie zawartości ramki danych."""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def log_baseline_to_wandb(
    metrics: dict[str, Any],
    raw_data: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> None:
    """Rejestruje eksperyment baseline w Weights & Biases."""

    # Inicjalizacja runu
    wandb.init(project="online-shoppers", name="baseline-run", job_type="baseline")

    # Logowanie parametrów i metadanych
    wandb.config.update(
        {
            "model_type": "baseline",
            "test_size": test_size,
            "random_state": random_state,
            "git_sha": get_git_sha(),
            "data_fingerprint": get_data_fingerprint(raw_data),
        }
    )

    # Logowanie wyników (metryk)
    wandb.log(metrics)

    # Zakończenie runu
    wandb.finish()


def log_autogluon_to_wandb(
    metrics: dict[str, Any], raw_data: pd.DataFrame, time_limit: int, presets: str, run_name: str
) -> None:
    """Rejestruje eksperyment AutoGluon w Weights & Biases."""

    # Inicjalizacja runu
    wandb.init(project="online-shoppers", name=run_name, job_type="autogluon_experiment")

    # Logowanie parametrów i metadanych
    wandb.config.update(
        {
            "model_type": "autogluon",
            "time_limit": time_limit,
            "presets": presets,
            "git_sha": get_git_sha(),
            "data_fingerprint": get_data_fingerprint(raw_data),
        }
    )

    # Logowanie wyników (metryk)
    wandb.log(metrics)

    # Zakończenie runu
    wandb.finish()
