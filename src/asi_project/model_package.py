from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from autogluon.tabular import TabularPredictor

from asi_project.tracking import get_data_fingerprint, get_git_sha


def _get_library_versions() -> dict[str, str]:
    """Zwraca wersje kluczowych bibliotek użytych w projekcie."""
    libs = ["python", "pandas", "numpy", "scikit-learn", "autogluon.tabular", "kedro"]
    versions: dict[str, str] = {}

    versions["python"] = sys.version.split()[0]

    for lib in libs[1:]:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            versions[lib] = "not-installed"

    return versions


def get_dir_size_mb(path: Path) -> float:
    """Oblicza łączny rozmiar katalogu (lub pliku) na dysku w MB."""
    if path.is_file():
        return round(path.stat().st_size / (1024 * 1024), 4)
    total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total_bytes / (1024 * 1024), 4)


def _get_process_ram_mb() -> float:
    """Zwraca aktualne zużycie pamięci RAM procesu (RSS) w MB."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        import resource

        # Fallback dla systemów Unix/Linux)
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def save_model_package(
    model: Any,
    package_dir: str | Path,
    *,
    model_type: str,
    features: list[str],
    target: str,
    positive_class: Any,
    metrics: dict[str, Any],
    fingerprint_df: pd.DataFrame | None = None,
    git_sha: str | None = None,
    data_fingerprint: str | None = None,
) -> Path:
    """Zapisuje model i plik metadata.json zgodnie z kontraktem pakietu."""
    pkg_path = Path(package_dir)
    if pkg_path.exists():
        shutil.rmtree(pkg_path)
    pkg_path.mkdir(parents=True, exist_ok=True)

    model_subdir = pkg_path / "model"

    if model_type == "autogluon":
        shutil.copytree(model.path, model_subdir)
    else:
        model_subdir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_subdir / "model.joblib")

    resolved_git_sha = git_sha if git_sha is not None else get_git_sha()
    if data_fingerprint is not None:
        resolved_fingerprint = data_fingerprint
    elif fingerprint_df is not None:
        resolved_fingerprint = get_data_fingerprint(fingerprint_df)
    else:
        resolved_fingerprint = "unknown"

    metadata = {
        "model_type": model_type,
        "features": list(features),
        "target": target,
        "positive_class": positive_class,
        "metrics": metrics,
        "git_sha": resolved_git_sha,
        "data_fingerprint": resolved_fingerprint,
        "library_versions": _get_library_versions(),
    }

    with open(pkg_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return pkg_path


def load_model_package(package_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    """Wczytuje model oraz jego metadata.json z katalogu pakietu."""
    pkg_path = Path(package_dir)
    with open(pkg_path / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    model_subdir = pkg_path / "model"
    if metadata["model_type"] == "autogluon":
        model = TabularPredictor.load(str(model_subdir))
    else:
        model = joblib.load(model_subdir / "model.joblib")

    return model, metadata


def benchmark_model_package(package_dir: str | Path, X_test: pd.DataFrame) -> dict[str, float]:
    """Mierzy rozmiar na dysku, przyrost RAM po wczytaniu i czas predykcji."""
    pkg_path = Path(package_dir)
    size_mb = get_dir_size_mb(pkg_path)

    ram_before = _get_process_ram_mb()
    model, _ = load_model_package(pkg_path)
    ram_after = _get_process_ram_mb()

    start_time = time.perf_counter()
    _ = model.predict(X_test)
    elapsed_sec = time.perf_counter() - start_time

    return {
        "artifact_size_mb": size_mb,
        "ram_loaded_mb": round(max(ram_after - ram_before, 0.0), 2),
        "ram_total_mb": round(ram_after, 2),
        "predict_time_sec": round(elapsed_sec, 4),
        "predict_time_per_row_ms": round((elapsed_sec / len(X_test)) * 1000, 4),
    }
