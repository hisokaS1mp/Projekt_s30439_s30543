from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from typing import Any

import joblib
import pandas as pd

MODEL_TYPES = {"sklearn", "autogluon"}
METADATA_FIELDS = {
    "model_type",
    "features",
    "target",
    "positive_class",
    "positive_class_label",
    "metrics",
    "git_sha",
    "data_fingerprint",
    "library_versions",
}


def _get_library_versions(model_type: str) -> dict[str, str]:
    """Zwraca wersje kluczowych bibliotek użytych w projekcie."""
    libs = ["pandas", "numpy", "scikit-learn", "scipy", "joblib"]
    if model_type == "autogluon":
        libs += ["autogluon.tabular", "catboost", "lightgbm"]
    versions = {"python": sys.version.split()[0]}
    for lib in libs:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _validate_metadata(metadata: Any) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json musi zawierać obiekt JSON.")
    missing = METADATA_FIELDS - metadata.keys()
    if missing:
        raise ValueError(f"Brak pól w metadata.json: {', '.join(sorted(missing))}.")
    model_type = metadata["model_type"]
    if not isinstance(model_type, str) or model_type not in MODEL_TYPES:
        raise ValueError(f"Nieobsługiwany model_type: {model_type!r}.")
    for key in ("target", "positive_class", "git_sha", "data_fingerprint"):
        value = metadata[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Pole {key} w metadata.json musi być niepustym tekstem.")
    if metadata["git_sha"] == "unknown" or metadata["data_fingerprint"] == "unknown":
        raise ValueError("Pakiet wymaga Git SHA i fingerprintu danych źródłowych.")

    features = metadata["features"]
    if (
        not isinstance(features, list)
        or not features
        or not all(isinstance(name, str) and name.strip() for name in features)
        or len(set(features)) != len(features)
        or metadata["target"] in features
    ):
        raise ValueError("Pole features musi zawierać unikalne nazwy cech, bez targetu.")
    # encode_target w data.py koduje klasę pozytywną jako 1.
    label = metadata["positive_class_label"]
    if type(label) is not int or label != 1:
        raise ValueError("Pole positive_class_label musi mieć wartość 1.")
    for key in ("metrics", "library_versions"):
        if not isinstance(metadata[key], dict) or not metadata[key]:
            raise ValueError(f"Pole {key} w metadata.json musi być niepustym obiektem.")


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
    positive_class: str,
    metrics: dict[str, Any],
    git_sha: str,
    data_fingerprint: str,
    overwrite: bool = False,
) -> Path:
    """Zapisuje wytrenowany model. Nadpisanie wymaga jawnego opt-in."""
    metadata = {
        "model_type": model_type,
        "features": list(features),
        "target": target,
        "positive_class": positive_class,
        "positive_class_label": 1,
        "metrics": metrics,
        "git_sha": git_sha,
        "data_fingerprint": data_fingerprint,
        "library_versions": _get_library_versions(model_type),
    }
    _validate_metadata(metadata)
    metadata_json = json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n"

    pkg_path = Path(package_dir)
    source = Path(model.path).resolve() if model_type == "autogluon" else None
    if source is not None:
        destination = pkg_path.resolve()
        if destination.is_relative_to(source) or source.is_relative_to(destination):
            raise ValueError("Ścieżki modelu AutoGluon i pakietu nie mogą się nakładać.")

    pkg_path.parent.mkdir(parents=True, exist_ok=True)
    # Przygotuj cały nowy pakiet przed zmianą istniejącego katalogu.
    with TemporaryDirectory(prefix=".model-package-", dir=pkg_path.parent) as temp_dir:
        staged = Path(temp_dir) / "staged"
        staged.mkdir()
        model_subdir = staged / "model"
        if source is not None:
            shutil.copytree(source, model_subdir)
        else:
            model_subdir.mkdir()
            joblib.dump(model, model_subdir / "model.joblib")
        (staged / "metadata.json").write_text(metadata_json, encoding="utf-8")

        if not pkg_path.exists() and not pkg_path.is_symlink():
            staged.rename(pkg_path)
        elif (
            overwrite
            and not pkg_path.is_symlink()
            and pkg_path.is_dir()
            and {entry.name for entry in pkg_path.iterdir()} == {"model", "metadata.json"}
            and (pkg_path / "model").is_dir()
            and (pkg_path / "metadata.json").is_file()
        ):
            # Kopia poza TemporaryDirectory, aby nie zniknęła, gdy przywrócenie się nie powiedzie.
            backup_dir = Path(mkdtemp(prefix=".model-package-backup-", dir=pkg_path.parent))
            previous = backup_dir / "previous"
            try:
                pkg_path.rename(previous)
                try:
                    staged.rename(pkg_path)
                except OSError:
                    previous.rename(pkg_path)
                    raise
            finally:
                if not previous.exists():
                    backup_dir.rmdir()
            shutil.rmtree(backup_dir)
        else:
            raise FileExistsError(f"Nie można nadpisać katalogu pakietu: {pkg_path}")

    return pkg_path


def load_model_package(package_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    """Wczytuje model z zaufanego pakietu; nie uruchamia treningu."""
    pkg_path = Path(package_dir)
    with open(pkg_path / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    _validate_metadata(metadata)

    model_subdir = pkg_path / "model"
    if metadata["model_type"] == "autogluon":
        from autogluon.tabular import TabularPredictor

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
