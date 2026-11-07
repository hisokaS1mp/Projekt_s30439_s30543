from kedro.pipeline import Pipeline

from asi_project.pipelines.data_processing import create_pipeline as data_processing
from asi_project.pipelines.data_science import create_pipeline as data_science


def register_pipelines() -> dict[str, Pipeline]:
    """Domyślnie uruchamia przygotowanie danych, trening i ewaluację."""
    processing = data_processing()
    science = data_science()
    return {
        "data_processing": processing,
        "data_science": science,
        "__default__": processing + science,
    }
