from kedro.pipeline import Pipeline, node, pipeline

from asi_project.data import filter_training_rows, normalize_input

from .nodes import validate_data


def create_pipeline(**kwargs) -> Pipeline:
    """Buduje pipeline przygotowania danych z logiki ze Sprintu 1."""
    return pipeline(
        [
            node(
                func=validate_data,
                inputs=["raw_data", "contract", "project_config"],
                outputs="validated_data",
                name="validate_data_node",
            ),
            node(
                func=normalize_input,
                inputs="validated_data",
                outputs="normalized_data",
                name="normalize_input_node",
            ),
            node(
                func=filter_training_rows,
                inputs=["normalized_data", "contract"],
                outputs="processed_data",
                name="filter_training_rows_node",
            ),
        ]
    )
