from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    """Buduje pipeline modelowania na danych po data_processing."""
    return pipeline(
        [
            node(
                func=split_data,
                inputs=[
                    "processed_data",
                    "contract",
                    "project_config",
                    "params:split.test_size",
                    "params:split.random_state",
                ],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "project_config"],
                outputs="model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["model", "X_test", "y_test"],
                outputs="metrics",
                name="evaluate_model_node",
            ),
        ]
    )
