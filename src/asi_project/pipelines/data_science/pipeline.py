from kedro.pipeline import Pipeline, node, pipeline

from asi_project.tracking import log_autogluon_to_wandb, log_baseline_to_wandb

from .nodes import (
    benchmark_candidates,
    evaluate_model,
    export_production_package,
    split_data,
    train_autogluon_model,
    train_model,
)


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
            node(
                func=log_baseline_to_wandb,
                inputs=[
                    "metrics",
                    "raw_data",
                    "params:split.test_size",
                    "params:split.random_state",
                ],
                outputs=None,
                name="log_baseline_wandb_node",
            ),
            # --- Konfiguracja 1: AutoGluon Fast ---
            node(
                func=train_autogluon_model,
                inputs=[
                    "X_train",
                    "y_train",
                    "params:target_col",
                    "params:ag_fast.time_limit",
                    "params:ag_fast.presets",
                ],
                outputs="ag_fast_model",
                name="train_ag_fast_node",
            ),
            node(
                func=evaluate_model,
                inputs=["ag_fast_model", "X_test", "y_test"],
                outputs="ag_fast_metrics",
                name="evaluate_ag_fast_node",
            ),
            node(
                func=lambda m, d, t, p: log_autogluon_to_wandb(m, d, t, p, "autogluon-fast"),
                inputs=[
                    "ag_fast_metrics",
                    "raw_data",
                    "params:ag_fast.time_limit",
                    "params:ag_fast.presets",
                ],
                outputs=None,
                name="log_ag_fast_wandb_node",
            ),
            # --- Konfiguracja 2: AutoGluon High Quality ---
            node(
                func=train_autogluon_model,
                inputs=[
                    "X_train",
                    "y_train",
                    "params:target_col",
                    "params:ag_hq.time_limit",
                    "params:ag_hq.presets",
                ],
                outputs="ag_hq_model",
                name="train_ag_hq_node",
            ),
            node(
                func=evaluate_model,
                inputs=["ag_hq_model", "X_test", "y_test"],
                outputs="ag_hq_metrics",
                name="evaluate_ag_hq_node",
            ),
            node(
                func=lambda m, d, t, p: log_autogluon_to_wandb(m, d, t, p, "autogluon-hq"),
                inputs=[
                    "ag_hq_metrics",
                    "raw_data",
                    "params:ag_hq.time_limit",
                    "params:ag_hq.presets",
                ],
                outputs=None,
                name="log_ag_hq_wandb_node",
            ),
            node(
                func=benchmark_candidates,
                inputs=[
                    "X_test",
                    "raw_data",
                    "contract",
                    "model",
                    "metrics",
                    "ag_fast_model",
                    "ag_fast_metrics",
                    "ag_hq_model",
                    "ag_hq_metrics",
                ],
                outputs="model_comparison",
                name="benchmark_candidates_node",
            ),
            node(
                func=export_production_package,
                inputs=[
                    "ag_hq_model",
                    "ag_hq_metrics",
                    "X_test",
                    "raw_data",
                    "contract",
                ],
                outputs=None,
                name="export_production_package_node",
            ),
        ]
    )
