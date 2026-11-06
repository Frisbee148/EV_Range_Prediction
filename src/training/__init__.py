from typing import Any

from src.training.seed import resolve_device, set_all_seeds

__all__ = [
    "TrainResult",
    "Trainer",
    "TrainerConfig",
    "build_cross_regime_splits",
    "evaluate_predictions_dict",
    "load_experiment_config",
    "resolve_device",
    "run_experiment_suite",
    "run_parameter_sensitivity_analysis",
    "set_all_seeds",
]


def __getattr__(name: str) -> Any:
    if name in {"TrainResult", "Trainer", "TrainerConfig", "evaluate_predictions_dict"}:
        from src.training import trainer

        return getattr(trainer, name)
    if name in {
        "build_cross_regime_splits",
        "load_experiment_config",
        "run_experiment_suite",
        "run_parameter_sensitivity_analysis",
    }:
        from src.training import experiment_runner

        return getattr(experiment_runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
