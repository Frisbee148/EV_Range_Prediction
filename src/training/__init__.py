from typing import Any

from src.training.seed import resolve_device, set_all_seeds

__all__ = [
    "TrainResult",
    "Trainer",
    "TrainerConfig",
    "evaluate_predictions_dict",
    "resolve_device",
    "set_all_seeds",
]


def __getattr__(name: str) -> Any:
    if name in {"TrainResult", "Trainer", "TrainerConfig", "evaluate_predictions_dict"}:
        from src.training import trainer

        return getattr(trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
