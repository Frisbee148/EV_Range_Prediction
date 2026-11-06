"""Model factory and matched-budget verification for Models A, B, and C (PRD §15)."""

from __future__ import annotations

from dataclasses import dataclass

from src.data.sequence_builder import DYNAMIC_FEATURES, STATIC_FEATURES
from src.models.lstm import EVTrajectoryLSTM


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    name: str
    use_physics: bool
    use_attention: bool


MODEL_SPECS: dict[str, ModelSpec] = {
    "A": ModelSpec(
        model_id="A",
        name="Data-only LSTM",
        use_physics=False,
        use_attention=False,
    ),
    "B": ModelSpec(
        model_id="B",
        name="Physics-informed LSTM",
        use_physics=True,
        use_attention=False,
    ),
    "C": ModelSpec(
        model_id="C",
        name="Physics-informed LSTM + Attention",
        use_physics=True,
        use_attention=True,
    ),
}


def build_model(
    model_id: str = "B",
    *,
    f_dyn: int = len(DYNAMIC_FEATURES),
    f_stat: int = len(STATIC_FEATURES),
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
    use_attention_override: bool | None = None,
) -> EVTrajectoryLSTM:
    """Instantiate Model A, B, or C with identical backbone architecture (PRD §15)."""
    key = model_id.upper()
    if key not in MODEL_SPECS:
        raise ValueError(f"Unknown model_id={model_id!r}; expected one of {sorted(MODEL_SPECS)}")
    spec = MODEL_SPECS[key]
    use_attn = spec.use_attention if use_attention_override is None else bool(use_attention_override)
    return EVTrajectoryLSTM(
        f_dyn=f_dyn,
        f_stat=f_stat,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_attention=use_attn,
    )


def verify_matched_architecture(model_a: EVTrajectoryLSTM, model_b: EVTrajectoryLSTM) -> None:
    """Verify Model A and Model B share identical architecture and parameter counts (PRD §15)."""
    params_a = [(n, tuple(p.shape)) for n, p in model_a.named_parameters()]
    params_b = [(n, tuple(p.shape)) for n, p in model_b.named_parameters()]
    if params_a != params_b:
        raise AssertionError(
            f"Matched-budget violation: Model A and Model B architectures differ: {params_a} vs {params_b}"
        )
