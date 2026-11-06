PYTHON := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

.PHONY: test data physics train smoke-train

test:
	$(PYTHON) -m pytest tests -q

physics:
	$(PYTHON) scripts/validate_physics_tum.py

data:
	$(PYTHON) scripts/build_sumo_dataset.py

smoke-train:
	$(PYTHON) -m src.training.trainer --model A --epochs 5 --warmup-epochs 2 --soc-k 10 --ablation-fraction 0.1 --run-id smoke_A
	$(PYTHON) -m src.training.trainer --model B --epochs 5 --warmup-epochs 2 --soc-k 10 --ablation-fraction 0.1 --run-id smoke_B

train:
	$(PYTHON) -m src.training.trainer --config configs/training.yaml
