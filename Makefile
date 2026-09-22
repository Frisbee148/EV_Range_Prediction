.PHONY: test data physics

test:
	python3 -m pytest tests -q

physics:
	python3 scripts/validate_physics_tum.py

data:
	python3 scripts/build_sumo_dataset.py
