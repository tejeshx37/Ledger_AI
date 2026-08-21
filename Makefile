.PHONY: install lint typecheck test data baseline train federated api frontend up down

PYTHON := python3
PIP := $(PYTHON) -m pip

install:
	$(PIP) install -e ".[dev,ml,api]"
	pre-commit install

lint:
	ruff check src tests
	black --check src tests

typecheck:
	mypy src

test:
	pytest

data:
	ledger data fetch $(DATASET)

baseline:
	ledger train --config configs/model_baseline.yaml

train:
	ledger train --config $(CONFIG)

federated:
	ledger federated run --config configs/federated.yaml

api:
	.venv/bin/uvicorn ledger.api.main:app --host 0.0.0.0 --port 8000 --reload

worker:
	.venv/bin/python -m ledger.api.worker

frontend:
	cd frontend && npm run dev

up:
	docker compose up --build

down:
	docker compose down
