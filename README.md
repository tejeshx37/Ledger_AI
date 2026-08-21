# LEDGER

A temporal-graph anti-money-laundering (AML) detection platform. Money
laundering's only structurally visible stage — layering — leaves a network
signature, not a transaction-level one, and that signature is often spread
across banks that legally cannot pool data to see it. LEDGER models
transactions as a temporal graph, detects laundering as a structural pattern,
learns that pattern collaboratively across institutions via federated
learning (no raw customer data leaves a bank), and explains every alert in
language sufficient to support a Suspicious Activity Report.

See the full design and phase-by-phase build plan in the project brief.
This repository is built incrementally, phase by phase; each phase's
deliverable must run and its tests must pass before the next begins.

## Status

- **Phase 1 — Foundations: done.** Typed config system (Pydantic v2,
  layered YAML + env + CLI overrides, `Settings.hash()`), reproducibility
  utilities (`seed_everything`, `RunManifest`), structured logging
  (structlog, JSON or console), dev tooling (`ruff`, `black`, `mypy
  --strict`, `pytest` with an 80% coverage gate, pre-commit, `Makefile`).
- Phases 2-10 (data layer, baseline model, graph models, federated
  learning, explanation layer, evaluation/fairness/robustness,
  persistence/API, frontend, packaging/operations) are not yet built. See
  the project brief for scope; **Phase 5 (federated learning) is the
  project's core contribution and must not be cut.**

## Engineering standards

1. No hardcoded values anywhere in `src/` — every threshold, path,
   hyperparameter, and credential is a named, validated field in
   `src/ledger/config/`, sourced from YAML + environment variables.
2. No fabricated results — a missing dataset or untrained model fails
   loudly; nothing simulates a result to look like it worked.
3. No mocked business logic in application code paths (tests may use
   synthetic fixtures).
4. Deterministic and reproducible — every run seeds Python/NumPy/PyTorch
   and writes a `RunManifest` (git SHA, config hash, dataset checksums,
   seed, library versions) to `runs/<run_id>/manifest.json`.
5. Fully typed — Pydantic v2 everywhere, `mypy --strict` passes.
6. Tested — pytest, ≥80% coverage on `src/`.
7. Observable — structured JSON logging, Prometheus metrics on the API.
8. Documented — every module explains what it does and why.

## Getting started

```bash
python3 -m venv .venv && source .venv/bin/activate
make install
make lint typecheck test
```

Inspect the resolved configuration:

```bash
ledger config show
ledger config hash
```

Copy `.env.example` to `.env` and fill in real values before running
anything that touches the database, Redis, or an LLM provider. No default
in `configs/*.yaml` contains a secret.

## License

Apache-2.0. See `LICENSE`.
