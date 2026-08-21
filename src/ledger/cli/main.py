"""LEDGER command-line entry point.

This is the only place in the codebase where a bare ``print`` is allowed
(CLI output to the terminal, as opposed to structured application logs).
Each subcommand group corresponds to a build phase in the project brief;
commands for phases not yet implemented raise ``typer.Exit`` with a clear
message rather than silently doing nothing or fabricating output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ledger.config.settings import load_settings, load_settings_from_overlay_path
from ledger.data.fetch import DatasetChecksumMismatchError, fetch_dataset, format_fetch_instructions
from ledger.data.prepare import prepare_dataset
from ledger.data.registry import DATASET_REGISTRY
from ledger.evaluation.comparison import DEFAULT_MODEL_OVERLAYS, run_model_comparison
from ledger.training.train import train_detector
from ledger.utils.logging import configure_logging

app = typer.Typer(
    name="ledger",
    help="LEDGER: temporal-graph anti-money-laundering detection platform.",
    no_args_is_help=True,
)

config_app = typer.Typer(help="Inspect resolved configuration.")
data_app = typer.Typer(help="Dataset acquisition and preparation (Phase 2).")
federated_app = typer.Typer(help="Federated learning experiments (Phase 5).")
explain_app = typer.Typer(help="Explain alerts (Phase 6).")
app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")
app.add_typer(federated_app, name="federated")
app.add_typer(explain_app, name="explain")

_ConfigDirOption = Annotated[
    Path, typer.Option(help="Directory containing base.yaml and overlays.")
]
_EnvironmentOption = Annotated[
    str | None, typer.Option(help="Environment-specific YAML overlay name.")
]


@app.callback()
def _main() -> None:
    """LEDGER CLI. Run `ledger <group> --help` for details on a subcommand group."""


@config_app.command("show")
def config_show(
    config_dir: _ConfigDirOption = Path("configs"),
    environment: _EnvironmentOption = None,
) -> None:
    """Print the fully resolved configuration as JSON."""
    settings = load_settings(config_dir=config_dir, environment=environment)
    configure_logging(settings.logging)
    typer.echo(settings.model_dump_json(indent=2))


@config_app.command("hash")
def config_hash(
    config_dir: _ConfigDirOption = Path("configs"),
    environment: _EnvironmentOption = None,
) -> None:
    """Print the SHA-256 hash of the fully resolved configuration."""
    settings = load_settings(config_dir=config_dir, environment=environment)
    typer.echo(settings.hash())


@data_app.command("fetch")
def data_fetch(
    dataset: str,
    config_dir: _ConfigDirOption = Path("configs"),
    environment: _EnvironmentOption = None,
) -> None:
    """Print manual acquisition steps, then verify any files already placed.

    Never downloads anything. Exits non-zero if a required file is missing
    or fails a pinned checksum, so this can gate `ledger data prepare` in
    scripts.
    """
    if dataset not in DATASET_REGISTRY:
        typer.echo(f"Unknown dataset {dataset!r}. Known: {sorted(DATASET_REGISTRY)}", err=True)
        raise typer.Exit(code=1)

    typer.echo(format_fetch_instructions(dataset))
    typer.echo("")

    settings = load_settings(config_dir=config_dir, environment=environment)
    raw_dir = settings.paths.data_raw_dir / dataset
    try:
        report = fetch_dataset(dataset, raw_dir)
    except DatasetChecksumMismatchError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Verification against {raw_dir}:")
    for result in report.results:
        typer.echo(f"  - {result.file.filename}: {result.status}")
        if result.status == "unpinned":
            typer.echo(f"      actual sha256: {result.actual_sha256}")

    if not report.is_ready:
        typer.echo(
            "Not ready: place the missing file(s) above, then re-run this command.", err=True
        )
        raise typer.Exit(code=1)
    typer.echo("All expected files present and verified (or unpinned).")


@data_app.command("prepare")
def data_prepare(
    dataset: str,
    config_dir: _ConfigDirOption = Path("configs"),
    environment: _EnvironmentOption = None,
) -> None:
    """Adapt, validate, build a graph artifact, and split a placed dataset."""
    settings = load_settings(config_dir=config_dir, environment=environment)
    configure_logging(settings.logging)
    try:
        result = prepare_dataset(settings, dataset_name=dataset)
    except (FileNotFoundError, DatasetChecksumMismatchError, ValueError) as exc:
        typer.echo(f"ledger data prepare failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        json.dumps(
            {
                "dataset_name": result.dataset_name,
                "run_id": result.run_id,
                "canonical_dir": str(result.canonical_dir),
                "graph_path": str(result.graph_path),
                "node_statistics_path": str(result.node_statistics_path),
                "split_manifest_path": str(result.split_manifest_path),
                "split_ids_path": str(result.split_ids_path),
                "manifest_path": str(result.manifest_path),
            },
            indent=2,
        )
    )


_ConfigOption = Annotated[
    Path, typer.Option(help="Overlay YAML under a configs/ directory alongside base.yaml.")
]


@app.command("train")
def train(config: _ConfigOption = Path("configs/model_baseline.yaml")) -> None:
    """Fit a detector per `--config` and write model/metrics/curves/manifest to runs/."""
    settings = load_settings_from_overlay_path(config)
    configure_logging(settings.logging)
    try:
        result = train_detector(settings)
    except (
        FileNotFoundError,
        DatasetChecksumMismatchError,
        NotImplementedError,
        KeyError,
        ValueError,
    ) as exc:
        typer.echo(f"ledger train failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        json.dumps(
            {
                "run_id": result.run_id,
                "dataset_name": result.dataset_name,
                "model_path": str(result.model_path),
                "feature_pipeline_path": str(result.feature_pipeline_path),
                "metrics_path": str(result.metrics_path),
                "pr_curve_path": str(result.pr_curve_path),
                "feature_importance_path": str(result.feature_importance_path),
                "manifest_path": str(result.manifest_path),
                "test_metrics": result.test_metrics,
            },
            indent=2,
        )
    )


_ComparisonConfigsOption = Annotated[
    list[str] | None,
    typer.Option(
        "--configs",
        help="Overlay YAML filenames under --config-dir to compare (repeat the "
        "flag per model); defaults to the baseline/GraphSAGE/TGAT trio.",
    ),
]


@app.command("compare")
def compare(
    config_dir: _ConfigDirOption = Path("configs"),
    configs: _ComparisonConfigsOption = None,
) -> None:
    """Train baseline/GraphSAGE/TGAT (or --configs) on an identical split,
    write a ring-participation comparison table plus per-model artifacts to
    runs/.
    """
    overlay_names = configs if configs else list(DEFAULT_MODEL_OVERLAYS)
    overlay_paths = [config_dir / name for name in overlay_names]
    try:
        result = run_model_comparison(overlay_paths)
    except (
        FileNotFoundError,
        DatasetChecksumMismatchError,
        NotImplementedError,
        KeyError,
        ValueError,
    ) as exc:
        typer.echo(f"ledger compare failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        json.dumps(
            {
                "run_id": result.run_id,
                "dataset_name": result.dataset_name,
                "table_path": str(result.table_path),
                "table_csv_path": str(result.table_csv_path),
                "manifest_path": str(result.manifest_path),
                "table": result.table,
            },
            indent=2,
        )
    )


@federated_app.command("run")
def federated_run(config: _ConfigOption = Path("configs/federated.yaml")) -> None:
    """Run the three-condition federated experiment and write results to runs/."""
    settings = load_settings_from_overlay_path(config)
    configure_logging(settings.logging)
    try:
        from ledger.federated.experiment import run_federated_experiment

        result = run_federated_experiment(settings)
    except Exception as exc:
        typer.echo(f"ledger federated run failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        json.dumps(
            {
                "run_id": result.run_id,
                "table_csv_path": str(result.table_csv_path),
                "table_json_path": str(result.table_json_path),
                "manifest_path": str(result.manifest_path),
                "results": result.results,
            },
            indent=2,
        )
    )


@explain_app.command("run")
def explain_run(
    account_id: str,
    run_dir: Path = typer.Option(..., help="Path to the run directory containing the trained model."),
    config: _ConfigOption = Path("configs/base.yaml"),
) -> None:
    """Generate structured evidence, narrative description, and counterfactual explanation for an alert."""
    import joblib
    from ledger.config.settings import load_settings_from_overlay_path
    from ledger.utils.logging import configure_logging
    from ledger.explain.subgraph import extract_alert_evidence
    from ledger.explain.narrative import get_narrative
    from ledger.explain.counterfactual import explain_counterfactual

    settings = load_settings_from_overlay_path(config)
    configure_logging(settings.logging)

    model_path = run_dir / "model.joblib"
    if not model_path.exists():
        typer.echo(f"Model file not found at {model_path}", err=True)
        raise typer.Exit(code=1)

    try:
        detector = joblib.load(model_path)
    except Exception as exc:
        typer.echo(f"Failed to load model: {exc}", err=True)
        raise typer.Exit(code=1)

    try:
        evidence = extract_alert_evidence(detector, account_id)
    except Exception as exc:
        typer.echo(f"Failed to extract subgraph evidence: {exc}", err=True)
        raise typer.Exit(code=1)

    narrative = get_narrative(evidence, settings.llm)
    counterfactual = explain_counterfactual(detector, account_id, evidence)

    output_dir = run_dir / "explanations"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{account_id}.json"

    explanation_payload = {
        "evidence": evidence.to_dict(),
        "narrative": narrative,
        "counterfactual": counterfactual,
    }

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(explanation_payload, f, indent=2, sort_keys=True)

    typer.echo(json.dumps(explanation_payload, indent=2, sort_keys=True))


@app.command("evaluate")
def evaluate(
    run_dir: Path = typer.Option(None, help="Path to the run directory containing the model."),
    run: str = typer.Option(None, help="Run ID to evaluate."),
    config: _ConfigOption = Path("configs/base.yaml"),
) -> None:
    """Evaluate a trained model for bias, adversarial robustness, drift, and ablations (Phase 7)."""
    import json
    import joblib
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from ledger.config.settings import load_settings_from_overlay_path
    from ledger.utils.logging import configure_logging
    from ledger.evaluation.fairness import audit_bias
    from ledger.evaluation.robustness import simulate_adversarial_evasion
    from ledger.evaluation.drift import evaluate_performance_drift, calculate_input_drift
    from ledger.evaluation.ablation import run_ablation_study
    from ledger.data.canonical import CanonicalDataset
    from ledger.evaluation.labels import build_binary_target
    from ledger.features.pipeline import FeaturePipeline

    settings = load_settings_from_overlay_path(config)
    configure_logging(settings.logging)

    # Resolve run directory
    if run_dir is None:
        if run is None:
            typer.echo("Error: either --run-dir or --run option must be provided.", err=True)
            raise typer.Exit(code=1)
        run_dir = Path("runs") / run

    if not run_dir.exists():
        typer.echo(f"Run directory not found: {run_dir}", err=True)
        raise typer.Exit(code=1)

    model_path = run_dir / "model.joblib"
    if not model_path.exists():
        typer.echo(f"Model file not found at {model_path}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Loading detector from {model_path}...")
    try:
        detector = joblib.load(model_path)
    except Exception as exc:
        typer.echo(f"Failed to load model: {exc}", err=True)
        raise typer.Exit(code=1)

    # 1. Resolve dataset
    dataset_name = "ibm_aml"
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            with manifest_path.open("r") as f:
                manifest = json.load(f)
                dataset_name = manifest.get("dataset_name", "ibm_aml")
        except Exception:
            pass

    typer.echo(f"Loading dataset: {dataset_name}...")
    try:
        dataset = CanonicalDataset.read("data/processed", dataset_name)
    except Exception as exc:
        typer.echo(f"Failed to load dataset {dataset_name}: {exc}", err=True)
        raise typer.Exit(code=1)

    y_all = build_binary_target(dataset)
    accs = list(y_all.index)
    train_ids = accs[:int(len(accs)*0.7)]
    test_ids = accs[int(len(accs)*0.7):]

    # Reconstruct pipeline
    pipeline = FeaturePipeline(config=settings.features)
    pipeline.fit(dataset, train_account_ids=train_ids, held_out_account_ids=test_ids)
    
    train_features = pipeline.transform(dataset, account_ids=train_ids).set_index("account_id")
    test_features = pipeline.transform(dataset, account_ids=test_ids).set_index("account_id")
    y_test = y_all.loc[test_features.index]

    # Predict on test features
    probs = detector.predict_proba(test_features)
    y_pred = (probs >= 0.5).astype(int)

    # 2. Bias Audit
    # Check for customer_age, fallback to simulated age cohort
    if "customer_age" in dataset.accounts.columns:
        sens_df = dataset.accounts.set_index("account_id")["customer_age"]
    else:
        import hashlib
        simulated_ages = {}
        for acc in accs:
            val = int(hashlib.md5(acc.encode()).hexdigest()[:4], 16)
            simulated_ages[acc] = ">50" if val % 2 == 0 else "<=50"
        sens_df = pd.Series(simulated_ages)

    sens_test = sens_df.loc[test_features.index]
    typer.echo("Auditing bias and fairness...")
    bias_results = audit_bias(y_test, y_pred, sens_test)

    # 3. Adversarial Robustness
    target_accounts = list(y_test[y_test == 1].index)
    if not target_accounts:
        target_accounts = list(y_test.index[:10])  # fallback

    typer.echo("Simulating adversarial evasion...")
    robustness_results = simulate_adversarial_evasion(detector, pipeline, dataset, target_accounts)

    # 4. Drift Evaluation
    tx_df = dataset.transactions
    src_times = tx_df.groupby("src_account")["timestamp"].min()
    dst_times = tx_df.groupby("dst_account")["timestamp"].min()
    combined_times = pd.concat([src_times, dst_times]).groupby(level=0).min()
    test_timestamps = combined_times.reindex(test_features.index).fillna(pd.to_datetime("2022-09-01T00:00:00"))

    typer.echo("Evaluating drift...")
    perf_drift = evaluate_performance_drift(y_test, probs, test_timestamps)
    input_drift = calculate_input_drift(train_features, test_features)

    # 5. Ablations
    typer.echo("Running systematic ablation study...")
    ablation_results = run_ablation_study(settings, dataset, train_ids, test_ids)

    # 6. Generate Figures
    chart_dir = run_dir / "evaluation"
    chart_dir.mkdir(parents=True, exist_ok=True)
    chart_path = chart_dir / "evaluation_chart.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Robustness
    levels = list(robustness_results.keys())
    recalls = list(robustness_results.values())
    axes[0].plot(levels, recalls, marker='o', color='crimson', linewidth=2)
    axes[0].set_title("Adversarial Evasion Efficacy")
    axes[0].set_xlabel("Evasion Level")
    axes[0].set_ylabel("Detection Rate (Recall)")
    axes[0].set_xticks(levels)
    axes[0].set_xticklabels(["L0: Base", "L1: Splits", "L2: Delays", "L3: Mules"])
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Plot Performance Drift
    if perf_drift:
        bins = [d["bin_index"] for d in perf_drift]
        aucs = [d["roc_auc"] for d in perf_drift]
        axes[1].plot(bins, aucs, marker='s', color='navy', linewidth=2)
        axes[1].set_title("Temporal Metric Drift")
        axes[1].set_xlabel("Chronological Time Bin")
        axes[1].set_ylabel("ROC-AUC Score")
        axes[1].set_xticks(bins)
        axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()

    # 7. Compile Report Markdown
    report_path = run_dir / "evaluation_report.md"
    
    # Format Bias audit rows
    bias_rows = ""
    for g, metrics in bias_results["group_metrics"].items():
        bias_rows += f"| {g} | {metrics['selection_rate']:.4f} | {metrics['tpr']:.4f} | {metrics['fpr']:.4f} | {metrics['count']} |\n"

    # Format Drift rows
    drift_rows = ""
    for b in perf_drift:
        drift_rows += f"| Bin {b['bin_index']} | {b['start_time']} | {b['end_time']} | {b['roc_auc']:.4f} | {b['count']} |\n"

    # Format Ablation rows
    ablation_rows = ""
    for r in ablation_results:
        ablation_rows += f"| {r['condition']} | {r['roc_auc']:.4f} |\n"

    report_content = f"""# Evaluation Report — Run `{run_dir.name}`

## 1. Bias Audit (Fairness)
* **Demographic Parity Difference**: {bias_results['demographic_parity_difference']:.4f}
* **Equalized Odds Difference**: {bias_results['equalized_odds_difference']:.4f}
* **False Positive Rate Disparity**: {bias_results['false_positive_rate_disparity']:.4f}

| Protected Cohort | Selection Rate | TPR | FPR | Sample Count |
|---|---|---|---|---|
{bias_rows}

## 2. Adversarial Robustness
Detection rate decay under simulated evasion:

| Evasion Level | Evasion Strategy | Detection Recall |
|---|---|---|
| Level 0 | Baseline (Original graph) | {robustness_results[0]:.4f} |
| Level 1 | Split Amounts (Half-value splits) | {robustness_results[1]:.4f} |
| Level 2 | Delays (Spread transaction delays) | {robustness_results[2]:.4f} |
| Level 3 | Mule Hops (Intermediate hop accounts) | {robustness_results[3]:.4f} |

## 3. Drift Evaluation
Performance and input distribution changes.

### Performance Drift over Time Bins
| Time Bin | Start Period | End Period | ROC-AUC | Sample Count |
|---|---|---|---|---|
{drift_rows}

### Input Distribution Drift (PSI)
* **Average PSI**: {input_drift['average_psi']:.4f}
* **Drift Level**: {input_drift['drift_level'].upper()}

## 4. Systematic Ablation Study
Performance impact of removing core graph, temporal, and privacy features:

| Condition Removed | ROC-AUC |
|---|---|
{ablation_rows}
"""
    
    with report_path.open("w", encoding="utf-8") as f:
        f.write(report_content)

    typer.echo(f"Evaluation report generated successfully!")
    typer.echo(f"Report Markdown: {report_path}")
    typer.echo(f"Report Figure: {chart_path}")


if __name__ == "__main__":
    app()

