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
from ledger.training.train import train_detector
from ledger.utils.logging import configure_logging

app = typer.Typer(
    name="ledger",
    help="LEDGER: temporal-graph anti-money-laundering detection platform.",
    no_args_is_help=True,
)

config_app = typer.Typer(help="Inspect resolved configuration.")
data_app = typer.Typer(help="Dataset acquisition and preparation (Phase 2).")
app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")

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


if __name__ == "__main__":
    app()
