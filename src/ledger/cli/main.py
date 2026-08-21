"""LEDGER command-line entry point.

This is the only place in the codebase where a bare ``print`` is allowed
(CLI output to the terminal, as opposed to structured application logs).
Each subcommand group corresponds to a build phase in the project brief;
commands for phases not yet implemented raise ``typer.Exit`` with a clear
message rather than silently doing nothing or fabricating output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ledger.config.settings import load_settings
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
def data_fetch(dataset: str) -> None:
    """Print manual acquisition steps for a dataset (Phase 2 — not yet implemented)."""
    typer.echo(
        f"'ledger data fetch {dataset}' is not implemented yet (Phase 2 of the "
        "project brief). This command must never silently succeed without a "
        "real dataset registry behind it.",
        err=True,
    )
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
