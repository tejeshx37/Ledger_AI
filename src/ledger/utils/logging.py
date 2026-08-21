"""Structured logging setup.

Production emits JSON lines (one event per line, machine-parseable by log
aggregation); development emits a human-readable console renderer. Both are
driven off :class:`ledger.config.models.LoggingConfig` — never a hardcoded
branch on an environment name. Every log line is bound with ``run_id`` so
log lines from a given experiment or API request can be correlated back to
its :class:`ledger.utils.manifest.RunManifest`.

No module outside ``src/ledger/cli`` should use a bare ``print``; use
``structlog.get_logger(__name__)`` instead.
"""

from __future__ import annotations

import logging
import sys

import structlog

from ledger.config.models import LoggingConfig


def configure_logging(config: LoggingConfig, run_id: str | None = None) -> None:
    """Configure structlog (and stdlib logging, which structlog wraps).

    Args:
        config: logging configuration (level, json vs. console format).
        run_id: if given, bound onto every subsequent log line via
            structlog's contextvars so it does not need to be passed
            explicitly at every call site.
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if config.json_format
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(config.level)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    structlog.contextvars.clear_contextvars()
    if run_id is not None:
        structlog.contextvars.bind_contextvars(run_id=run_id)


__all__ = ["configure_logging"]
