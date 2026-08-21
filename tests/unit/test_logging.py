"""Unit tests for structured logging configuration."""

from __future__ import annotations

import structlog

from ledger.config.models import LoggingConfig
from ledger.utils.logging import configure_logging


def test_configure_logging_json_emits_parseable_output(capsys) -> None:  # type: ignore[no-untyped-def]
    configure_logging(LoggingConfig(level="INFO", json_format=True), run_id="run_abc")
    log = structlog.get_logger("test")
    log.info("something happened", account_id="acc_1")

    captured = capsys.readouterr().out.strip()
    assert captured, "expected a log line to be printed"

    import json

    payload = json.loads(captured)
    assert payload["event"] == "something happened"
    assert payload["account_id"] == "acc_1"
    assert payload["run_id"] == "run_abc"
    assert payload["level"] == "info"


def test_configure_logging_respects_level_filter(capsys) -> None:  # type: ignore[no-untyped-def]
    configure_logging(LoggingConfig(level="WARNING", json_format=True))
    log = structlog.get_logger("test")
    log.info("should be filtered out")
    log.warning("should appear")

    captured = capsys.readouterr().out
    assert "should be filtered out" not in captured
    assert "should appear" in captured


def test_configure_logging_console_mode_does_not_crash(capsys) -> None:  # type: ignore[no-untyped-def]
    configure_logging(LoggingConfig(level="DEBUG", json_format=False))
    log = structlog.get_logger("test")
    log.debug("console mode line")
    captured = capsys.readouterr().out
    assert "console mode line" in captured
