"""Logging setup. Never log API keys, full documents, or student answers."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    """Configure the root logger once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    if level is None:
        try:
            from config.settings import get_settings

            level = get_settings().log_level
        except Exception:  # configuration must never break logging
            level = "INFO"

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
