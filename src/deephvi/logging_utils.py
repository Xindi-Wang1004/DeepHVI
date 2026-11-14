"""Logging helpers.

The package never replaces ``sys.stdout``/``sys.stderr`` at import time. A
logger named ``deephvi`` is configured once per entry point and optionally
writes to a file inside the run directory.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOGGER_NAME = "deephvi"


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> logging.Logger:
    """Configure and return the package logger."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # Remove handlers created by previous calls in the same process.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """Return the package logger without reconfiguring it."""
    return logging.getLogger(LOGGER_NAME)
