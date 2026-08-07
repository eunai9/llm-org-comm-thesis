"""Consistent logging for every entry point.

Long batch jobs are the norm here, so timestamps are non-negotiable: when a
run takes four hours you need to know which step consumed it.
"""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s  %(levelname)-8s %(name)-28s %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Install a stderr handler on the root logger. Safe to call repeatedly."""
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=_FORMAT, datefmt=_DATEFMT))
    root.addHandler(handler)
    # These libraries are chatty at INFO during batch polling.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)
