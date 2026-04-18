"""Configure loguru: JSON file sink + colored stderr sink."""

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from prefvlm.config import cfg

_configured = False


def setup_logging() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    logger.remove()

    # Human-readable stderr
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
        colorize=True,
    )

    # JSON file log
    cfg.paths.logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = cfg.paths.logs_dir / f"prefvlm_{ts}.jsonl"
    logger.add(
        str(log_path),
        format="{time} | {level} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        serialize=True,
        rotation="100 MB",
    )
    logger.info(f"Logging to {log_path}")
