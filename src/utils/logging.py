# src/utils/logging.py

import logging
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config import LOG_DIR, LOG_LEVEL

LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"

# requests exceptions embed the full URL, query string included (FMP: ?apikey=,
# FRED: ?api_key=). Scrub any such value from every log line and traceback.
_SECRET_PARAM = re.compile(r"(?i)\b(apikey|api_key|token|access_key)=[^&\s'\")]+")


def redact_secrets(text: str) -> str:
    return _SECRET_PARAM.sub(r"\1=***", text)


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact_secrets(super().format(record))


def get_logger(name: str, log_dir: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if get_logger is called multiple times
    if not logger.handlers:
        level = getattr(logging, LOG_LEVEL, logging.INFO)
        logger.setLevel(level)

        formatter = RedactingFormatter(LOG_FORMAT)

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Handler
        # Use provided log_dir or fallback to the central log directory
        final_log_dir = Path(log_dir) if log_dir else LOG_DIR
        final_log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            final_log_dir / f"{name}.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.propagate = False

    return logger


def get_log_file(logger: logging.Logger) -> str | None:
    """Return the absolute path of a logger's file handler, or None if it has none."""
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return handler.baseFilename
    return None
