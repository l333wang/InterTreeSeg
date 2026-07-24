"""File + console logging (migrated from the legacy ``log2file``)."""

from __future__ import annotations

import logging
import os
from datetime import datetime

_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def log2file(log_dir: str = "log", sub_dir: str | None = None) -> logging.Logger:
    """Configure and return the root logger writing to console + ``<log_dir>/<sub_dir>/logfile.log``.

    Guards against attaching duplicate handlers if called more than once with
    the same target (the legacy version re-added handlers on every call).
    """
    if sub_dir is None:
        sub_dir = datetime.now().strftime("%Y-%m-%d_%H-%M")
    full_log_dir = os.path.join(log_dir, sub_dir)
    os.makedirs(full_log_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_filename = os.path.abspath(os.path.join(full_log_dir, "logfile.log"))

    # Avoid duplicate handlers pointing at the same file / a console stream.
    existing_files = {
        os.path.abspath(h.baseFilename)
        for h in logger.handlers
        if isinstance(h, logging.FileHandler)
    }
    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )

    if not has_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(console_handler)

    if log_filename not in existing_files:
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(file_handler)

    return logger
