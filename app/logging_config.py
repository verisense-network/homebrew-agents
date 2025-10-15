"""Centralized logging configuration using rich."""

import logging
import os
from rich.logging import RichHandler
from rich.console import Console


def setup_logging(name: str = None) -> logging.Logger:
    """
    Setup rich logging for a module.

    Args:
        name: Logger name, typically __name__

    Returns:
        Configured logger instance
    """
    # Only configure the root logger once
    if not logging.getLogger().handlers:
        log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        console = Console()

        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(message)s",
            datefmt="[%Y-%m-%d %H:%M:%S]",
            handlers=[
                RichHandler(
                    console=console,
                    rich_tracebacks=True,
                    tracebacks_show_locals=True,
                    show_time=True,
                    show_path=True,
                    enable_link_path=True,
                    markup=True,
                    log_time_format="[%Y-%m-%d %H:%M:%S]",
                )
            ],
        )

    return logging.getLogger(name)
