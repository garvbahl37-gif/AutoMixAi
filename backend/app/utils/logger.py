"""
AutoMixAI – Logging Utility

Provides a pre-configured logger factory so every module gets consistent,
timestamped console output.
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger with the given *name*.

    The logger writes to ``stdout`` with a format that includes the
    timestamp, level, module name, and message.  Calling this function
    multiple times with the same name returns the same logger instance
    (standard ``logging`` behaviour).

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when called more than once
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            fmt="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
