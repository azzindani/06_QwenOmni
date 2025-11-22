"""
Logging configuration for Qwen Omni Voice Assistant.
"""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (usually __name__)
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)

        # Console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        # Format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


if __name__ == "__main__":
    print("=" * 60)
    print("LOGGER TEST")
    print("=" * 60)

    logger = get_logger(__name__)
    logger.info("Test info message")
    logger.warning("Test warning message")
    logger.error("Test error message")

    print("  ✓ Logger working correctly")
    print("=" * 60)
