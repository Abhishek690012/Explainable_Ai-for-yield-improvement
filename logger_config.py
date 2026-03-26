import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Sets up and returns a logger that outputs to both a file ('pipeline.log')
    and the console.

    Args:
        name: Name of the logger.

    Returns:
        Configured logging.Logger.
    """
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler('pipeline.log', mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
