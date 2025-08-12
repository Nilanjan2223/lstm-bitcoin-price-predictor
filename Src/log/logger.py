# log/logger.py

import logging
import os
from datetime import datetime

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance that logs to a file named after the given module.
    """
    log_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(log_dir, exist_ok=True)

    today = datetime.now().strftime('%Y-%m-%d')
    log_file = os.path.join(log_dir, f'{name}_{today}.log')

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent multiple handlers if logger is already set
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, mode='a')
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
