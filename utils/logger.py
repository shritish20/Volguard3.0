"""
Logging configuration
"""

import logging
from datetime import datetime
from pathlib import Path

from config.settings import PATH_CONFIG


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup logger with file and console handlers"""
    
    # Create logs directory
    PATH_CONFIG.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # File handler
    log_file = PATH_CONFIG.LOGS_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create logger"""
    return setup_logger(name)
