"""
Logging configuration utilities for the trading system.
Provides consistent logging setup across all modules.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json


class ColoredFormatter(logging.Formatter):
    """
    Colored log formatter for console output.
    Makes logs easier to read with color coding by level.
    """
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to the level name
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        
        # Format the message
        formatted = super().format(record)
        
        # Add emoji indicators for different levels
        if record.levelname.find('ERROR') != -1:
            formatted = f"❌ {formatted}"
        elif record.levelname.find('WARNING') != -1:
            formatted = f"⚠️  {formatted}"
        elif record.levelname.find('INFO') != -1 and 'complete' in record.getMessage().lower():
            formatted = f"✅ {formatted}"
        elif record.levelname.find('DEBUG') != -1:
            formatted = f"🔍 {formatted}"
            
        return formatted


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging (useful for log aggregation)."""
    
    def format(self, record):
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
            
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'msecs', 'levelname', 
                          'levelno', 'pathname', 'filename', 'module', 'exc_info',
                          'exc_text', 'stack_info', 'lineno', 'funcName', 'processName',
                          'process', 'threadName', 'thread', 'getMessage']:
                log_obj[key] = value
                
        return json.dumps(log_obj)


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[Path] = None,
    json_logging: bool = False,
    component: str = 'trading_system'
) -> logging.Logger:
    """
    Setup logging configuration for a component.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        json_logging: Use JSON format for logs
        component: Component name for the logger
        
    Returns:
        Configured logger instance
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Get or create logger
    logger = logging.getLogger(component)
    logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if json_logging:
        console_formatter = JsonFormatter()
    else:
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        
        # Always use detailed format for file logs
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Suppress noisy third-party loggers
    suppress_loggers = [
        'urllib3',
        'urllib3.connectionpool',
        'yfinance',
        'matplotlib',
        'h5py',
        'alpaca',
        'asyncio',
        'peewee',
        'boto3',
        'botocore',
        'websockets',
        'httpx',
        'httpcore',
    ]
    
    for logger_name in suppress_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    return logger


def setup_training_logger(
    level: str = 'INFO',
    log_dir: Optional[Path] = None
) -> logging.Logger:
    """
    Setup logging specifically for the training pipeline.
    
    Args:
        level: Logging level
        log_dir: Directory for log files (defaults to logs/training/)
        
    Returns:
        Configured logger for training
    """
    if log_dir is None:
        log_dir = Path('logs/training')
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'training_{timestamp}.log'
    
    return setup_logging(
        level=level,
        log_file=log_file,
        component='training_pipeline'
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the current configuration.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LogContext:
    """Context manager for temporary log level changes."""
    
    def __init__(self, logger: logging.Logger, level: str):
        self.logger = logger
        self.new_level = getattr(logging, level.upper())
        self.old_level = logger.level
        
    def __enter__(self):
        self.logger.setLevel(self.new_level)
        return self.logger
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)


def log_performance(func):
    """Decorator to log function performance metrics."""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {e}")
            raise
            
    return wrapper


def log_async_performance(func):
    """Decorator to log async function performance metrics."""
    import time
    import functools
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {e}")
            raise
            
    return wrapper