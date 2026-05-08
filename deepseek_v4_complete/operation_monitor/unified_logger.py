"""
统一日志系统 - Unified Logger
提供集中式的日志记录功能，支持多级别日志、文件输出、控制台输出、日志轮转等
"""

import logging
import os
import sys
import json
import traceback
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from functools import wraps
import threading
import gzip
import shutil


class LogLevel:
    """日志级别定义"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    
    @staticmethod
    def get_name(level: int) -> str:
        """获取级别名称"""
        level_names = {
            logging.DEBUG: "DEBUG",
            logging.INFO: "INFO",
            logging.WARNING: "WARNING",
            logging.ERROR: "ERROR",
            logging.CRITICAL: "CRITICAL"
        }
        return level_names.get(level, "UNKNOWN")


class JsonFormatter(logging.Formatter):
    """JSON格式日志格式化器"""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """将日志记录格式化为JSON字符串"""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": record.process,
            "thread_id": record.thread,
            "thread_name": record.threadName
        }
        
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        if self.include_extra:
            extra_fields = {k: v for k, v in record.__dict__.items() 
                          if k not in logging.LogRecord(
                              "dummy", logging.DEBUG, "", 0, "", (), None).__dict__}
            if extra_fields:
                log_data["extra"] = extra_fields
        
        return json.dumps(log_data, ensure_ascii=False, default=str)


class ColoredFormatter(logging.Formatter):
    """彩色控制台格式化器"""
    
    COLOR_CODES = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
        'RESET': '\033[0m'
    }
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, use_color: bool = True):
        super().__init__(fmt, datefmt)
        self.use_color = use_color and sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        if self.use_color:
            color = self.COLOR_CODES.get(record.levelname, self.COLOR_CODES['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLOR_CODES['RESET']}"
            record.msg = f"{color}{record.msg}{self.COLOR_CODES['RESET']}" if isinstance(record.msg, str) else record.msg
        
        return super().format(record)


class StructuredLogger:
    """结构化日志记录器 - 支持键值对格式的日志"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._context: Dict[str, Any] = {}
        self._lock = threading.Lock()
    
    def set_context(self, **kwargs):
        """设置日志上下文"""
        with self._lock:
            self._context.update(kwargs)
    
    def clear_context(self):
        """清除日志上下文"""
        with self._lock:
            self._context.clear()
    
    def _build_message(self, message: str, **kwargs) -> str:
        """构建带键值对的消息"""
        parts = [message]
        if kwargs or self._context:
            context = {**self._context, **kwargs}
            context_str = " ".join([f"{k}={v}" for k, v in context.items()])
            parts.append(f"[{context_str}]")
        return " ".join(parts)
    
    def debug(self, message: str, **kwargs):
        """记录DEBUG级别日志"""
        self.logger.debug(self._build_message(message, **kwargs))
    
    def info(self, message: str, **kwargs):
        """记录INFO级别日志"""
        self.logger.info(self._build_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """记录WARNING级别日志"""
        self.logger.warning(self._build_message(message, **kwargs))
    
    def error(self, message: str, **kwargs):
        """记录ERROR级别日志"""
        self.logger.error(self._build_message(message, **kwargs))
    
    def critical(self, message: str, **kwargs):
        """记录CRITICAL级别日志"""
        self.logger.critical(self._build_message(message, **kwargs))


class LogManager:
    """日志管理器 - 统一管理所有日志配置"""
    
    _instance: Optional['LogManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self._loggers: Dict[str, logging.Logger] = {}
        self._config = config or self._default_config()
        self._log_dir = Path(self._config.get('log_dir', './logs'))
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._rotation_handler: Optional[Union[RotatingFileHandler, TimedRotatingFileHandler]] = None
    
    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            'log_dir': './logs',
            'level': logging.DEBUG,
            'console': True,
            'file': True,
            'json_format': False,
            'max_bytes': 100 * 1024 * 1024,
            'backup_count': 10,
            'when': 'midnight',
            'interval': 1
        }
    
    def get_logger(self, name: str = __name__, level: Optional[int] = None) -> logging.Logger:
        """获取日志记录器"""
        if name in self._loggers:
            return self._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(level or self._config.get('level', logging.DEBUG))
        logger.handlers.clear()
        
        if self._config.get('console', True):
            self._add_console_handler(logger)
        
        if self._config.get('file', True):
            self._add_file_handlers(logger)
        
        self._loggers[name] = logger
        return logger
    
    def get_structured_logger(self, name: str = __name__) -> StructuredLogger:
        """获取结构化日志记录器"""
        logger = self.get_logger(name)
        return StructuredLogger(logger)
    
    def _add_console_handler(self, logger: logging.Logger):
        """添加控制台处理器"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self._config.get('level', logging.DEBUG))
        
        if self._config.get('json_format', False):
            formatter = JsonFormatter()
        else:
            fmt = self._config.get('console_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            datefmt = self._config.get('date_format', '%Y-%m-%d %H:%M:%S')
            formatter = ColoredFormatter(fmt, datefmt)
        
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    def _add_file_handlers(self, logger: logging.Logger):
        """添加文件处理器"""
        rotation_type = self._config.get('rotation_type', 'size')
        
        if rotation_type == 'time':
            self._rotation_handler = TimedRotatingFileHandler(
                self._log_dir / f"{logger.name}.log",
                when=self._config.get('when', 'midnight'),
                interval=self._config.get('interval', 1),
                backupCount=self._config.get('backup_count', 10),
                encoding='utf-8'
            )
        else:
            self._rotation_handler = RotatingFileHandler(
                self._log_dir / f"{logger.name}.log",
                maxBytes=self._config.get('max_bytes', 100 * 1024 * 1024),
                backupCount=self._config.get('backup_count', 10),
                encoding='utf-8'
            )
        
        if self._config.get('json_format', False):
            formatter = JsonFormatter()
        else:
            fmt = self._config.get('file_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            datefmt = self._config.get('date_format', '%Y-%m-%d %H:%M:%S')
            formatter = logging.Formatter(fmt, datefmt)
        
        self._rotation_handler.setFormatter(formatter)
        self._rotation_handler.setLevel(self._config.get('level', logging.DEBUG))
        logger.addHandler(self._rotation_handler)
    
    def set_level(self, name: str, level: int):
        """设置日志级别"""
        if name in self._loggers:
            self._loggers[name].setLevel(level)
    
    def add_file_handler(self, name: str, filepath: str, level: Optional[int] = None):
        """为指定日志器添加额外的文件处理器"""
        if name not in self._loggers:
            self.get_logger(name)
        
        file_handler = logging.FileHandler(filepath, encoding='utf-8')
        file_handler.setLevel(level or self._config.get('level', logging.DEBUG))
        
        fmt = self._config.get('file_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        datefmt = self._config.get('date_format', '%Y-%m-%d %H:%M:%S')
        formatter = logging.Formatter(fmt, datefmt)
        file_handler.setFormatter(formatter)
        
        self._loggers[name].addHandler(file_handler)
    
    def cleanup_old_logs(self, days: int = 30):
        """清理旧的日志文件"""
        cutoff = datetime.now() - timedelta(days=days)
        for log_file in self._log_dir.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff.timestamp():
                try:
                    log_file.unlink()
                except OSError:
                    pass
    
    def compress_old_logs(self):
        """压缩旧的日志文件"""
        for log_file in self._log_dir.glob("*.log"):
            if log_file.stat().st_size > 1024 * 1024:
                gzip_path = Path(str(log_file) + '.gz')
                try:
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(gzip_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    log_file.unlink()
                except Exception:
                    pass


def get_logger(name: str = __name__, level: Optional[int] = None) -> logging.Logger:
    """
    获取日志记录器的便捷函数
    
    Args:
        name: 日志记录器名称，默认为模块名
        level: 日志级别，默认使用配置中的级别
    
    Returns:
        logging.Logger实例
    """
    manager = LogManager()
    return manager.get_logger(name, level)


def get_structured_logger(name: str = __name__) -> StructuredLogger:
    """
    获取结构化日志记录器的便捷函数
    
    Args:
        name: 日志记录器名称，默认为模块名
    
    Returns:
        StructuredLogger实例
    """
    manager = LogManager()
    return manager.get_structured_logger(name)


def configure_logger(config: Dict[str, Any]):
    """
    配置日志管理器
    
    Args:
        config: 配置字典
    """
    LogManager(config)


def log_decorator(func):
    """
    函数日志装饰器 - 自动记录函数调用
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"调用函数: {func.__name__}, 参数: args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"函数 {func.__name__} 返回: {result}")
            return result
        except Exception as e:
            logger.error(f"函数 {func.__name__} 执行出错: {e}", exc_info=True)
            raise
    return wrapper


class ContextLogger:
    """
    上下文日志记录器 - 支持上下文信息的日志记录
    """
    
    _local = threading.local()
    
    @classmethod
    def set_context(cls, **kwargs):
        """设置线程局部上下文"""
        if not hasattr(cls._local, 'context'):
            cls._local.context = {}
        cls._local.context.update(kwargs)
    
    @classmethod
    def get_context(cls) -> Dict[str, Any]:
        """获取当前线程的上下文"""
        return getattr(cls._local, 'context', {}).copy()
    
    @classmethod
    def clear_context(cls):
        """清除当前线程的上下文"""
        cls._local.context = {}
    
    @classmethod
    def log(cls, level: int, message: str, **kwargs):
        """使用当前上下文记录日志"""
        context = cls.get_context()
        context.update(kwargs)
        logger = get_logger('context')
        logger.log(level, f"{message} [{' '.join([f'{k}={v}' for k, v in context.items()])}]")
    
    @classmethod
    def debug(cls, message: str, **kwargs):
        cls.log(logging.DEBUG, message, **kwargs)
    
    @classmethod
    def info(cls, message: str, **kwargs):
        cls.log(logging.INFO, message, **kwargs)
    
    @classmethod
    def warning(cls, message: str, **kwargs):
        cls.log(logging.WARNING, message, **kwargs)
    
    @classmethod
    def error(cls, message: str, **kwargs):
        cls.log(logging.ERROR, message, **kwargs)


def exception_logger(logger: Optional[logging.Logger] = None):
    """
    异常日志装饰器 - 捕获并记录异常
    
    Args:
        logger: 可选的日志记录器，默认创建新的
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"函数 {func.__name__} 发生异常",
                    exc_info=True,
                    extra={
                        'function': func.__name__,
                        'args': str(args)[:200],
                        'kwargs': str(kwargs)[:200],
                        'exception_type': type(e).__name__,
                        'exception_message': str(e)
                    }
                )
                raise
        return wrapper
    return decorator


class PerformanceLogger:
    """性能日志记录器 - 记录函数执行时间"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger('performance')
        self._timings: Dict[str, list] = {}
    
    def timeit(self, func):
        """计时装饰器"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = datetime.now()
            try:
                result = func(*args, **kwargs)
                elapsed = (datetime.now() - start).total_seconds()
                self.logger.info(f"{func.__name__} 执行耗时: {elapsed:.3f}s")
                if func.__name__ not in self._timings:
                    self._timings[func.__name__] = []
                self._timings[func.__name__].append(elapsed)
                return result
            except Exception as e:
                elapsed = (datetime.now() - start).total_seconds()
                self.logger.error(f"{func.__name__} 执行失败，耗时: {elapsed:.3f}s, 错误: {e}")
                raise
        return wrapper
    
    def get_stats(self, func_name: str) -> Dict[str, float]:
        """获取函数执行统计"""
        if func_name not in self._timings:
            return {}
        timings = self._timings[func_name]
        return {
            'count': len(timings),
            'total': sum(timings),
            'avg': sum(timings) / len(timings),
            'min': min(timings),
            'max': max(timings)
        }


__all__ = [
    'LogManager',
    'LogLevel',
    'JsonFormatter',
    'ColoredFormatter',
    'StructuredLogger',
    'ContextLogger',
    'PerformanceLogger',
    'get_logger',
    'get_structured_logger',
    'configure_logger',
    'log_decorator',
    'exception_logger'
]
