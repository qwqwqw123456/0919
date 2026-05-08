"""
错误捕获和记录 - Error Catch and Record
统一的异常捕获、记录、上报系统
支持多种错误存储后端（文件、数据库、日志服务等）
提供错误聚合分析、根因分析辅助功能
"""

import os
import sys
import time
import json
import traceback
import threading
import inspect
import hashlib
import smtplib
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Union, Callable, Type
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
from enum import Enum
from functools import wraps
import logging
import linecache


class ErrorLevel(Enum):
    """错误级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    
    def __lt__(self, other):
        order = [ErrorLevel.DEBUG, ErrorLevel.INFO, ErrorLevel.WARNING, 
                 ErrorLevel.ERROR, ErrorLevel.CRITICAL]
        return order.index(self) < order.index(other)


class ErrorCategory(Enum):
    """错误分类"""
    UNKNOWN = "UNKNOWN"
    NETWORK = "NETWORK"
    DATABASE = "DATABASE"
    TIMEOUT = "TIMEOUT"
    PERMISSION = "PERMISSION"
    VALIDATION = "VALIDATION"
    CONFIGURATION = "CONFIGURATION"
    RESOURCE = "RESOURCE"
    EXTERNAL_SERVICE = "EXTERNAL_SERVICE"
    CODE = "CODE"
    
    @classmethod
    def from_exception(cls, exc: Exception) -> 'ErrorCategory':
        """根据异常类型推断分类"""
        exc_type = type(exc).__name__.lower()
        exc_msg = str(exc).lower()
        
        if 'timeout' in exc_type or 'timeout' in exc_msg:
            return cls.TIMEOUT
        elif 'network' in exc_type or 'connection' in exc_type or 'socket' in exc_type:
            return cls.NETWORK
        elif 'database' in exc_type or 'sql' in exc_type or 'query' in exc_type:
            return cls.DATABASE
        elif 'permission' in exc_type or 'access' in exc_type or 'denied' in exc_msg:
            return cls.PERMISSION
        elif 'validate' in exc_type or 'invalid' in exc_msg:
            return cls.VALIDATION
        elif 'config' in exc_type or 'setting' in exc_msg:
            return cls.CONFIGURATION
        elif 'memory' in exc_type or 'disk' in exc_type or 'resource' in exc_type:
            return cls.RESOURCE
        elif 'api' in exc_type or 'http' in exc_type or 'service' in exc_type:
            return cls.EXTERNAL_SERVICE
        else:
            return cls.CODE


@dataclass
class ErrorInfo:
    """错误信息数据类"""
    error_id: str = ""
    level: str = "ERROR"
    category: str = "UNKNOWN"
    message: str = ""
    exception_type: str = ""
    exception_message: str = ""
    traceback: str = ""
    stack_frames: List[Dict[str, Any]] = field(default_factory=list)
    
    module: str = ""
    function: str = ""
    filename: str = ""
    line_number: int = 0
    
    request_id: str = ""
    user_id: str = ""
    session_id: str = ""
    
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    occurred_at: datetime = field(default_factory=datetime.now)
    first_occurrence: datetime = field(default_factory=datetime.now)
    last_occurrence: datetime = field(default_factory=datetime.now)
    
    count: int = 1
    is_resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    tags: List[str] = field(default_factory=list)
    assignee: str = ""
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['occurred_at'] = self.occurred_at.isoformat()
        data['first_occurrence'] = self.first_occurrence.isoformat()
        data['last_occurrence'] = self.last_occurrence.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        return data
    
    @property
    def fingerprint(self) -> str:
        """错误指纹 - 用于聚合相同类型的错误"""
        key = f"{self.module}:{self.function}:{self.exception_type}:{self.line_number}"
        return hashlib.md5(key.encode()).hexdigest()[:16]
    
    @property
    def summary(self) -> str:
        """错误摘要"""
        return f"[{self.exception_type}] {self.message}"


@dataclass
class ErrorStats:
    """错误统计信息"""
    total_errors: int = 0
    errors_by_level: Dict[str, int] = field(default_factory=dict)
    errors_by_category: Dict[str, int] = field(default_factory=dict)
    top_errors: List[Dict[str, Any]] = field(default_factory=list)
    error_trend: List[Dict[str, int]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class ErrorBackend(ABC):
    """错误存储后端抽象基类"""
    
    @abstractmethod
    def save(self, error: ErrorInfo) -> bool:
        """保存错误"""
        pass
    
    @abstractmethod
    def get(self, error_id: str) -> Optional[ErrorInfo]:
        """获取错误"""
        pass
    
    @abstractmethod
    def query(self, filters: Dict[str, Any]) -> List[ErrorInfo]:
        """查询错误"""
        pass
    
    @abstractmethod
    def update(self, error_id: str, updates: Dict[str, Any]) -> bool:
        """更新错误"""
        pass
    
    @abstractmethod
    def delete(self, error_id: str) -> bool:
        """删除错误"""
        pass


class FileBackend(ErrorBackend):
    """文件存储后端"""
    
    def __init__(self, log_dir: str = "./logs/errors"):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self._log_dir / "index.json"
        self._lock = threading.Lock()
        self._index = self._load_index()
    
    def _load_index(self) -> Dict[str, Dict]:
        """加载索引文件"""
        if self._index_file.exists():
            try:
                with open(self._index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {"errors": {}, "fingerprints": defaultdict(list)}
    
    def _save_index(self):
        """保存索引文件"""
        with self._lock:
            with open(self._index_file, 'w', encoding='utf-8') as f:
                json.dump(self._index, f, indent=2, default=str)
    
    def save(self, error: ErrorInfo) -> bool:
        """保存错误到文件"""
        try:
            error_file = self._log_dir / f"{error.error_id}.json"
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(error.to_dict(), f, indent=2, ensure_ascii=False, default=str)
            
            self._index['errors'][error.error_id] = {
                'fingerprint': error.fingerprint,
                'timestamp': error.occurred_at.isoformat(),
                'level': error.level,
                'category': error.category,
                'message': error.message[:200]
            }
            self._index['fingerprints'][error.fingerprint].append(error.error_id)
            self._save_index()
            return True
        except Exception as e:
            logging.error(f"保存错误到文件失败: {e}")
            return False
    
    def get(self, error_id: str) -> Optional[ErrorInfo]:
        """从文件读取错误"""
        error_file = self._log_dir / f"{error_id}.json"
        if not error_file.exists():
            return None
        
        try:
            with open(error_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return self._dict_to_error(data)
        except Exception:
            return None
    
    def query(self, filters: Dict[str, Any]) -> List[ErrorInfo]:
        """查询错误"""
        results = []
        for error_id, meta in self._index.get('errors', {}).items():
            if self._matches_filter(meta, filters):
                error = self.get(error_id)
                if error:
                    results.append(error)
        
        return sorted(results, key=lambda e: e.occurred_at, reverse=True)
    
    def _matches_filter(self, meta: Dict, filters: Dict) -> bool:
        """检查是否匹配过滤器"""
        for key, value in filters.items():
            if key not in meta:
                return False
            if isinstance(value, list):
                if meta[key] not in value:
                    return False
            elif meta[key] != value:
                return False
        return True
    
    def update(self, error_id: str, updates: Dict[str, Any]) -> bool:
        """更新错误"""
        error = self.get(error_id)
        if not error:
            return False
        
        for key, value in updates.items():
            if hasattr(error, key):
                setattr(error, key, value)
        
        return self.save(error)
    
    def delete(self, error_id: str) -> bool:
        """删除错误"""
        error_file = self._log_dir / f"{error_id}.json"
        try:
            if error_file.exists():
                error_file.unlink()
            if error_id in self._index['errors']:
                del self._index['errors'][error_id]
                self._save_index()
            return True
        except Exception:
            return False
    
    def _dict_to_error(self, data: Dict) -> ErrorInfo:
        """字典转ErrorInfo"""
        data = data.copy()
        for dt_field in ['occurred_at', 'first_occurrence', 'last_occurrence', 'resolved_at']:
            if dt_field in data and isinstance(data[dt_field], str):
                data[dt_field] = datetime.fromisoformat(data[dt_field])
        return ErrorInfo(**data)


class MemoryBackend(ErrorBackend):
    """内存存储后端 - 用于测试或临时存储"""
    
    def __init__(self):
        self._errors: Dict[str, ErrorInfo] = {}
        self._fingerprints: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def save(self, error: ErrorInfo) -> bool:
        """保存错误到内存"""
        with self._lock:
            self._errors[error.error_id] = error
            self._fingerprints[error.fingerprint].append(error.error_id)
            return True
    
    def get(self, error_id: str) -> Optional[ErrorInfo]:
        """从内存获取错误"""
        return self._errors.get(error_id)
    
    def query(self, filters: Dict[str, Any]) -> List[ErrorInfo]:
        """查询错误"""
        results = []
        for error in self._errors.values():
            if self._matches_filter(error, filters):
                results.append(error)
        return sorted(results, key=lambda e: e.occurred_at, reverse=True)
    
    def _matches_filter(self, error: ErrorInfo, filters: Dict) -> bool:
        """检查是否匹配过滤器"""
        for key, value in filters.items():
            error_value = getattr(error, key, None)
            if error_value is None:
                return False
            if isinstance(value, list):
                if error_value not in value:
                    return False
            elif error_value != value:
                return False
        return True
    
    def update(self, error_id: str, updates: Dict[str, Any]) -> bool:
        """更新错误"""
        with self._lock:
            if error_id not in self._errors:
                return False
            for key, value in updates.items():
                if hasattr(self._errors[error_id], key):
                    setattr(self._errors[error_id], key, value)
            return True
    
    def delete(self, error_id: str) -> bool:
        """删除错误"""
        with self._lock:
            if error_id in self._errors:
                del self._errors[error_id]
                return True
            return False
    
    def get_all(self) -> List[ErrorInfo]:
        """获取所有错误"""
        return list(self._errors.values())


class ErrorCatcher:
    """
    错误捕获和记录器
    
    功能特性：
    - 自动捕获未处理异常
    - 手动记录错误
    - 错误聚合去重
    - 多种存储后端支持
    - 错误统计分析
    """
    
    _instance: Optional['ErrorCatcher'] = None
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
        self._config = config or {}
        
        self._backend_type = self._config.get('backend', 'memory')
        self._init_backend()
        
        self._callbacks: List[Callable] = []
        self._rate_limiter: Dict[str, List[datetime]] = defaultdict(list)
        self._rate_limit_window = self._config.get('rate_limit_window', 60)
        self._rate_limit_max = self._config.get('rate_limit_max', 100)
        
        self._logger = logging.getLogger(__name__)
        
        self._install_excepthook()
    
    def _init_backend(self):
        """初始化存储后端"""
        if self._backend_type == 'file':
            log_dir = self._config.get('log_dir', './logs/errors')
            self._backend = FileBackend(log_dir)
        else:
            self._backend = MemoryBackend()
    
    def _install_excepthook(self):
        """安装全局异常钩子"""
        self._old_excepthook = sys.excepthook
        sys.excepthook = self._excepthook
    
    def _excepthook(self, exc_type, exc_value, exc_traceback):
        """全局异常处理"""
        if issubclass(exc_type, KeyboardInterrupt):
            self._old_excepthook(exc_type, exc_value, exc_traceback)
            return
        
        error = self._create_error_from_exception(exc_type, exc_value, exc_traceback)
        self.save(error)
        
        for callback in self._callbacks:
            try:
                callback(error)
            except Exception as e:
                self._logger.error(f"错误回调执行失败: {e}")
        
        self._old_excepthook(exc_type, exc_value, exc_traceback)
    
    def _create_error_from_exception(self, exc_type: Type[Exception], 
                                    exc_value: Exception,
                                    exc_traceback) -> ErrorInfo:
        """从异常创建ErrorInfo"""
        tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        
        stack_frames = []
        for frame_summary in traceback.extract_tb(exc_traceback):
            stack_frames.append({
                'filename': frame_summary.filename,
                'lineno': frame_summary.lineno,
                'function': frame_summary.name,
                'code': linecache.getline(frame_summary.filename, frame_summary.lineno).strip()
            })
        
        first_frame = stack_frames[0] if stack_frames else {}
        
        error_id = self._generate_error_id()
        
        return ErrorInfo(
            error_id=error_id,
            level=ErrorLevel.ERROR.value,
            category=ErrorCategory.from_exception(exc_value).value,
            message=str(exc_value),
            exception_type=exc_type.__name__,
            exception_message=str(exc_value),
            traceback=tb_str,
            stack_frames=stack_frames,
            module=first_frame.get('filename', ''),
            function=first_frame.get('function', ''),
            filename=first_frame.get('filename', ''),
            line_number=first_frame.get('lineno', 0),
            occurred_at=datetime.now(),
            first_occurrence=datetime.now(),
            last_occurrence=datetime.now()
        )
    
    def _generate_error_id(self) -> str:
        """生成错误ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        return f"ERR_{timestamp}_{threading.get_ident() % 10000:04d}"
    
    def _check_rate_limit(self, fingerprint: str) -> bool:
        """检查速率限制"""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self._rate_limit_window)
        
        self._rate_limiter[fingerprint] = [
            t for t in self._rate_limiter[fingerprint] if t > cutoff
        ]
        
        if len(self._rate_limiter[fingerprint]) >= self._rate_limit_max:
            return False
        
        self._rate_limiter[fingerprint].append(now)
        return True
    
    def log_error(self, error: Union[ErrorInfo, Exception],
                  message: str = "",
                  level: str = "ERROR",
                  context: Optional[Dict[str, Any]] = None,
                  **kwargs) -> Optional[ErrorInfo]:
        """
        记录错误
        
        Args:
            error: ErrorInfo对象或Exception
            message: 错误消息
            level: 错误级别
            context: 额外上下文
            **kwargs: 其他ErrorInfo字段
        
        Returns:
            ErrorInfo对象或None（如果被速率限制）
        """
        if isinstance(error, Exception):
            error_info = self._create_error_from_exception(
                type(error), error, error.__traceback__
            )
            if message:
                error_info.message = message
            error_info.level = level
        else:
            error_info = error
            if message:
                error_info.message = message
            error_info.level = level
        
        if context:
            error_info.context.update(context)
        
        for key, value in kwargs.items():
            if hasattr(error_info, key):
                setattr(error_info, key, value)
        
        self._update_or_create(error_info)
        
        if self._check_rate_limit(error_info.fingerprint):
            self._backend.save(error_info)
        
        for callback in self._callbacks:
            try:
                callback(error_info)
            except Exception as e:
                self._logger.error(f"错误回调执行失败: {e}")
        
        return error_info
    
    def _update_or_create(self, error_info: ErrorInfo):
        """更新或创建错误记录"""
        existing = self._backend.query({
            'fingerprint': error_info.fingerprint
        })
        
        if existing:
            old_error = existing[0]
            error_info.error_id = old_error.error_id
            error_info.count = old_error.count + 1
            error_info.first_occurrence = old_error.first_occurrence
            error_info.last_occurrence = datetime.now()
    
    def catch(self, func: Callable = None,
              level: str = "ERROR",
              reraise: bool = True,
              context_func: Optional[Callable] = None):
        """
        错误捕获装饰器
        
        Args:
            func: 要装饰的函数
            level: 错误级别
            reraise: 是否重新抛出异常
            context_func: 上下文函数，返回字典
        
        Usage:
            @ErrorCatcher().catch()
            def my_function():
                ...
        """
        def decorator(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    context = {}
                    if context_func:
                        try:
                            context = context_func(*args, **kwargs)
                        except Exception:
                            pass
                    
                    self.log_error(
                        e,
                        level=level,
                        context=context,
                        metadata={'function': f.__name__, 'args': str(args)[:200]}
                    )
                    
                    if reraise:
                        raise
                    return None
            return wrapper
        
        if func is None:
            return decorator
        return decorator(func)
    
    def save(self, error: ErrorInfo) -> bool:
        """保存错误"""
        return self._backend.save(error)
    
    def get(self, error_id: str) -> Optional[ErrorInfo]:
        """获取错误"""
        return self._backend.get(error_id)
    
    def query(self, filters: Optional[Dict[str, Any]] = None,
              limit: int = 100,
              offset: int = 0) -> List[ErrorInfo]:
        """
        查询错误
        
        Args:
            filters: 查询过滤器
            limit: 返回数量限制
            offset: 偏移量
        
        Returns:
            错误列表
        """
        filters = filters or {}
        results = self._backend.query(filters)
        return results[offset:offset + limit]
    
    def get_by_fingerprint(self, fingerprint: str) -> List[ErrorInfo]:
        """根据指纹获取所有相同错误"""
        return self._backend.query({'fingerprint': fingerprint})
    
    def update(self, error_id: str, updates: Dict[str, Any]) -> bool:
        """更新错误"""
        return self._backend.update(error_id, updates)
    
    def resolve(self, error_id: str) -> bool:
        """标记错误为已解决"""
        return self.update(error_id, {
            'is_resolved': True,
            'resolved_at': datetime.now()
        })
    
    def delete(self, error_id: str) -> bool:
        """删除错误"""
        return self._backend.delete(error_id)
    
    def register_callback(self, callback: Callable):
        """注册错误回调"""
        if callback not in self._callbacks:
            self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable):
        """取消注册回调"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def get_stats(self, time_range: Optional[int] = None) -> ErrorStats:
        """
        获取错误统计
        
        Args:
            time_range: 时间范围（秒）
        
        Returns:
            错误统计信息
        """
        filters = {}
        if time_range:
            cutoff = datetime.now() - timedelta(seconds=time_range)
            filters['occurred_at'] = {'$gte': cutoff}
        
        errors = self._backend.query(filters)
        
        by_level = Counter(e.level for e in errors)
        by_category = Counter(e.category for e in errors)
        
        fingerprint_counts = Counter(e.fingerprint for e in errors)
        top_errors = [
            {'fingerprint': fp, 'count': count}
            for fp, count in fingerprint_counts.most_common(10)
        ]
        
        trend = self._calculate_trend(errors)
        
        return ErrorStats(
            total_errors=len(errors),
            errors_by_level=dict(by_level),
            errors_by_category=dict(by_category),
            top_errors=top_errors,
            error_trend=trend,
            timestamp=datetime.now()
        )
    
    def _calculate_trend(self, errors: List[ErrorInfo]) -> List[Dict[str, int]]:
        """计算错误趋势"""
        if not errors:
            return []
        
        hours = 24
        trend = []
        now = datetime.now()
        
        for i in range(hours, 0, -1):
            hour_start = now - timedelta(hours=i)
            hour_end = hour_start + timedelta(hours=1)
            count = sum(1 for e in errors if hour_start <= e.occurred_at < hour_end)
            trend.append({
                'hour': hour_start.strftime('%Y-%m-%d %H:00'),
                'count': count
            })
        
        return trend
    
    def get_unresolved(self, limit: int = 100) -> List[ErrorInfo]:
        """获取未解决的错误"""
        return self.query({'is_resolved': False}, limit=limit)
    
    def get_recent(self, limit: int = 100) -> List[ErrorInfo]:
        """获取最近的错误"""
        return self.query(limit=limit)
    
    def export_errors(self, filepath: str, filters: Optional[Dict[str, Any]] = None):
        """导出错误到文件"""
        errors = self.query(filters or {}, limit=10000)
        data = {
            'export_time': datetime.now().isoformat(),
            'total': len(errors),
            'errors': [e.to_dict() for e in errors]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        self._logger.info(f"已导出 {len(errors)} 条错误到 {filepath}")
    
    def clear_old_errors(self, days: int = 30) -> int:
        """清理旧错误"""
        cutoff = datetime.now() - timedelta(days=days)
        all_errors = self._backend.query({})
        count = 0
        
        for error in all_errors:
            if error.occurred_at < cutoff and error.is_resolved:
                if self._backend.delete(error.error_id):
                    count += 1
        
        self._logger.info(f"已清理 {count} 条旧错误")
        return count
    
    def close(self):
        """关闭并恢复原始excepthook"""
        sys.excepthook = self._old_excepthook
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_catcher(config: Optional[Dict[str, Any]] = None) -> ErrorCatcher:
    """
    获取错误捕获器单例
    
    Args:
        config: 配置字典
    
    Returns:
        ErrorCatcher实例
    """
    return ErrorCatcher(config)


def catch_errors(level: str = "ERROR", reraise: bool = True):
    """
    错误捕获装饰器工厂函数
    
    Args:
        level: 错误级别
        reraise: 是否重新抛出异常
    
    Usage:
        @catch_errors()
        def my_function():
            ...
    """
    catcher = ErrorCatcher()
    return catcher.catch(level=level, reraise=reraise)


class ErrorContext:
    """错误上下文管理器"""
    
    def __init__(self, error_catcher: Optional[ErrorCatcher] = None,
                 context: Optional[Dict[str, Any]] = None):
        self._catcher = error_catcher or ErrorCatcher()
        self._context = context or {}
        self._error: Optional[ErrorInfo] = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self._error = self._catcher.log_error(
                exc_val,
                context=self._context
            )
        return False
    
    @property
    def error(self) -> Optional[ErrorInfo]:
        """获取捕获的错误"""
        return self._error


__all__ = [
    'ErrorLevel',
    'ErrorCategory',
    'ErrorInfo',
    'ErrorStats',
    'ErrorBackend',
    'FileBackend',
    'MemoryBackend',
    'ErrorCatcher',
    'ErrorContext',
    'get_catcher',
    'catch_errors'
]
