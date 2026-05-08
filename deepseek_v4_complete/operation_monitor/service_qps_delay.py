"""
服务 QPS/延迟监控 - Service QPS and Delay Monitor
监控服务的请求量(QPS)、响应延迟、错误率等指标
支持多服务、多端点的监控，提供实时统计和历史分析
"""

import os
import time
import threading
import json
import bisect
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from functools import wraps
import logging


@dataclass
class RequestRecord:
    """请求记录"""
    endpoint: str
    method: str = "GET"
    latency: float = 0.0
    status_code: int = 200
    timestamp: datetime = field(default_factory=datetime.now)
    request_id: str = ""
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'endpoint': self.endpoint,
            'method': self.method,
            'latency': self.latency,
            'status_code': self.status_code,
            'timestamp': self.timestamp.isoformat(),
            'request_id': self.request_id,
            'error': self.error,
            'metadata': self.metadata
        }
    
    @property
    def is_success(self) -> bool:
        """是否成功请求"""
        return 200 <= self.status_code < 400 and not self.error
    
    @property
    def is_error(self) -> bool:
        """是否错误请求"""
        return not self.is_success


@dataclass
class EndpointStats:
    """端点统计信息"""
    endpoint: str
    total_requests: int = 0
    success_requests: int = 0
    error_requests: int = 0
    total_latency: float = 0.0
    min_latency: float = float('inf')
    max_latency: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    qps: float = 0.0
    error_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'endpoint': self.endpoint,
            'total_requests': self.total_requests,
            'success_requests': self.success_requests,
            'error_requests': self.error_requests,
            'total_latency': self.total_latency,
            'min_latency': self.min_latency if self.min_latency != float('inf') else 0,
            'max_latency': self.max_latency,
            'latency_p50': self.latency_p50,
            'latency_p95': self.latency_p95,
            'latency_p99': self.latency_p99,
            'qps': self.qps,
            'error_rate': self.error_rate,
            'timestamp': self.timestamp.isoformat()
        }
    
    @property
    def avg_latency(self) -> float:
        """平均延迟"""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency / self.total_requests
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_requests == 0:
            return 0.0
        return (self.success_requests / self.total_requests) * 100


@dataclass 
class TimeWindowStats:
    """时间窗口统计"""
    window_start: datetime
    window_end: datetime
    total_requests: int = 0
    success_requests: int = 0
    error_requests: int = 0
    qps: float = 0.0
    avg_latency: float = 0.0
    min_latency: float = 0.0
    max_latency: float = 0.0
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    error_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'window_start': self.window_start.isoformat(),
            'window_end': self.window_end.isoformat(),
            'total_requests': self.total_requests,
            'success_requests': self.success_requests,
            'error_requests': self.error_requests,
            'qps': self.qps,
            'avg_latency': self.avg_latency,
            'min_latency': self.min_latency,
            'max_latency': self.max_latency,
            'p50_latency': self.p50_latency,
            'p95_latency': self.p95_latency,
            'p99_latency': self.p99_latency,
            'error_rate': self.error_rate
        }


class CircularBuffer:
    """循环缓冲区 - 用于存储固定大小的历史数据"""
    
    def __init__(self, max_size: int):
        self._max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()
    
    def append(self, item: Any):
        """添加元素"""
        with self._lock:
            self._buffer.append(item)
    
    def get_all(self) -> List[Any]:
        """获取所有元素"""
        with self._lock:
            return list(self._buffer)
    
    def get_recent(self, n: int) -> List[Any]:
        """获取最近的n个元素"""
        with self._lock:
            return list(self._buffer)[-n:]
    
    def clear(self):
        """清空缓冲区"""
        with self._lock:
            self._buffer.clear()
    
    def __len__(self) -> int:
        return len(self._buffer)


class PercentileCalculator:
    """百分位数计算器"""
    
    def __init__(self, values: List[float]):
        self._values = sorted(values)
    
    def percentile(self, p: float) -> float:
        """计算百分位数"""
        if not self._values:
            return 0.0
        index = (p / 100.0) * (len(self._values) - 1)
        lower = int(index)
        upper = lower + 1
        if upper >= len(self._values):
            return self._values[-1]
        weight = index - lower
        return self._values[lower] * (1 - weight) + self._values[upper] * weight


class ServiceQPSDelayMonitor:
    """
    服务QPS和延迟监控器
    
    功能特性：
    - 记录每个请求的延迟、状态码、错误信息
    - 实时计算QPS、平均延迟、百分位延迟
    - 支持多端点分组统计
    - 历史数据查询
    - 阈值告警
    """
    
    _instance: Optional['ServiceQPSDelayMonitor'] = None
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
        
        self._records: Dict[str, CircularBuffer] = defaultdict(
            lambda: CircularBuffer(self._config.get('max_records', 10000))
        )
        self._all_records = CircularBuffer(self._config.get('max_records', 10000))
        
        self._endpoint_stats: Dict[str, EndpointStats] = {}
        self._window_stats: List[TimeWindowStats] = []
        
        self._request_counts: Dict[str, List[tuple]] = defaultdict(list)
        self._last_qps_calc: Dict[str, datetime] = {}
        
        self._callbacks: Dict[str, List[Callable]] = {
            'alert': [],
            'stats': []
        }
        
        self._alert_thresholds = self._config.get('alert_thresholds', {
            'qps_min': 1.0,
            'qps_max': 10000.0,
            'latency_p95': 1000.0,
            'error_rate': 5.0
        })
        
        self._window_size = self._config.get('window_size', 60)
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitoring = False
        self._logger = logging.getLogger(__name__)
    
    def record(self, endpoint: str, latency: float, status_code: int = 200,
               method: str = "GET", error: str = "", 
               request_id: str = "", metadata: Optional[Dict[str, Any]] = None):
        """
        记录一次请求
        
        Args:
            endpoint: 端点路径
            latency: 延迟时间（毫秒）
            status_code: HTTP状态码
            method: 请求方法
            error: 错误信息
            request_id: 请求ID
            metadata: 额外元数据
        """
        record = RequestRecord(
            endpoint=endpoint,
            method=method,
            latency=latency,
            status_code=status_code,
            timestamp=datetime.now(),
            request_id=request_id,
            error=error,
            metadata=metadata or {}
        )
        
        self._records[endpoint].append(record)
        self._all_records.append(record)
        
        self._request_counts[endpoint].append((datetime.now(), record))
        
        self._check_alerts(endpoint, record)
    
    def record_success(self, endpoint: str, latency: float, 
                       method: str = "GET", request_id: str = "", 
                       metadata: Optional[Dict[str, Any]] = None):
        """记录成功请求"""
        self.record(endpoint, latency, 200, method, "", request_id, metadata)
    
    def record_error(self, endpoint: str, latency: float, 
                     status_code: int = 500, error: str = "",
                     method: str = "GET", request_id: str = "",
                     metadata: Optional[Dict[str, Any]] = None):
        """记录错误请求"""
        self.record(endpoint, latency, status_code, method, error, request_id, metadata)
    
    def get_endpoint_stats(self, endpoint: str, time_range: Optional[int] = None) -> EndpointStats:
        """
        获取端点统计信息
        
        Args:
            endpoint: 端点路径
            time_range: 时间范围（秒），None表示全部历史
        
        Returns:
            端点统计信息
        """
        records = self._records.get(endpoint, CircularBuffer(1)).get_all()
        
        if time_range:
            cutoff = datetime.now() - timedelta(seconds=time_range)
            records = [r for r in records if r.timestamp >= cutoff]
        
        if not records:
            return EndpointStats(endpoint=endpoint)
        
        latencies = [r.latency for r in records]
        success_count = sum(1 for r in records if r.is_success)
        error_count = len(records) - success_count
        
        calc = PercentileCalculator(latencies)
        
        time_span = (records[-1].timestamp - records[0].timestamp).total_seconds() if len(records) > 1 else 1
        qps = len(records) / time_span if time_span > 0 else 0
        
        return EndpointStats(
            endpoint=endpoint,
            total_requests=len(records),
            success_requests=success_count,
            error_requests=error_count,
            total_latency=sum(latencies),
            min_latency=min(latencies) if latencies else 0,
            max_latency=max(latencies) if latencies else 0,
            latency_p50=calc.percentile(50),
            latency_p95=calc.percentile(95),
            latency_p99=calc.percentile(99),
            qps=qps,
            error_rate=(error_count / len(records) * 100) if records else 0,
            timestamp=datetime.now()
        )
    
    def get_all_stats(self, time_range: Optional[int] = None) -> Dict[str, EndpointStats]:
        """获取所有端点的统计信息"""
        return {
            endpoint: self.get_endpoint_stats(endpoint, time_range)
            for endpoint in self._records.keys()
        }
    
    def get_qps(self, endpoint: Optional[str] = None, time_range: int = 60) -> Union[float, Dict[str, float]]:
        """
        获取QPS
        
        Args:
            endpoint: 端点路径，None表示所有端点
            time_range: 时间窗口（秒）
        
        Returns:
            QPS值或QPS字典
        """
        now = datetime.now()
        cutoff = now - timedelta(seconds=time_range)
        
        if endpoint:
            records = self._records.get(endpoint, CircularBuffer(1)).get_all()
            recent_records = [r for r in records if r.timestamp >= cutoff]
            return len(recent_records) / time_range if time_range > 0 else 0
        
        result = {}
        for ep in self._records.keys():
            records = self._records.get(ep, CircularBuffer(1)).get_all()
            recent_records = [r for r in records if r.timestamp >= cutoff]
            result[ep] = len(recent_records) / time_range if time_range > 0 else 0
        
        return result
    
    def get_latency_percentiles(self, endpoint: str, 
                                percentiles: List[float] = [50, 90, 95, 99],
                                time_range: Optional[int] = None) -> Dict[str, float]:
        """
        获取延迟百分位数
        
        Args:
            endpoint: 端点路径
            percentiles: 百分位列表
            time_range: 时间范围（秒）
        
        Returns:
            百分位延迟字典
        """
        records = self._records.get(endpoint, CircularBuffer(1)).get_all()
        
        if time_range:
            cutoff = datetime.now() - timedelta(seconds=time_range)
            records = [r for r in records if r.timestamp >= cutoff]
        
        if not records:
            return {f'p{p}': 0.0 for p in percentiles}
        
        latencies = [r.latency for r in records]
        calc = PercentileCalculator(latencies)
        
        return {f'p{p}': calc.percentile(p) for p in percentiles}
    
    def get_error_rate(self, endpoint: Optional[str] = None,
                       time_range: int = 60) -> Union[float, Dict[str, float]]:
        """获取错误率"""
        now = datetime.now()
        cutoff = now - timedelta(seconds=time_range)
        
        if endpoint:
            records = self._records.get(endpoint, CircularBuffer(1)).get_all()
            recent_records = [r for r in records if r.timestamp >= cutoff]
            if not recent_records:
                return 0.0
            error_count = sum(1 for r in recent_records if r.is_error)
            return (error_count / len(recent_records)) * 100
        
        result = {}
        for ep in self._records.keys():
            records = self._records.get(ep, CircularBuffer(1)).get_all()
            recent_records = [r for r in records if r.timestamp >= cutoff]
            if not recent_records:
                result[ep] = 0.0
            else:
                error_count = sum(1 for r in recent_records if r.is_error)
                result[ep] = (error_count / len(recent_records)) * 100
        
        return result
    
    def get_time_window_stats(self, window_seconds: int = 60,
                              count: int = 10) -> List[TimeWindowStats]:
        """获取历史时间窗口统计"""
        now = datetime.now()
        windows = []
        
        for i in range(count):
            window_end = now - timedelta(seconds=window_seconds * i)
            window_start = window_end - timedelta(seconds=window_seconds)
            
            all_records = self._all_records.get_all()
            window_records = [
                r for r in all_records 
                if window_start <= r.timestamp < window_end
            ]
            
            if not window_records:
                windows.append(TimeWindowStats(
                    window_start=window_start,
                    window_end=window_end
                ))
                continue
            
            latencies = [r.latency for r in window_records]
            calc = PercentileCalculator(latencies)
            success_count = sum(1 for r in window_records if r.is_success)
            error_count = len(window_records) - success_count
            
            windows.append(TimeWindowStats(
                window_start=window_start,
                window_end=window_end,
                total_requests=len(window_records),
                success_requests=success_count,
                error_requests=error_count,
                qps=len(window_records) / window_seconds,
                avg_latency=sum(latencies) / len(latencies),
                min_latency=min(latencies),
                max_latency=max(latencies),
                p50_latency=calc.percentile(50),
                p95_latency=calc.percentile(95),
                p99_latency=calc.percentile(99),
                error_rate=(error_count / len(window_records)) * 100
            ))
        
        return list(reversed(windows))
    
    def get_recent_errors(self, limit: int = 100) -> List[RequestRecord]:
        """获取最近的错误请求"""
        all_records = self._all_records.get_recent(limit * 2)
        return [r for r in all_records if r.is_error][:limit]
    
    def get_records(self, endpoint: Optional[str] = None,
                    limit: int = 100) -> List[RequestRecord]:
        """获取请求记录"""
        if endpoint:
            return self._records.get(endpoint, CircularBuffer(1)).get_recent(limit)
        return self._all_records.get_recent(limit)
    
    def _check_alerts(self, endpoint: str, record: RequestRecord):
        """检查是否触发告警"""
        stats = self.get_endpoint_stats(endpoint, time_range=60)
        
        alerts = []
        
        if stats.qps < self._alert_thresholds.get('qps_min', 1.0):
            alerts.append(f"QPS过低: {stats.qps:.2f}")
        
        if stats.qps > self._alert_thresholds.get('qps_max', 10000.0):
            alerts.append(f"QPS过高: {stats.qps:.2f}")
        
        if stats.latency_p95 > self._alert_thresholds.get('latency_p95', 1000.0):
            alerts.append(f"P95延迟过高: {stats.latency_p95:.2f}ms")
        
        if stats.error_rate > self._alert_thresholds.get('error_rate', 5.0):
            alerts.append(f"错误率过高: {stats.error_rate:.2f}%")
        
        if record.is_error:
            alerts.append(f"请求失败: {record.error or f'Status {record.status_code}'}")
        
        for alert in alerts:
            self._logger.warning(f"[{endpoint}] {alert}")
            for callback in self._callbacks.get('alert', []):
                try:
                    callback(endpoint, alert, record)
                except Exception as e:
                    self._logger.error(f"告警回调执行失败: {e}")
    
    def register_alert_callback(self, callback: Callable):
        """注册告警回调"""
        if callback not in self._callbacks['alert']:
            self._callbacks['alert'].append(callback)
    
    def unregister_alert_callback(self, callback: Callable):
        """取消注册告警回调"""
        if callback in self._callbacks['alert']:
            self._callbacks['alert'].remove(callback)
    
    def set_thresholds(self, thresholds: Dict[str, float]):
        """设置告警阈值"""
        self._alert_thresholds.update(thresholds)
    
    def clear_stats(self, endpoint: Optional[str] = None):
        """清除统计数据"""
        if endpoint:
            if endpoint in self._records:
                self._records[endpoint].clear()
            if endpoint in self._endpoint_stats:
                del self._endpoint_stats[endpoint]
        else:
            for buffer in self._records.values():
                buffer.clear()
            self._all_records.clear()
            self._endpoint_stats.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        all_stats = self.get_all_stats(time_range=60)
        
        total_requests = sum(s.total_requests for s in all_stats.values())
        total_errors = sum(s.error_requests for s in all_stats.values())
        avg_qps = sum(s.qps for s in all_stats.values())
        
        all_latencies = []
        for buffer in self._records.values():
            for record in buffer.get_recent(1000):
                all_latencies.append(record.latency)
        
        p50 = p95 = p99 = 0.0
        if all_latencies:
            calc = PercentileCalculator(all_latencies)
            p50 = calc.percentile(50)
            p95 = calc.percentile(95)
            p99 = calc.percentile(99)
        
        return {
            'total_requests': total_requests,
            'total_errors': total_errors,
            'overall_error_rate': (total_errors / total_requests * 100) if total_requests else 0,
            'avg_qps': avg_qps,
            'p50_latency': p50,
            'p95_latency': p95,
            'p99_latency': p99,
            'endpoints': len(all_stats),
            'timestamp': datetime.now().isoformat()
        }
    
    def export_data(self, filepath: str, format: str = "json"):
        """导出监控数据"""
        data = {
            'summary': self.get_summary(),
            'endpoint_stats': {k: v.to_dict() for k, v in self.get_all_stats().items()},
            'recent_errors': [r.to_dict() for r in self.get_recent_errors(100)]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            if format == "json":
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            else:
                f.write(str(data))
        
        self._logger.info(f"监控数据已导出到: {filepath}")
    
    def start_monitoring(self):
        """开始监控线程"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self._logger.info("服务监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            self._monitor_thread = None
        self._logger.info("服务监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self._monitoring:
            try:
                stats = self.get_summary()
                for callback in self._callbacks.get('stats', []):
                    try:
                        callback(stats)
                    except Exception as e:
                        self._logger.error(f"统计回调执行失败: {e}")
            except Exception as e:
                self._logger.error(f"监控循环出错: {e}")
            time.sleep(self._window_size)
    
    def close(self):
        """关闭监控器"""
        self.stop_monitoring()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def track_request(monitor: Optional[ServiceQPSDelayMonitor] = None,
                  endpoint: Optional[str] = None):
    """
    请求跟踪装饰器
    
    Args:
        monitor: 监控器实例，None则使用全局单例
        endpoint: 端点名称，None则使用函数路径
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal monitor, endpoint
            
            if monitor is None:
                monitor = ServiceQPSDelayMonitor()
            
            if endpoint is None:
                ep = f"{func.__module__}.{func.__name__}"
            else:
                ep = endpoint
            
            start_time = time.time()
            error_msg = ""
            status_code = 200
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error_msg = str(e)
                status_code = 500
                raise
            finally:
                latency = (time.time() - start_time) * 1000
                monitor.record(
                    endpoint=ep,
                    latency=latency,
                    status_code=status_code,
                    method="CALL",
                    error=error_msg
                )
        
        return wrapper
    return decorator


def get_monitor(config: Optional[Dict[str, Any]] = None) -> ServiceQPSDelayMonitor:
    """
    获取监控器单例
    
    Args:
        config: 配置字典
    
    Returns:
        ServiceQPSDelayMonitor实例
    """
    return ServiceQPSDelayMonitor(config)


class ContextQPSMonitor:
    """
    上下文QPS监控器 - 用于Flask/Django等Web框架的请求跟踪
    
    使用方式:
    1. 初始化中间件
    2. 在请求处理中记录
    """
    
    def __init__(self, monitor: Optional[ServiceQPSDelayMonitor] = None):
        self._monitor = monitor or ServiceQPSDelayMonitor()
        self._local = threading.local()
    
    def before_request(self, endpoint: str, method: str):
        """请求开始前调用"""
        self._local.start_time = time.time()
        self._local.endpoint = endpoint
        self._local.method = method
    
    def after_request(self, status_code: int = 200):
        """请求结束后调用"""
        if hasattr(self._local, 'start_time') and hasattr(self._local, 'endpoint'):
            latency = (time.time() - self._local.start_time) * 1000
            self._monitor.record(
                endpoint=self._local.endpoint,
                latency=latency,
                status_code=status_code,
                method=getattr(self._local, 'method', 'GET')
            )
    
    def on_error(self, error: Exception):
        """请求出错时调用"""
        if hasattr(self._local, 'start_time') and hasattr(self._local, 'endpoint'):
            latency = (time.time() - self._local.start_time) * 1000
            self._monitor.record_error(
                endpoint=self._local.endpoint,
                latency=latency,
                error=str(error),
                method=getattr(self._local, 'method', 'GET')
            )


__all__ = [
    'RequestRecord',
    'EndpointStats',
    'TimeWindowStats',
    'ServiceQPSDelayMonitor',
    'CircularBuffer',
    'PercentileCalculator',
    'ContextQPSMonitor',
    'track_request',
    'get_monitor'
]
