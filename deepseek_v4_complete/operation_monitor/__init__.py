"""
运维监控模块包 - Operation Monitor Package

提供完整的系统运维监控功能，包括：
- 统一日志系统
- GPU状态监控
- CPU/内存监控
- 服务QPS/延迟监控
- 错误捕获和记录
- 短信/邮件告警
"""

from .unified_logger import (
    LogManager,
    LogLevel,
    JsonFormatter,
    ColoredFormatter,
    StructuredLogger,
    ContextLogger,
    PerformanceLogger,
    get_logger,
    get_structured_logger,
    configure_logger,
    log_decorator,
    exception_logger
)

from .gpu_status_monitor import (
    GPUInfo,
    GPUBackend,
    NVMLBackend,
    SMIBackend,
    MockGPUBackend,
    GPUStatusMonitor,
    get_monitor as get_gpu_monitor
)

from .cpu_mem_monitor import (
    CPUInfo,
    MemoryInfo,
    DiskInfo,
    NetworkInfo,
    LoadAverage,
    SystemMonitorBackend,
    PSUtilBackend,
    CommandBackend,
    MockBackend,
    CPUMemMonitor,
    get_monitor as get_cpu_monitor
)

from .service_qps_delay import (
    RequestRecord,
    EndpointStats,
    TimeWindowStats,
    ServiceQPSDelayMonitor,
    CircularBuffer,
    PercentileCalculator,
    ContextQPSMonitor,
    track_request,
    get_monitor as get_qps_monitor
)

from .error_catch_record import (
    ErrorLevel,
    ErrorCategory,
    ErrorInfo,
    ErrorStats,
    ErrorBackend,
    FileBackend,
    MemoryBackend,
    ErrorCatcher,
    ErrorContext,
    get_catcher,
    catch_errors
)

from .sms_email_alert import (
    AlertLevel,
    AlertChannel,
    Alert,
    AlertTemplate,
    AlertBackend,
    EmailBackend,
    SMSBackend,
    WebhookBackend,
    DingTalkBackend,
    WeChatBackend,
    FeishuBackend,
    ConsoleBackend,
    AlertSender,
    AlertDecorator,
    alert_on_error,
    get_sender
)


__version__ = "1.0.0"

__all__ = [
    "LogManager",
    "LogLevel",
    "JsonFormatter",
    "ColoredFormatter",
    "StructuredLogger",
    "ContextLogger",
    "PerformanceLogger",
    "get_logger",
    "get_structured_logger",
    "configure_logger",
    "log_decorator",
    "exception_logger",
    "GPUInfo",
    "GPUBackend",
    "NVMLBackend",
    "SMIBackend",
    "MockGPUBackend",
    "GPUStatusMonitor",
    "get_gpu_monitor",
    "CPUInfo",
    "MemoryInfo",
    "DiskInfo",
    "NetworkInfo",
    "LoadAverage",
    "SystemMonitorBackend",
    "PSUtilBackend",
    "CommandBackend",
    "MockBackend",
    "CPUMemMonitor",
    "get_cpu_monitor",
    "RequestRecord",
    "EndpointStats",
    "TimeWindowStats",
    "ServiceQPSDelayMonitor",
    "CircularBuffer",
    "PercentileCalculator",
    "ContextQPSMonitor",
    "track_request",
    "get_qps_monitor",
    "ErrorLevel",
    "ErrorCategory",
    "ErrorInfo",
    "ErrorStats",
    "ErrorBackend",
    "FileBackend",
    "MemoryBackend",
    "ErrorCatcher",
    "ErrorContext",
    "get_catcher",
    "catch_errors",
    "AlertLevel",
    "AlertChannel",
    "Alert",
    "AlertTemplate",
    "AlertBackend",
    "EmailBackend",
    "SMSBackend",
    "WebhookBackend",
    "DingTalkBackend",
    "WeChatBackend",
    "FeishuBackend",
    "ConsoleBackend",
    "AlertSender",
    "AlertDecorator",
    "alert_on_error",
    "get_sender",
]
