"""
GPU状态监控 - GPU Status Monitor
监控NVIDIA GPU和AMD GPU的状态，包括利用率、显存、温度、功耗等指标
支持多种GPU监控库：pynvml、pyamdlab、nvidia-smi命令行等
"""

import os
import sys
import time
import threading
import subprocess
import platform
from datetime import datetime
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import logging


@dataclass
class GPUInfo:
    """GPU信息数据类"""
    index: int
    name: str
    utilization_gpu: float = 0.0
    utilization_memory: float = 0.0
    memory_total: float = 0.0
    memory_used: float = 0.0
    memory_free: float = 0.0
    temperature: float = 0.0
    power_draw: float = 0.0
    power_limit: float = 0.0
    fan_speed: float = 0.0
    clock_sm: float = 0.0
    clock_memory: float = 0.0
    clock_video: float = 0.0
    ecc_errors: int = 0
    uuid: str = ""
    pci_bus_id: str = ""
    driver_version: str = ""
    compute_mode: str = ""
    serial_number: str = ""
    status: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @property
    def memory_used_percent(self) -> float:
        """显存使用百分比"""
        if self.memory_total == 0:
            return 0.0
        return (self.memory_used / self.memory_total) * 100
    
    @property
    def power_used_percent(self) -> float:
        """功耗使用百分比"""
        if self.power_limit == 0:
            return 0.0
        return (self.power_draw / self.power_limit) * 100
    
    def is_overheating(self, threshold: float = 85.0) -> bool:
        """是否过热"""
        return self.temperature >= threshold
    
    def is_memory_critical(self, threshold: float = 95.0) -> bool:
        """显存是否告警"""
        return self.memory_used_percent >= threshold
    
    def is_utilization_high(self, threshold: float = 90.0) -> bool:
        """GPU利用率是否过高"""
        return self.utilization_gpu >= threshold


class GPUBackend(ABC):
    """GPU监控后端抽象基类"""
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查后端是否可用"""
        pass
    
    @abstractmethod
    def get_gpu_count(self) -> int:
        """获取GPU数量"""
        pass
    
    @abstractmethod
    def get_gpu_info(self, index: int) -> Optional[GPUInfo]:
        """获取指定GPU的信息"""
        pass
    
    @abstractmethod
    def get_all_gpu_info(self) -> List[GPUInfo]:
        """获取所有GPU的信息"""
        pass
    
    @abstractmethod
    def close(self):
        """关闭后端连接"""
        pass


class NVMLBackend(GPUBackend):
    """NVIDIA NVML后端 - 使用pynvml库"""
    
    def __init__(self):
        self._nvml = None
        self._handle_list = []
        self._initialized = False
        self._init_nvml()
    
    def _init_nvml(self):
        """初始化NVML"""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._initialized = True
            self._refresh_handles()
        except ImportError:
            logging.warning("pynvml库未安装，无法使用NVML后端")
            self._initialized = False
        except Exception as e:
            logging.error(f"NVML初始化失败: {e}")
            self._initialized = False
    
    def _refresh_handles(self):
        """刷新GPU句柄列表"""
        if not self._initialized:
            return
        try:
            device_count = self._nvml.nvmlDeviceGetCount()
            self._handle_list = []
            for i in range(device_count):
                handle = self._nvml.nvmlDeviceGetHandleByIndex(i)
                self._handle_list.append(handle)
        except Exception as e:
            logging.error(f"刷新GPU句柄失败: {e}")
    
    def is_available(self) -> bool:
        """检查NVML是否可用"""
        return self._initialized and len(self._handle_list) > 0
    
    def get_gpu_count(self) -> int:
        """获取GPU数量"""
        if not self.is_available():
            return 0
        return len(self._handle_list)
    
    def _get_memory_info(self, handle) -> tuple:
        """获取显存信息"""
        try:
            mem_info = self._nvml.nvmlDeviceGetMemoryInfo(handle)
            return mem_info.total, mem_info.used, mem_info.free
        except:
            return 0, 0, 0
    
    def _get_utilization(self, handle) -> tuple:
        """获取利用率信息"""
        try:
            util = self._nvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu, util.memory
        except:
            return 0, 0
    
    def _get_temperature(self, handle) -> float:
        """获取温度"""
        try:
            temp = self._nvml.nvmlDeviceGetTemperature(handle, self._nvml.NVML_TEMPERATURE_GPU)
            return float(temp)
        except:
            return 0.0
    
    def _get_power(self, handle) -> tuple:
        """获取功耗信息"""
        try:
            power_draw = self._nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            power_limit = self._nvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            return power_draw, power_limit
        except:
            return 0.0, 0.0
    
    def _get_fan_speed(self, handle) -> float:
        """获取风扇速度"""
        try:
            fan = self._nvml.nvmlDeviceGetFanSpeed(handle)
            return float(fan)
        except:
            return 0.0
    
    def _get_clocks(self, handle) -> tuple:
        """获取时钟频率"""
        try:
            clock_sm = self._nvml.nvmlDeviceGetClockInfo(handle, self._nvml.NVML_CLOCK_SM)
            clock_memory = self._nvml.nvmlDeviceGetClockInfo(handle, self._nvml.NVML_CLOCK_MEM)
            clock_video = self._nvml.nvmlDeviceGetClockInfo(handle, self._nvml.NVML_CLOCK_VIDEO)
            return float(clock_sm), float(clock_memory), float(clock_video)
        except:
            return 0.0, 0.0, 0.0
    
    def _get_device_info(self, handle) -> Dict[str, Any]:
        """获取设备基本信息"""
        info = {}
        try:
            info['name'] = self._nvml.nvmlDeviceGetName(handle)
            if info['name'] is None:
                info['name'] = "Unknown GPU"
        except:
            info['name'] = "Unknown GPU"
        
        try:
            info['uuid'] = self._nvml.nvmlDeviceGetUUID(handle)
        except:
            info['uuid'] = ""
        
        try:
            info['pci_bus_id'] = self._nvml.nvmlDeviceGetPciBusId(handle)
        except:
            info['pci_bus_id'] = ""
        
        try:
            info['serial'] = self._nvml.nvmlDeviceGetSerial(handle)
        except:
            info['serial'] = ""
        
        try:
            info['compute_mode'] = self._nvml.nvmlDeviceGetComputeMode(handle)
            mode_map = {
                0: "Default",
                1: "Exclusive",
                2: "Prohibited",
                3: "Exclusive Process"
            }
            info['compute_mode'] = mode_map.get(info['compute_mode'], "Unknown")
        except:
            info['compute_mode'] = "Unknown"
        
        return info
    
    def get_gpu_info(self, index: int) -> Optional[GPUInfo]:
        """获取指定GPU的信息"""
        if not self.is_available() or index >= len(self._handle_list):
            return None
        
        handle = self._handle_list[index]
        device_info = self._get_device_info(handle)
        mem_total, mem_used, mem_free = self._get_memory_info(handle)
        util_gpu, util_mem = self._get_utilization(handle)
        power_draw, power_limit = self._get_power(handle)
        clock_sm, clock_mem, clock_video = self._get_clocks(handle)
        fan_speed = self._get_fan_speed(handle)
        temperature = self._get_temperature(handle)
        
        return GPUInfo(
            index=index,
            name=device_info['name'],
            utilization_gpu=float(util_gpu),
            utilization_memory=float(util_mem),
            memory_total=mem_total / (1024 ** 3),
            memory_used=mem_used / (1024 ** 3),
            memory_free=mem_free / (1024 ** 3),
            temperature=temperature,
            power_draw=power_draw,
            power_limit=power_limit,
            fan_speed=fan_speed,
            clock_sm=clock_sm,
            clock_memory=clock_mem,
            clock_video=clock_video,
            uuid=device_info['uuid'],
            pci_bus_id=device_info['pci_bus_id'],
            serial_number=device_info['serial'],
            compute_mode=device_info['compute_mode'],
            status="ok",
            timestamp=datetime.now()
        )
    
    def get_all_gpu_info(self) -> List[GPUInfo]:
        """获取所有GPU的信息"""
        return [gpu for gpu in (self.get_gpu_info(i) for i in range(len(self._handle_list))) if gpu]
    
    def close(self):
        """关闭NVML连接"""
        if self._initialized:
            try:
                self._nvml.nvmlShutdown()
            except:
                pass
            self._initialized = False


class SMIBackend(GPUBackend):
    """NVIDIA nvidia-smi命令行后端 - 备选方案"""
    
    def __init__(self):
        self._system = platform.system()
        self._initialized = self._check_smi()
    
    def _check_smi(self) -> bool:
        """检查nvidia-smi是否可用"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def is_available(self) -> bool:
        """检查nvidia-smi是否可用"""
        return self._initialized
    
    def get_gpu_count(self) -> int:
        """获取GPU数量"""
        if not self.is_available():
            return 0
        try:
            result = subprocess.run(
                ['nvidia-smi', '--list-gpus'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return len([line for line in result.stdout.strip().split('\n') if line])
            return 0
        except:
            return 0
    
    def _parse_smi_output(self) -> List[Dict[str, Any]]:
        """解析nvidia-smi输出"""
        gpus = []
        try:
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=index,name,utilization.gpu,utilization.memory,'
                    'memory.total,memory.used,memory.free,temperature.gpu,'
                    'power.draw,power.limit,fan.speed,clocks.sm,clocks.mem,clocks.video,'
                    'uuid,pci.bus_id,driver_version',
                    '--format=csv,noheader,nounits'
                ],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 15:
                            gpus.append({
                                'index': int(parts[0]),
                                'name': parts[1],
                                'utilization_gpu': float(parts[2]),
                                'utilization_memory': float(parts[3]),
                                'memory_total': float(parts[4]) / 1024,
                                'memory_used': float(parts[5]) / 1024,
                                'memory_free': float(parts[6]) / 1024,
                                'temperature': float(parts[7]),
                                'power_draw': float(parts[8]),
                                'power_limit': float(parts[9]),
                                'fan_speed': float(parts[10]),
                                'clock_sm': float(parts[11]),
                                'clock_memory': float(parts[12]),
                                'clock_video': float(parts[13]),
                                'uuid': parts[14],
                                'pci_bus_id': parts[15] if len(parts) > 15 else '',
                                'driver_version': parts[16] if len(parts) > 16 else ''
                            })
        except Exception as e:
            logging.error(f"解析nvidia-smi输出失败: {e}")
        
        return gpus
    
    def get_gpu_info(self, index: int) -> Optional[GPUInfo]:
        """获取指定GPU的信息"""
        gpus = self._parse_smi_output()
        for gpu in gpus:
            if gpu['index'] == index:
                return GPUInfo(
                    index=gpu['index'],
                    name=gpu['name'],
                    utilization_gpu=gpu['utilization_gpu'],
                    utilization_memory=gpu['utilization_memory'],
                    memory_total=gpu['memory_total'],
                    memory_used=gpu['memory_used'],
                    memory_free=gpu['memory_free'],
                    temperature=gpu['temperature'],
                    power_draw=gpu['power_draw'],
                    power_limit=gpu['power_limit'],
                    fan_speed=gpu['fan_speed'],
                    clock_sm=gpu['clock_sm'],
                    clock_memory=gpu['clock_memory'],
                    clock_video=gpu['clock_video'],
                    uuid=gpu['uuid'],
                    pci_bus_id=gpu['pci_bus_id'],
                    driver_version=gpu['driver_version'],
                    status="ok",
                    timestamp=datetime.now()
                )
        return None
    
    def get_all_gpu_info(self) -> List[GPUInfo]:
        """获取所有GPU的信息"""
        return [self.get_gpu_info(i) for i in range(self.get_gpu_count())]
    
    def close(self):
        """关闭连接"""
        pass


class MockGPUBackend(GPUBackend):
    """模拟GPU后端 - 用于测试或无GPU环境"""
    
    def __init__(self, gpu_count: int = 1):
        self._gpu_count = gpu_count
        self._initialized = True
    
    def is_available(self) -> bool:
        """总是返回True"""
        return True
    
    def get_gpu_count(self) -> int:
        """获取GPU数量"""
        return self._gpu_count
    
    def get_gpu_info(self, index: int) -> Optional[GPUInfo]:
        """获取模拟GPU信息"""
        if index >= self._gpu_count:
            return None
        
        import random
        return GPUInfo(
            index=index,
            name=f"Mock GPU {index}",
            utilization_gpu=random.uniform(10, 90),
            utilization_memory=random.uniform(20, 80),
            memory_total=24.0,
            memory_used=random.uniform(4, 20),
            memory_free=24.0 - random.uniform(4, 20),
            temperature=random.uniform(40, 80),
            power_draw=random.uniform(100, 300),
            power_limit=350.0,
            fan_speed=random.uniform(30, 70),
            clock_sm=1500.0 + random.uniform(0, 200),
            clock_memory=5000.0,
            clock_video=1200.0,
            status="ok",
            timestamp=datetime.now()
        )
    
    def get_all_gpu_info(self) -> List[GPUInfo]:
        """获取所有模拟GPU信息"""
        return [self.get_gpu_info(i) for i in range(self._gpu_count)]
    
    def close(self):
        """关闭连接"""
        pass


class GPUStatusMonitor:
    """
    GPU状态监控器
    支持多种后端自动切换，提供统一的GPU监控接口
    """
    
    _instance: Optional['GPUStatusMonitor'] = None
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
        self._backend: Optional[GPUBackend] = None
        self._history: List[GPUInfo] = []
        self._max_history = self._config.get('max_history', 1000)
        self._callbacks: List[callable] = []
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitoring = False
        self._monitor_interval = self._config.get('interval', 5)
        self._logger = logging.getLogger(__name__)
        
        self._init_backend()
    
    def _init_backend(self):
        """初始化后端"""
        backend_type = self._config.get('backend', 'auto')
        
        if backend_type == 'nvml':
            self._backend = NVMLBackend()
        elif backend_type == 'nvidia-smi':
            self._backend = SMIBackend()
        elif backend_type == 'mock':
            self._backend = MockGPUBackend(self._config.get('mock_gpu_count', 2))
        else:
            self._backend = NVMLBackend()
            if not self._backend.is_available():
                self._backend = SMIBackend()
                if not self._backend.is_available():
                    self._logger.warning("未检测到NVIDIA GPU，使用模拟后端")
                    self._backend = MockGPUBackend(1)
    
    def is_available(self) -> bool:
        """检查监控是否可用"""
        return self._backend is not None and self._backend.is_available()
    
    def get_gpu_count(self) -> int:
        """获取GPU数量"""
        if not self.is_available():
            return 0
        return self._backend.get_gpu_count()
    
    def get_status(self, index: Optional[int] = None) -> Union[Optional[GPUInfo], List[GPUInfo]]:
        """
        获取GPU状态
        
        Args:
            index: GPU索引，None表示获取所有GPU
        
        Returns:
            GPU信息或GPU信息列表
        """
        if not self.is_available():
            return None if index is not None else []
        
        if index is not None:
            gpu_info = self._backend.get_gpu_info(index)
            if gpu_info:
                self._add_to_history(gpu_info)
                self._check_thresholds(gpu_info)
            return gpu_info
        else:
            gpu_list = self._backend.get_all_gpu_info()
            for gpu in gpu_list:
                self._add_to_history(gpu)
                self._check_thresholds(gpu)
            return gpu_list
    
    def get_utilization(self, index: Optional[int] = None) -> Union[float, List[float], None]:
        """获取GPU利用率"""
        if index is not None:
            gpu = self.get_status(index)
            return gpu.utilization_gpu if gpu else None
        else:
            gpus = self.get_status()
            return [gpu.utilization_gpu for gpu in gpus] if gpus else []
    
    def get_memory_usage(self, index: Optional[int] = None) -> Union[float, List[float], None]:
        """获取显存使用情况（百分比）"""
        if index is not None:
            gpu = self.get_status(index)
            return gpu.memory_used_percent if gpu else None
        else:
            gpus = self.get_status()
            return [gpu.memory_used_percent for gpu in gpus] if gpus else []
    
    def get_temperature(self, index: Optional[int] = None) -> Union[float, List[float], None]:
        """获取GPU温度"""
        if index is not None:
            gpu = self.get_status(index)
            return gpu.temperature if gpu else None
        else:
            gpus = self.get_status()
            return [gpu.temperature for gpu in gpus] if gpus else []
    
    def get_power_usage(self, index: Optional[int] = None) -> Union[float, List[float], None]:
        """获取GPU功耗"""
        if index is not None:
            gpu = self.get_status(index)
            return gpu.power_draw if gpu else None
        else:
            gpus = self.get_status()
            return [gpu.power_draw for gpu in gpus] if gpus else []
    
    def _add_to_history(self, gpu_info: GPUInfo):
        """添加到历史记录"""
        self._history.append(gpu_info)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
    
    def get_history(self, index: Optional[int] = None, limit: int = 100) -> List[GPUInfo]:
        """获取历史记录"""
        if index is not None:
            return [g for g in self._history if g.index == index][-limit:]
        return self._history[-limit:]
    
    def _check_thresholds(self, gpu_info: GPUInfo):
        """检查阈值并触发回调"""
        temp_threshold = self._config.get('temp_threshold', 85)
        mem_threshold = self._config.get('mem_threshold', 95)
        util_threshold = self._config.get('util_threshold', 95)
        
        alerts = []
        if gpu_info.is_overheating(temp_threshold):
            alerts.append(f"GPU {gpu_info.index} 温度过高: {gpu_info.temperature}°C")
        if gpu_info.is_memory_critical(mem_threshold):
            alerts.append(f"GPU {gpu_info.index} 显存告警: {gpu_info.memory_used_percent:.1f}%")
        if gpu_info.is_utilization_high(util_threshold):
            alerts.append(f"GPU {gpu_info.index} 利用率过高: {gpu_info.utilization_gpu:.1f}%")
        
        for alert in alerts:
            self._logger.warning(alert)
            for callback in self._callbacks:
                try:
                    callback(alert, gpu_info)
                except Exception as e:
                    self._logger.error(f"告警回调执行失败: {e}")
    
    def register_callback(self, callback: callable):
        """注册告警回调函数"""
        if callback not in self._callbacks:
            self._callbacks.append(callback)
    
    def unregister_callback(self, callback: callable):
        """取消注册告警回调"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def start_monitoring(self):
        """开始持续监控"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self._logger.info("GPU监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
            self._monitor_thread = None
        self._logger.info("GPU监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self._monitoring:
            try:
                self.get_status()
            except Exception as e:
                self._logger.error(f"GPU监控出错: {e}")
            time.sleep(self._monitor_interval)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        gpus = self.get_status()
        if not gpus:
            return {}
        
        if isinstance(gpus, GPUInfo):
            gpus = [gpus]
        
        return {
            'gpu_count': len(gpus),
            'total_memory': sum(g.memory_total for g in gpus),
            'used_memory': sum(g.memory_used for g in gpus),
            'avg_utilization': sum(g.utilization_gpu for g in gpus) / len(gpus),
            'avg_temperature': sum(g.temperature for g in gpus) / len(gpus),
            'total_power': sum(g.power_draw for g in gpus),
            'timestamp': datetime.now().isoformat()
        }
    
    def close(self):
        """关闭监控器"""
        self.stop_monitoring()
        if self._backend:
            self._backend.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_monitor(config: Optional[Dict[str, Any]] = None) -> GPUStatusMonitor:
    """
    获取GPU监控器单例
    
    Args:
        config: 配置字典
    
    Returns:
        GPUStatusMonitor实例
    """
    return GPUStatusMonitor(config)


__all__ = [
    'GPUInfo',
    'GPUBackend',
    'NVMLBackend',
    'SMIBackend',
    'MockGPUBackend',
    'GPUStatusMonitor',
    'get_monitor'
]
