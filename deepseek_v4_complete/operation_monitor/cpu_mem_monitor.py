"""
CPU/内存监控 - CPU and Memory Monitor
监控系统的CPU、内存、磁盘、网络等系统资源使用情况
支持实时监控、历史记录、告警阈值等功能
"""

import os
import sys
import time
import threading
import platform
import subprocess
from datetime import datetime
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import logging


@dataclass
class CPUInfo:
    """CPU信息数据类"""
    user: float = 0.0
    system: float = 0.0
    idle: float = 100.0
    iowait: float = 0.0
    irq: float = 0.0
    softirq: float = 0.0
    steal: float = 0.0
    guest: float = 0.0
    guest_nice: float = 0.0
    count: int = 0
    count_logical: int = 0
    percent: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @property
    def used_percent(self) -> float:
        """已使用百分比"""
        return 100.0 - self.idle
    
    @property
    def is_busy(self) -> bool:
        """是否繁忙"""
        return self.percent > 80.0


@dataclass
class MemoryInfo:
    """内存信息数据类"""
    total: float = 0.0
    available: float = 0.0
    used: float = 0.0
    free: float = 0.0
    buffers: float = 0.0
    cached: float = 0.0
    percent: float = 0.0
    active: float = 0.0
    inactive: float = 0.0
    buffers_cached: float = 0.0
    swap_total: float = 0.0
    swap_used: float = 0.0
    swap_free: float = 0.0
    swap_percent: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @property
    def used_percent(self) -> float:
        """已使用百分比"""
        if self.total == 0:
            return 0.0
        return (self.used / self.total) * 100
    
    @property
    def available_percent(self) -> float:
        """可用百分比"""
        if self.total == 0:
            return 0.0
        return (self.available / self.total) * 100
    
    def is_low_memory(self, threshold: float = 20.0) -> bool:
        """是否内存不足"""
        return self.available_percent < threshold
    
    def is_swap_used(self) -> bool:
        """是否使用了swap"""
        return self.swap_used > 0


@dataclass
class DiskInfo:
    """磁盘信息数据类"""
    mountpoint: str = "/"
    device: str = ""
    fstype: str = ""
    total: float = 0.0
    used: float = 0.0
    free: float = 0.0
    percent: float = 0.0
    inodes_total: float = 0.0
    inodes_used: float = 0.0
    inodes_free: float = 0.0
    inodes_percent: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @property
    def used_percent(self) -> float:
        """已使用百分比"""
        return self.percent
    
    def is_disk_full(self, threshold: float = 90.0) -> bool:
        """磁盘是否将满"""
        return self.percent >= threshold


@dataclass
class NetworkInfo:
    """网络信息数据类"""
    interface: str = ""
    bytes_sent: int = 0
    bytes_recv: int = 0
    packets_sent: int = 0
    packets_recv: int = 0
    errin: int = 0
    errout: int = 0
    dropin: int = 0
    dropout: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class LoadAverage:
    """系统负载信息"""
    load1: float = 0.0
    load5: float = 0.0
    load15: float = 0.0
    running_threads: int = 0
    total_threads: int = 0
    last_pid: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class SystemMonitorBackend(ABC):
    """系统监控后端抽象基类"""
    
    @abstractmethod
    def get_cpu_info(self) -> CPUInfo:
        """获取CPU信息"""
        pass
    
    @abstractmethod
    def get_memory_info(self) -> MemoryInfo:
        """获取内存信息"""
        pass
    
    @abstractmethod
    def get_disk_info(self, mountpoint: str = "/") -> DiskInfo:
        """获取磁盘信息"""
        pass
    
    @abstractmethod
    def get_network_info(self, interface: str = "") -> List[NetworkInfo]:
        """获取网络信息"""
        pass
    
    @abstractmethod
    def get_load_average(self) -> LoadAverage:
        """获取系统负载"""
        pass
    
    @abstractmethod
    def get_uptime(self) -> float:
        """获取系统运行时间（秒）"""
        pass
    
    @abstractmethod
    def get_process_count(self) -> int:
        """获取进程数量"""
        pass


class PSUtilBackend(SystemMonitorBackend):
    """使用psutil库的后端实现"""
    
    def __init__(self):
        self._psutil = None
        self._initialized = False
        self._prev_cpu_times: Optional[Any] = None
        self._init_psutil()
    
    def _init_psutil(self):
        """初始化psutil"""
        try:
            import psutil
            self._psutil = psutil
            self._initialized = True
        except ImportError:
            logging.warning("psutil库未安装，无法使用psutil后端")
            self._initialized = False
        except Exception as e:
            logging.error(f"psutil初始化失败: {e}")
            self._initialized = False
    
    @property
    def is_available(self) -> bool:
        """检查psutil是否可用"""
        return self._initialized
    
    def get_cpu_info(self) -> CPUInfo:
        """获取CPU信息"""
        if not self.is_available:
            return CPUInfo()
        
        try:
            cpu_times = self._psutil.cpu_times()
            cpu_percent = self._psutil.cpu_percent(interval=0.1)
            cpu_count = self._psutil.cpu_count(logical=False) or 0
            cpu_count_logical = self._psutil.cpu_count(logical=True) or 0
            
            return CPUInfo(
                user=cpu_times.user,
                system=cpu_times.system,
                idle=cpu_times.idle,
                iowait=getattr(cpu_times, 'iowait', 0.0),
                irq=getattr(cpu_times, 'irq', 0.0),
                softirq=getattr(cpu_times, 'softirq', 0.0),
                steal=getattr(cpu_times, 'steal', 0.0),
                guest=getattr(cpu_times, 'guest', 0.0),
                guest_nice=getattr(cpu_times, 'guest_nice', 0.0),
                count=cpu_count,
                count_logical=cpu_count_logical,
                percent=cpu_percent,
                timestamp=datetime.now()
            )
        except Exception as e:
            logging.error(f"获取CPU信息失败: {e}")
            return CPUInfo()
    
    def get_memory_info(self) -> MemoryInfo:
        """获取内存信息"""
        if not self.is_available:
            return MemoryInfo()
        
        try:
            vm = self._psutil.virtual_memory()
            swap = self._psutil.swap_memory()
            
            return MemoryInfo(
                total=vm.total / (1024 ** 3),
                available=vm.available / (1024 ** 3),
                used=vm.used / (1024 ** 3),
                free=vm.free / (1024 ** 3),
                buffers=getattr(vm, 'buffers', 0) / (1024 ** 3) if hasattr(vm, 'buffers') else 0,
                cached=getattr(vm, 'cached', 0) / (1024 ** 3) if hasattr(vm, 'cached') else 0,
                percent=vm.percent,
                active=getattr(vm, 'active', 0) / (1024 ** 3) if hasattr(vm, 'active') else 0,
                inactive=getattr(vm, 'inactive', 0) / (1024 ** 3) if hasattr(vm, 'inactive') else 0,
                buffers_cached=(getattr(vm, 'buffers', 0) + getattr(vm, 'cached', 0)) / (1024 ** 3) if hasattr(vm, 'buffers') else 0,
                swap_total=swap.total / (1024 ** 3),
                swap_used=swap.used / (1024 ** 3),
                swap_free=swap.free / (1024 ** 3),
                swap_percent=swap.percent,
                timestamp=datetime.now()
            )
        except Exception as e:
            logging.error(f"获取内存信息失败: {e}")
            return MemoryInfo()
    
    def get_disk_info(self, mountpoint: str = "/") -> DiskInfo:
        """获取磁盘信息"""
        if not self.is_available:
            return DiskInfo()
        
        try:
            disk = self._psutil.disk_usage(mountpoint)
            disk_part = self._psutil.disk_partitions(all=False)
            
            device = ""
            fstype = ""
            for part in disk_part:
                if part.mountpoint == mountpoint:
                    device = part.device
                    fstype = part.fstype
                    break
            
            return DiskInfo(
                mountpoint=mountpoint,
                device=device,
                fstype=fstype,
                total=disk.total / (1024 ** 3),
                used=disk.used / (1024 ** 3),
                free=disk.free / (1024 ** 3),
                percent=disk.percent,
                timestamp=datetime.now()
            )
        except Exception as e:
            logging.error(f"获取磁盘信息失败: {e}")
            return DiskInfo()
    
    def get_network_info(self, interface: str = "") -> List[NetworkInfo]:
        """获取网络信息"""
        if not self.is_available:
            return []
        
        try:
            net_io = self._psutil.net_io_counters(pernic=True)
            result = []
            
            for iface, stats in net_io.items():
                if interface and iface != interface:
                    continue
                result.append(NetworkInfo(
                    interface=iface,
                    bytes_sent=stats.bytes_sent,
                    bytes_recv=stats.bytes_recv,
                    packets_sent=stats.packets_sent,
                    packets_recv=stats.packets_recv,
                    errin=stats.errin,
                    errout=stats.errout,
                    dropin=stats.dropin,
                    dropout=stats.dropout,
                    timestamp=datetime.now()
                ))
            return result
        except Exception as e:
            logging.error(f"获取网络信息失败: {e}")
            return []
    
    def get_load_average(self) -> LoadAverage:
        """获取系统负载"""
        try:
            load = os.getloadavg()
            proc_num = self._psutil.process_num_threads() if self.is_available else 0
            
            return LoadAverage(
                load1=load[0],
                load5=load[1],
                load15=load[2],
                running_threads=proc_num,
                timestamp=datetime.now()
            )
        except Exception as e:
            logging.error(f"获取系统负载失败: {e}")
            return LoadAverage()
    
    def get_uptime(self) -> float:
        """获取系统运行时间"""
        try:
            if self.is_available:
                boot_time = self._psutil.boot_time()
                return time.time() - boot_time
            return 0.0
        except Exception as e:
            logging.error(f"获取系统运行时间失败: {e}")
            return 0.0
    
    def get_process_count(self) -> int:
        """获取进程数量"""
        try:
            if self.is_available:
                return len(self._psutil.pids())
            return 0
        except Exception as e:
            logging.error(f"获取进程数量失败: {e}")
            return 0


class CommandBackend(SystemMonitorBackend):
    """使用系统命令的后端实现（备选方案）"""
    
    def __init__(self):
        self._system = platform.system()
        self._initialized = True
    
    @property
    def is_available(self) -> bool:
        return self._initialized
    
    def _run_command(self, cmd: List[str]) -> str:
        """执行命令并返回输出"""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return result.stdout
        except Exception:
            return ""
    
    def get_cpu_info(self) -> CPUInfo:
        """通过top命令获取CPU信息"""
        cpu_info = CPUInfo()
        
        if self._system == "Linux":
            try:
                with open('/proc/stat', 'r') as f:
                    line = f.readline()
                    parts = line.split()
                    if parts[0] == 'cpu':
                        cpu_info.user = float(parts[1])
                        cpu_info.system = float(parts[3])
                        cpu_info.idle = float(parts[4])
                        cpu_info.iowait = float(parts[5]) if len(parts) > 5 else 0
                        cpu_info.irq = float(parts[6]) if len(parts) > 6 else 0
                        cpu_info.softirq = float(parts[7]) if len(parts) > 7 else 0
            except Exception:
                pass
            
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    content = f.read()
                    cpu_info.count = content.count('processor')
                    cpu_info.count_logical = cpu_info.count
            except Exception:
                pass
        
        return cpu_info
    
    def get_memory_info(self) -> MemoryInfo:
        """通过free命令获取内存信息"""
        mem_info = MemoryInfo()
        
        if self._system == "Linux":
            output = self._run_command(['free', '-b'])
            lines = output.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                if len(parts) >= 3:
                    mem_info.total = float(parts[1]) / (1024 ** 3)
                    mem_info.used = float(parts[2]) / (1024 ** 3)
                    mem_info.free = float(parts[3]) / (1024 ** 3) if len(parts) > 3 else 0
                    mem_info.buffers = float(parts[5]) / (1024 ** 3) if len(parts) > 5 else 0
                    mem_info.cached = float(parts[6]) / (1024 ** 3) if len(parts) > 6 else 0
        
        return mem_info
    
    def get_disk_info(self, mountpoint: str = "/") -> DiskInfo:
        """通过df命令获取磁盘信息"""
        disk_info = DiskInfo()
        
        if self._system == "Linux":
            output = self._run_command(['df', '-B1', mountpoint])
            lines = output.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                if len(parts) >= 6:
                    disk_info.total = float(parts[1]) / (1024 ** 3)
                    disk_info.used = float(parts[2]) / (1024 ** 3)
                    disk_info.free = float(parts[3]) / (1024 ** 3)
                    disk_info.percent = float(parts[4].rstrip('%'))
                    disk_info.mountpoint = mountpoint
        
        return disk_info
    
    def get_network_info(self, interface: str = "") -> List[NetworkInfo]:
        """通过/proc/net/dev获取网络信息"""
        net_info_list = []
        
        if self._system == "Linux":
            try:
                with open('/proc/net/dev', 'r') as f:
                    lines = f.readlines()[2:]
                    for line in lines:
                        parts = line.split(':')
                        if len(parts) == 2:
                            iface = parts[0].strip()
                            values = parts[1].split()
                            if len(values) >= 10:
                                net_info_list.append(NetworkInfo(
                                    interface=iface,
                                    bytes_recv=int(values[0]),
                                    packets_recv=int(values[1]),
                                    errin=int(values[2]),
                                    dropin=int(values[3]),
                                    bytes_sent=int(values[8]),
                                    packets_sent=int(values[9]),
                                    errout=int(values[10]) if len(values) > 10 else 0,
                                    dropout=int(values[11]) if len(values) > 11 else 0
                                ))
            except Exception:
                pass
        
        return net_info_list
    
    def get_load_average(self) -> LoadAverage:
        """通过os.getloadavg获取负载"""
        try:
            load = os.getloadavg()
            return LoadAverage(
                load1=load[0],
                load5=load[1],
                load15=load[2],
                timestamp=datetime.now()
            )
        except Exception:
            return LoadAverage()
    
    def get_uptime(self) -> float:
        """通过/proc/uptime获取运行时间"""
        try:
            with open('/proc/uptime', 'r') as f:
                return float(f.read().split()[0])
        except Exception:
            return 0.0
    
    def get_process_count(self) -> int:
        """通过/proc获取进程数"""
        try:
            if self._system == "Linux":
                return len([d for d in os.listdir('/proc') if d.isdigit()])
            return 0
        except Exception:
            return 0


class MockBackend(SystemMonitorBackend):
    """模拟后端 - 用于测试"""
    
    def __init__(self):
        self._initialized = True
        import random
        self._random = random
    
    @property
    def is_available(self) -> bool:
        return True
    
    def get_cpu_info(self) -> CPUInfo:
        """生成模拟CPU信息"""
        return CPUInfo(
            user=self._random.uniform(10, 40),
            system=self._random.uniform(5, 20),
            idle=self._random.uniform(40, 80),
            iowait=self._random.uniform(0, 5),
            irq=self._random.uniform(0, 2),
            softirq=self._random.uniform(0, 2),
            count=8,
            count_logical=16,
            percent=self._random.uniform(20, 60),
            timestamp=datetime.now()
        )
    
    def get_memory_info(self) -> MemoryInfo:
        """生成模拟内存信息"""
        total = 64.0
        used = self._random.uniform(20, 50)
        return MemoryInfo(
            total=total,
            used=used,
            free=total - used,
            available=total - used + self._random.uniform(5, 10),
            buffers=self._random.uniform(1, 5),
            cached=self._random.uniform(10, 20),
            percent=(used / total) * 100,
            swap_total=32.0,
            swap_used=self._random.uniform(0, 2),
            swap_free=32.0,
            swap_percent=self._random.uniform(0, 5),
            timestamp=datetime.now()
        )
    
    def get_disk_info(self, mountpoint: str = "/") -> DiskInfo:
        """生成模拟磁盘信息"""
        total = 500.0
        used = self._random.uniform(100, 400)
        return DiskInfo(
            mountpoint=mountpoint,
            device="/dev/sda1",
            fstype="ext4",
            total=total,
            used=used,
            free=total - used,
            percent=(used / total) * 100,
            timestamp=datetime.now()
        )
    
    def get_network_info(self, interface: str = "") -> List[NetworkInfo]:
        """生成模拟网络信息"""
        return [NetworkInfo(
            interface="eth0",
            bytes_sent=self._random.randint(1000000, 100000000),
            bytes_recv=self._random.randint(1000000, 100000000),
            packets_sent=self._random.randint(1000, 10000),
            packets_recv=self._random.randint(1000, 10000),
            timestamp=datetime.now()
        )]
    
    def get_load_average(self) -> LoadAverage:
        """生成模拟负载信息"""
        return LoadAverage(
            load1=self._random.uniform(0.5, 3.0),
            load5=self._random.uniform(0.5, 2.5),
            load15=self._random.uniform(0.5, 2.0),
            running_threads=self._random.randint(100, 500),
            total_threads=self._random.randint(500, 1000),
            timestamp=datetime.now()
        )
    
    def get_uptime(self) -> float:
        """返回模拟运行时间"""
        return self._random.uniform(86400, 864000)
    
    def get_process_count(self) -> int:
        """返回模拟进程数"""
        return self._random.randint(200, 500)


class CPUMemMonitor:
    """
    CPU/内存监控器
    提供统一的系统资源监控接口
    """
    
    _instance: Optional['CPUMemMonitor'] = None
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
        self._backend: Optional[SystemMonitorBackend] = None
        self._cpu_history: List[CPUInfo] = []
        self._mem_history: List[MemoryInfo] = []
        self._max_history = self._config.get('max_history', 1000)
        self._callbacks: Dict[str, List[callable]] = {
            'cpu': [],
            'memory': [],
            'disk': [],
            'network': []
        }
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitoring = False
        self._monitor_interval = self._config.get('interval', 5)
        self._logger = logging.getLogger(__name__)
        
        self._init_backend()
    
    def _init_backend(self):
        """初始化后端"""
        backend_type = self._config.get('backend', 'auto')
        
        if backend_type == 'psutil':
            self._backend = PSUtilBackend()
            if not self._backend.is_available:
                self._logger.warning("psutil后端不可用，使用命令行后端")
                self._backend = CommandBackend()
        elif backend_type == 'command':
            self._backend = CommandBackend()
        elif backend_type == 'mock':
            self._backend = MockBackend()
        else:
            self._backend = PSUtilBackend()
            if not self._backend.is_available:
                self._backend = CommandBackend()
                if not self._backend.is_available:
                    self._logger.warning("系统监控不可用，使用模拟后端")
                    self._backend = MockBackend()
    
    @property
    def is_available(self) -> bool:
        """检查监控是否可用"""
        return self._backend is not None and self._backend.is_available
    
    def get_usage(self) -> Dict[str, Any]:
        """
        获取CPU和内存使用情况（兼容旧接口）
        
        Returns:
            包含cpu和mem使用情况的字典
        """
        cpu = self.get_cpu()
        mem = self.get_memory()
        
        return {
            "cpu": f"{cpu.percent:.1f}%",
            "mem": f"{mem.used_percent:.1f}%"
        }
    
    def get_cpu(self) -> CPUInfo:
        """获取CPU信息"""
        cpu_info = self._backend.get_cpu_info()
        self._add_to_history('cpu', cpu_info)
        self._check_thresholds('cpu', cpu_info)
        return cpu_info
    
    def get_memory(self) -> MemoryInfo:
        """获取内存信息"""
        mem_info = self._backend.get_memory_info()
        self._add_to_history('memory', mem_info)
        self._check_thresholds('memory', mem_info)
        return mem_info
    
    def get_disk(self, mountpoint: str = "/") -> DiskInfo:
        """获取磁盘信息"""
        disk_info = self._backend.get_disk_info(mountpoint)
        self._add_to_history('disk', disk_info)
        self._check_thresholds('disk', disk_info)
        return disk_info
    
    def get_network(self, interface: str = "") -> List[NetworkInfo]:
        """获取网络信息"""
        return self._backend.get_network_info(interface)
    
    def get_load(self) -> LoadAverage:
        """获取系统负载"""
        return self._backend.get_load_average()
    
    def get_uptime(self) -> float:
        """获取系统运行时间"""
        return self._backend.get_uptime()
    
    def get_process_count(self) -> int:
        """获取进程数量"""
        return self._backend.get_process_count()
    
    def _add_to_history(self, category: str, info):
        """添加到历史记录"""
        if category == 'cpu':
            self._cpu_history.append(info)
            if len(self._cpu_history) > self._max_history:
                self._cpu_history = self._cpu_history[-self._max_history:]
        elif category == 'memory':
            self._mem_history.append(info)
            if len(self._mem_history) > self._max_history:
                self._mem_history = self._mem_history[-self._max_history:]
    
    def get_cpu_history(self, limit: int = 100) -> List[CPUInfo]:
        """获取CPU历史记录"""
        return self._cpu_history[-limit:]
    
    def get_memory_history(self, limit: int = 100) -> List[MemoryInfo]:
        """获取内存历史记录"""
        return self._mem_history[-limit:]
    
    def _check_thresholds(self, category: str, info):
        """检查阈值并触发回调"""
        if category == 'cpu':
            cpu_threshold = self._config.get('cpu_threshold', 90)
            if info.percent > cpu_threshold:
                self._trigger_callbacks('cpu', f"CPU使用率过高: {info.percent:.1f}%")
        elif category == 'memory':
            mem_threshold = self._config.get('mem_threshold', 90)
            if info.used_percent > mem_threshold:
                self._trigger_callbacks('memory', f"内存使用率过高: {info.used_percent:.1f}%")
            if info.is_swap_used():
                self._trigger_callbacks('memory', f"系统正在使用swap: {info.swap_used:.2f}GB")
        elif category == 'disk':
            disk_threshold = self._config.get('disk_threshold', 90)
            if info.is_disk_full(disk_threshold):
                self._trigger_callbacks('disk', f"磁盘使用率过高: {info.percent:.1f}%")
    
    def _trigger_callbacks(self, category: str, message: str):
        """触发回调函数"""
        self._logger.warning(message)
        for callback in self._callbacks.get(category, []):
            try:
                callback(message)
            except Exception as e:
                self._logger.error(f"回调执行失败: {e}")
    
    def register_callback(self, category: str, callback: callable):
        """注册回调函数"""
        if category in self._callbacks and callback not in self._callbacks[category]:
            self._callbacks[category].append(callback)
    
    def unregister_callback(self, category: str, callback: callable):
        """取消注册回调"""
        if category in self._callbacks and callback in self._callbacks[category]:
            self._callbacks[category].remove(callback)
    
    def start_monitoring(self):
        """开始持续监控"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self._logger.info("系统监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
            self._monitor_thread = None
        self._logger.info("系统监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self._monitoring:
            try:
                self.get_cpu()
                self.get_memory()
                self.get_disk()
            except Exception as e:
                self._logger.error(f"系统监控出错: {e}")
            time.sleep(self._monitor_interval)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        cpu = self.get_cpu()
        mem = self.get_memory()
        disk = self.get_disk()
        load = self.get_load()
        
        return {
            'cpu': {
                'percent': cpu.percent,
                'count': cpu.count,
                'count_logical': cpu.count_logical
            },
            'memory': {
                'total': mem.total,
                'used': mem.used,
                'available': mem.available,
                'percent': mem.used_percent
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent
            },
            'load': {
                '1min': load.load1,
                '5min': load.load5,
                '15min': load.load15
            },
            'uptime': self.get_uptime(),
            'process_count': self.get_process_count(),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_all_info(self) -> Dict[str, Any]:
        """获取所有监控信息"""
        return {
            'cpu': self.get_cpu().to_dict(),
            'memory': self.get_memory().to_dict(),
            'disk': {d.mountpoint: d.to_dict() for d in [self.get_disk()]},
            'network': [n.to_dict() for n in self.get_network()],
            'load': self.get_load().to_dict(),
            'timestamp': datetime.now().isoformat()
        }
    
    def close(self):
        """关闭监控器"""
        self.stop_monitoring()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_monitor(config: Optional[Dict[str, Any]] = None) -> CPUMemMonitor:
    """
    获取CPU/内存监控器单例
    
    Args:
        config: 配置字典
    
    Returns:
        CPUMemMonitor实例
    """
    return CPUMemMonitor(config)


__all__ = [
    'CPUInfo',
    'MemoryInfo',
    'DiskInfo',
    'NetworkInfo',
    'LoadAverage',
    'SystemMonitorBackend',
    'PSUtilBackend',
    'CommandBackend',
    'MockBackend',
    'CPUMemMonitor',
    'get_monitor'
]
