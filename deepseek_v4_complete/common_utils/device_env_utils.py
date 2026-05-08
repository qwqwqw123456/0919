"""
设备环境检测工具模块

提供完整的设备环境检测功能，包括：
- GPU/CUDA 检测和配置
- CPU 检测
- 内存检测
- 磁盘空间检测
- 操作系统检测
- Python 环境检测
- 深度学习框架检测
"""

import os
import sys
import platform
import subprocess
import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class DeviceType(Enum):
    """设备类型枚举"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    CPU_FALLBACK = "cpu_fallback"


@dataclass
class GPUInfo:
    """GPU 信息数据类"""
    index: int
    name: str
    total_memory: float
    available_memory: float
    utilization: float
    temperature: Optional[float] = None
    compute_capability: Optional[Tuple[int, int]] = None
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    is_available: bool = True


@dataclass
class MemoryInfo:
    """内存信息数据类"""
    total: float
    available: float
    used: float
    percent: float


@dataclass
class DiskInfo:
    """磁盘信息数据类"""
    total: float
    used: float
    free: float
    percent: float
    mount_point: str = "/"


@dataclass
class SystemInfo:
    """系统信息数据类"""
    os_name: str
    os_version: str
    architecture: str
    processor: str
    python_version: str
    hostname: str


@dataclass
class DeviceInfo:
    """完整设备信息数据类"""
    device_type: DeviceType
    primary_device: str
    gpu_count: int = 0
    gpus: List[GPUInfo] = field(default_factory=list)
    cpu_count: int = 0
    total_memory: float = 0.0
    available_memory: float = 0.0
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    torch_version: Optional[str] = None
    tensorflow_version: Optional[str] = None
    system_info: Optional[SystemInfo] = None


class DeviceManager:
    """
    设备管理器类

    提供统一的设备检测和配置接口
    """

    _instance: Optional['DeviceManager'] = None

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._torch_device = None
        self._device_info: Optional[DeviceInfo] = None

    def get_device(self, device_id: Optional[int] = None) -> Any:
        """获取 PyTorch 设备对象"""
        try:
            import torch
            if device_id is not None:
                return torch.device(f"cuda:{device_id}")
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            warnings.warn("PyTorch 未安装，返回 CPU 设备")
            return "cpu"

    def get_device_info(self, force_refresh: bool = False) -> DeviceInfo:
        """获取完整的设备信息"""
        if self._device_info is None or force_refresh:
            self._device_info = self._collect_device_info()
        return self._device_info

    def _collect_device_info(self) -> DeviceInfo:
        """收集所有设备信息"""
        device_type = self._detect_device_type()
        primary_device = self._get_primary_device()

        info = DeviceInfo(
            device_type=device_type,
            primary_device=primary_device,
            cuda_available=self._check_cuda_available(),
            cuda_version=self._get_cuda_version(),
            torch_version=self._get_package_version("torch"),
            tensorflow_version=self._get_package_version("tensorflow")
        )

        if device_type == DeviceType.CUDA:
            info.gpu_count = self._get_gpu_count()
            info.gpus = self._get_gpu_details()

        info.cpu_count = self._get_cpu_count()
        mem_info = self.get_memory_info()
        info.total_memory = mem_info.total
        info.available_memory = mem_info.available
        info.system_info = self.get_system_info()

        return info

    def _detect_device_type(self) -> DeviceType:
        """检测设备类型"""
        if self._check_cuda_available():
            return DeviceType.CUDA
        elif self._check_mps_available():
            return DeviceType.MPS
        else:
            return DeviceType.CPU

    def _get_primary_device(self) -> str:
        """获取主设备字符串"""
        device_type = self._detect_device_type()
        if device_type == DeviceType.CUDA:
            return "cuda:0"
        elif device_type == DeviceType.MPS:
            return "mps"
        return "cpu"

    def _check_cuda_available(self) -> bool:
        """检查 CUDA 是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _check_mps_available(self) -> bool:
        """检查 MPS (Apple Silicon) 是否可用"""
        try:
            import torch
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except (ImportError, AttributeError):
            return False

    def _get_cuda_version(self) -> Optional[str]:
        """获取 CUDA 版本"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.version.cuda
            return None
        except ImportError:
            pass

        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        parts = line.split('release')
                        if len(parts) > 1:
                            return parts[1].strip().split()[0]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return None

    def _get_gpu_count(self) -> int:
        """获取 GPU 数量"""
        try:
            import torch
            return torch.cuda.device_count()
        except ImportError:
            return 0

    def _get_gpu_details(self) -> List[GPUInfo]:
        """获取 GPU 详细信息"""
        gpus = []
        try:
            import torch
            if not torch.cuda.is_available():
                return gpus

            for i in range(torch.cuda.device_count()):
                try:
                    props = torch.cuda.get_device_properties(i)
                    total_mem = props.total_memory / (1024 ** 3)

                    gpu_info = GPUInfo(
                        index=i,
                        name=props.name,
                        total_memory=total_mem,
                        available_memory=total_mem,
                        utilization=0.0,
                        compute_capability=(props.major, props.minor)
                    )
                    gpus.append(gpu_info)
                except Exception:
                    continue

        except ImportError:
            pass

        return gpus

    def _get_cpu_count(self) -> int:
        """获取 CPU 核心数"""
        return os.cpu_count() or 1

    def _get_package_version(self, package_name: str) -> Optional[str]:
        """获取 Python 包版本"""
        try:
            module = __import__(package_name)
            return getattr(module, '__version__', None)
        except ImportError:
            return None

    def get_memory_info(self) -> MemoryInfo:
        """获取系统内存信息"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return MemoryInfo(
                total=mem.total / (1024 ** 3),
                available=mem.available / (1024 ** 3),
                used=mem.used / (1024 ** 3),
                percent=mem.percent
            )
        except ImportError:
            return self._get_memory_info_fallback()

    def _get_memory_info_fallback(self) -> MemoryInfo:
        """使用命令行获取内存信息（psutil 不可用时）"""
        if sys.platform == "linux":
            try:
                with open('/proc/meminfo', 'r') as f:
                    lines = f.readlines()
                    mem = {}
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 2:
                            key = parts[0].rstrip(':')
                            try:
                                mem[key] = int(parts[1]) / (1024 ** 2)
                            except ValueError:
                                continue

                    total = mem.get('MemTotal', 0)
                    available = mem.get('MemAvailable', mem.get('MemFree', 0))
                    used = total - available
                    percent = (used / total * 100) if total > 0 else 0

                    return MemoryInfo(
                        total=total,
                        available=available,
                        used=used,
                        percent=percent
                    )
            except Exception:
                pass

        return MemoryInfo(total=0, available=0, used=0, percent=0)

    def get_disk_info(self, path: str = "/") -> DiskInfo:
        """获取磁盘信息"""
        try:
            import psutil
            disk = psutil.disk_usage(path)
            return DiskInfo(
                total=disk.total / (1024 ** 3),
                used=disk.used / (1024 ** 3),
                free=disk.free / (1024 ** 3),
                percent=disk.percent,
                mount_point=path
            )
        except ImportError:
            return self._get_disk_info_fallback(path)

    def _get_disk_info_fallback(self, path: str) -> DiskInfo:
        """使用命令行获取磁盘信息"""
        if sys.platform == "linux":
            try:
                result = subprocess.run(
                    ["df", "-BG", path],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) >= 2:
                        parts = lines[1].split()
                        if len(parts) >= 4:
                            total = float(parts[1].rstrip('G'))
                            used = float(parts[2].rstrip('G'))
                            free = float(parts[3].rstrip('G'))
                            percent = (used / total * 100) if total > 0 else 0
                            return DiskInfo(
                                total=total,
                                used=used,
                                free=free,
                                percent=percent,
                                mount_point=path
                            )
            except Exception:
                pass

        return DiskInfo(total=0, used=0, free=0, percent=0, mount_point=path)

    def get_system_info(self) -> SystemInfo:
        """获取系统信息"""
        return SystemInfo(
            os_name=platform.system(),
            os_version=platform.version(),
            architecture=platform.machine(),
            processor=platform.processor(),
            python_version=sys.version,
            hostname=platform.node()
        )

    def check_memory_sufficient(self, required_gb: float) -> bool:
        """检查可用内存是否满足要求"""
        mem_info = self.get_memory_info()
        return mem_info.available >= required_gb

    def check_disk_sufficient(self, required_gb: float, path: str = "/") -> bool:
        """检查可用磁盘空间是否满足要求"""
        disk_info = self.get_disk_info(path)
        return disk_info.free >= required_gb

    def get_gpu_memory_info(self, device_id: int = 0) -> Tuple[float, float]:
        """获取指定 GPU 的内存信息"""
        try:
            import torch
            if torch.cuda.is_available() and device_id < torch.cuda.device_count():
                allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
                total = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)
                return allocated, total - allocated
        except Exception:
            pass
        return 0.0, 0.0

    def print_device_info(self) -> None:
        """打印设备信息"""
        info = self.get_device_info()
        print("=" * 50)
        print("设备信息")
        print("=" * 50)
        print(f"设备类型: {info.device_type.value}")
        print(f"主设备: {info.primary_device}")
        print(f"CPU 核心数: {info.cpu_count}")
        print(f"总内存: {info.total_memory:.2f} GB")
        print(f"可用内存: {info.available_memory:.2f} GB")
        print(f"CUDA 可用: {info.cuda_available}")

        if info.cuda_version:
            print(f"CUDA 版本: {info.cuda_version}")
        if info.torch_version:
            print(f"PyTorch 版本: {info.torch_version}")
        if info.tensorflow_version:
            print(f"TensorFlow 版本: {info.tensorflow_version}")

        if info.gpu_count > 0:
            print(f"\nGPU 数量: {info.gpu_count}")
            for gpu in info.gpus:
                print(f"  GPU {gpu.index}: {gpu.name}")
                print(f"    显存: {gpu.total_memory:.2f} GB")
                if gpu.compute_capability:
                    print(f"    计算能力: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}")

        if info.system_info:
            print(f"\n系统信息:")
            print(f"  操作系统: {info.system_info.os_name}")
            print(f"  版本: {info.system_info.os_version}")
            print(f"  架构: {info.system_info.architecture}")
            print(f"  主机名: {info.system_info.hostname}")

        print("=" * 50)


def get_device() -> Any:
    """便捷函数：获取当前可用设备"""
    return DeviceManager().get_device()


def get_device_info(force_refresh: bool = False) -> DeviceInfo:
    """便捷函数：获取设备信息"""
    return DeviceManager().get_device_info(force_refresh)


def check_cuda_available() -> bool:
    """便捷函数：检查 CUDA 是否可用"""
    return DeviceManager()._check_cuda_available()


def get_gpu_count() -> int:
    """便捷函数：获取 GPU 数量"""
    return DeviceManager()._get_gpu_count()


def get_memory_info() -> MemoryInfo:
    """便捷函数：获取内存信息"""
    return DeviceManager().get_memory_info()


def get_disk_info(path: str = "/") -> DiskInfo:
    """便捷函数：获取磁盘信息"""
    return DeviceManager().get_disk_info(path)


def get_system_info() -> SystemInfo:
    """便捷函数：获取系统信息"""
    return DeviceManager().get_system_info()


def set_gpu_device(device_id: int) -> None:
    """便捷函数：设置当前使用的 GPU 设备"""
    try:
        import torch
        torch.cuda.set_device(device_id)
    except ImportError:
        warnings.warn("PyTorch 未安装")


def empty_cache() -> None:
    """便捷函数：清空 GPU 缓存"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def reset_peak_memory_stats() -> None:
    """便捷函数：重置 GPU 峰值内存统计"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass
