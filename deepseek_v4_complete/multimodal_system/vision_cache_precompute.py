"""
Vision Cache Precompute - 视觉缓存预计算模块
============================================

该模块提供视觉特征的缓存和预计算功能，用于加速多模态推理。

主要组件：
- VisionCacheConfig: 缓存配置类
- VisionCachePrecompute: 视觉缓存预计算器
- CacheManager: 缓存管理器
- TileCacheManager: 瓦片级缓存管理器

功能：
- 支持批量预计算图像特征
- 支持缓存存储和加载
- 支持LRU缓存淘汰策略
- 支持多进程并行计算
- 支持瓦片级缓存管理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import hashlib
import json
import shutil
import threading
import queue
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image
from collections import OrderedDict
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


@dataclass
class VisionCacheConfig:
    """
    视觉缓存配置

    参数:
        cache_dir: 缓存存储目录
        max_cache_size: 最大缓存数量
        cache_policy: 缓存策略 ('lru', 'lfu', 'fifo')
        precompute_batch_size: 预计算批次大小
        num_workers: 并行工作进程数
        enable_disk_cache: 是否启用磁盘缓存
        enable_memory_cache: 是否启用内存缓存
        cache_compression: 是否压缩缓存
        cache_format: 缓存格式 ('pt', 'npz', 'onnx')
        tile_size: 瓦片大小（用于高分辨率图像）
        overlap_ratio: 瓦片重叠比例
    """
    cache_dir: str = "vision_cache"
    max_cache_size: int = 10000
    cache_policy: str = "lru"
    precompute_batch_size: int = 32
    num_workers: int = 4
    enable_disk_cache: bool = True
    enable_memory_cache: bool = True
    cache_compression: bool = False
    cache_format: str = "pt"
    tile_size: int = 448
    overlap_ratio: float = 0.25
    cache_metadata_file: str = "cache_metadata.json"

    def __post_init__(self):
        self.tile_overlap = int(self.tile_size * self.overlap_ratio)


class LRUCache:
    """
    LRU (Least Recently Used) 缓存实现
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def put(self, key: str, value: Any):
        """放入缓存项"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.capacity:
                    self.cache.popitem(last=False)
            self.cache[key] = value

    def remove(self, key: str):
        """移除缓存项"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]

    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()

    def __len__(self):
        return len(self.cache)

    def keys(self):
        return list(self.cache.keys())


class LFUCache:
    """
    LFU (Least Frequently Used) 缓存实现
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.freq = {}
        self.lock = threading.Lock()
        self.min_freq = 0

    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        with self.lock:
            if key in self.cache:
                self.freq[key] += 1
                freq = self.freq[key]
                return self.cache[key]
            return None

    def put(self, key: str, value: Any):
        """放入缓存项"""
        with self.lock:
            if key in self.cache:
                self.cache[key] = value
                self.freq[key] += 1
            else:
                if len(self.cache) >= self.capacity:
                    min_freq_keys = [k for k, v in self.freq.items() if v == self.min_freq]
                    if min_freq_keys:
                        del self.cache[min_freq_keys[0]]
                        del self.freq[min_freq_keys[0]]
                self.cache[key] = value
                self.freq[key] = 1
                self.min_freq = 1

    def remove(self, key: str):
        """移除缓存项"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                del self.freq[key]

    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.freq.clear()
            self.min_freq = 0

    def __len__(self):
        return len(self.cache)


class FIFOCache:
    """
    FIFO (First In First Out) 缓存实现
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        with self.lock:
            if key in self.cache:
                return self.cache[key]
            return None

    def put(self, key: str, value: Any):
        """放入缓存项"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.capacity:
                    self.cache.popitem(last=False)
            self.cache[key] = value

    def remove(self, key: str):
        """移除缓存项"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]

    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()

    def __len__(self):
        return len(self.cache)


class CacheManager:
    """
    缓存管理器

    负责内存缓存和磁盘缓存的管理
    """

    def __init__(self, config: VisionCacheConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.metadata_file = self.cache_dir / config.cache_metadata_file

        if config.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if config.cache_policy == "lru":
            self.memory_cache = LRUCache(config.max_cache_size)
        elif config.cache_policy == "lfu":
            self.memory_cache = LFUCache(config.max_cache_size)
        elif config.cache_policy == "fifo":
            self.memory_cache = FIFOCache(config.max_cache_size)
        else:
            self.memory_cache = LRUCache(config.max_cache_size)

        self.metadata = self._load_metadata()
        self.lock = threading.Lock()

    def _load_metadata(self) -> Dict:
        """加载缓存元数据"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {"entries": {}, "stats": {"hits": 0, "misses": 0}}
        return {"entries": {}, "stats": {"hits": 0, "misses": 0}}

    def _save_metadata(self):
        """保存缓存元数据"""
        with self.lock:
            try:
                with open(self.metadata_file, 'w') as f:
                    json.dump(self.metadata, f, indent=2)
            except Exception as e:
                print(f"Failed to save metadata: {e}")

    def _compute_hash(self, identifier: str) -> str:
        """计算缓存键的哈希值"""
        return hashlib.sha256(identifier.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        prefix = cache_key[:2]
        return self.cache_dir / prefix / f"{cache_key}.{self.config.cache_format}"

    def get(self, identifier: str) -> Optional[torch.Tensor]:
        """
        获取缓存

        参数:
            identifier: 缓存标识符（通常是图像路径或哈希）

        返回:
            缓存的特征张量，如果不存在则返回 None
        """
        cache_key = self._compute_hash(identifier)

        if self.config.enable_memory_cache:
            cached = self.memory_cache.get(cache_key)
            if cached is not None:
                self.metadata["stats"]["hits"] += 1
                self._save_metadata()
                return cached

        if self.config.enable_disk_cache:
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                try:
                    if self.config.cache_format == "pt":
                        cached = torch.load(cache_path, map_location='cpu')
                    elif self.config.cache_format == "npz":
                        data = np.load(cache_path)
                        cached = torch.from_numpy(data['features'])
                    else:
                        cached = torch.load(cache_path, map_location='cpu')

                    if self.config.enable_memory_cache:
                        self.memory_cache.put(cache_key, cached)

                    self.metadata["stats"]["hits"] += 1
                    self.metadata["entries"][cache_key] = {
                        "identifier": identifier,
                        "path": str(cache_path),
                        "size": cached.numel() * cached.element_size()
                    }
                    self._save_metadata()
                    return cached
                except Exception as e:
                    print(f"Failed to load cache from {cache_path}: {e}")

        self.metadata["stats"]["misses"] += 1
        self._save_metadata()
        return None

    def put(
        self,
        identifier: str,
        features: torch.Tensor,
        metadata: Optional[Dict] = None
    ):
        """
        存储缓存

        参数:
            identifier: 缓存标识符
            features: 要缓存的特征张量
            metadata: 额外的元数据
        """
        cache_key = self._compute_hash(identifier)

        if self.config.enable_memory_cache:
            self.memory_cache.put(cache_key, features.clone())

        if self.config.enable_disk_cache:
            cache_path = self._get_cache_path(cache_key)
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                if self.config.cache_format == "pt":
                    torch.save(features, cache_path)
                elif self.config.cache_format == "npz":
                    np.save(cache_path, features.cpu().numpy())
                else:
                    torch.save(features, cache_path)

                self.metadata["entries"][cache_key] = {
                    "identifier": identifier,
                    "path": str(cache_path),
                    "size": features.numel() * features.element_size(),
                    "shape": list(features.shape),
                    "metadata": metadata or {}
                }
                self._save_metadata()
            except Exception as e:
                print(f"Failed to save cache to {cache_path}: {e}")

    def remove(self, identifier: str):
        """移除缓存项"""
        cache_key = self._compute_hash(identifier)

        self.memory_cache.remove(cache_key)

        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            cache_path.unlink()

        if cache_key in self.metadata["entries"]:
            del self.metadata["entries"][cache_key]
            self._save_metadata()

    def clear(self):
        """清空所有缓存"""
        self.memory_cache.clear()

        if self.cache_dir.exists():
            for item in self.cache_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

        self.metadata = {"entries": {}, "stats": {"hits": 0, "misses": 0}}
        self._save_metadata()

    def get_stats(self) -> Dict:
        """获取缓存统计信息"""
        total_size = sum(
            entry.get("size", 0)
            for entry in self.metadata["entries"].values()
        )

        return {
            "num_entries": len(self.metadata["entries"]),
            "memory_cache_size": len(self.memory_cache),
            "total_size_mb": total_size / (1024 * 1024),
            "hits": self.metadata["stats"]["hits"],
            "misses": self.metadata["stats"]["misses"],
            "hit_rate": (
                self.metadata["stats"]["hits"] /
                (self.metadata["stats"]["hits"] + self.metadata["stats"]["misses"])
                if self.metadata["stats"]["hits"] + self.metadata["stats"]["misses"] > 0
                else 0
            )
        }

    def list_cached(self) -> List[str]:
        """列出所有缓存的标识符"""
        return [
            entry["identifier"]
            for entry in self.metadata["entries"].values()
        ]

    def preload(self, identifiers: List[str]):
        """预加载指定的缓存项"""
        for identifier in identifiers:
            self.get(identifier)


class TileCacheManager:
    """
    瓦片级缓存管理器

    用于高分辨率图像的瓦片级缓存管理
    """

    def __init__(self, config: VisionCacheConfig):
        self.config = config
        self.tile_size = config.tile_size
        self.overlap = config.tile_overlap

        self.cache_dir = Path(config.cache_dir) / "tiles"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.tile_cache = LRUCache(config.max_cache_size)
        self.lock = threading.Lock()

    def _get_tile_key(
        self,
        image_path: str,
        tile_row: int,
        tile_col: int,
        scale: float = 1.0
    ) -> str:
        """生成瓦片缓存键"""
        base_key = f"{image_path}_{tile_row}_{tile_col}_{scale}"
        return hashlib.sha256(base_key.encode()).hexdigest()

    def _split_into_tiles(
        self,
        image: torch.Tensor
    ) -> List[Tuple[torch.Tensor, Tuple[int, int]]]:
        """
        将图像分割成瓦片

        参数:
            image: 输入图像 [B, C, H, W]

        返回:
            tiles: [(tile, (row, col)), ...]
        """
        B, C, H, W = image.shape
        stride = self.tile_size - self.overlap

        tiles = []
        for row in range(0, H - self.overlap, stride):
            for col in range(0, W - self.overlap, stride):
                tile = image[
                    :,
                    :,
                    row:min(row + self.tile_size, H),
                    col:min(col + self.tile_size, W)
                ]

                if tile.shape[2] < self.tile_size or tile.shape[3] < self.tile_size:
                    pad_h = self.tile_size - tile.shape[2]
                    pad_w = self.tile_size - tile.shape[3]
                    tile = F.pad(tile, (0, pad_w, 0, pad_h))

                tiles.append((tile, (row, col)))

        return tiles

    def get_tile(
        self,
        image_path: str,
        image: torch.Tensor,
        tile_row: int,
        tile_col: int,
        scale: float = 1.0
    ) -> Optional[torch.Tensor]:
        """
        获取单个瓦片（从缓存或原始图像）

        参数:
            image_path: 图像路径
            image: 原始图像张量
            tile_row: 瓦片行索引
            tile_col: 瓦片列索引
            scale: 缩放因子

        返回:
            tile: 瓦片张量
        """
        cache_key = self._get_tile_key(image_path, tile_row, tile_col, scale)

        cached = self.tile_cache.get(cache_key)
        if cached is not None:
            return cached

        tiles = self._split_into_tiles(image)

        for tile, (row, col) in tiles:
            if row == tile_row and col == tile_col:
                self.tile_cache.put(cache_key, tile.clone())
                return tile

        return None

    def cache_tiles(
        self,
        image_path: str,
        tiles: List[torch.Tensor],
        positions: List[Tuple[int, int]],
        scale: float = 1.0
    ):
        """
        缓存多个瓦片

        参数:
            image_path: 图像路径
            tiles: 瓦片列表
            positions: 瓦片位置列表
            scale: 缩放因子
        """
        for tile, (row, col) in zip(tiles, positions):
            cache_key = self._get_tile_key(image_path, row, col, scale)
            self.tile_cache.put(cache_key, tile.clone())

    def get_or_compute_tile(
        self,
        image_path: str,
        image: torch.Tensor,
        tile_row: int,
        tile_col: int,
        backbone: nn.Module,
        scale: float = 1.0
    ) -> torch.Tensor:
        """
        获取或计算瓦片特征

        参数:
            image_path: 图像路径
            image: 原始图像
            tile_row: 瓦片行索引
            tile_col: 瓦片列索引
            backbone: 特征提取网络
            scale: 缩放因子

        返回:
            tile_features: 瓦片特征
        """
        tile = self.get_tile(image_path, image, tile_row, tile_col, scale)
        if tile is None:
            return None

        with torch.no_grad():
            features = backbone(tile)
            if isinstance(features, dict):
                features = features['last_hidden_state']

        return features


class VisionCachePrecompute:
    """
    视觉缓存预计算器

    负责批量预计算和缓存视觉特征
    """

    def __init__(
        self,
        backbone: nn.Module,
        config: Optional[VisionCacheConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化预计算器

        参数:
            backbone: 视觉骨干网络
            config: 缓存配置
            device: 计算设备
        """
        self.backbone = backbone
        self.device = device
        self.config = config or VisionCacheConfig()

        self.cache_manager = CacheManager(self.config)
        self.tile_cache_manager = TileCacheManager(self.config)

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.transform = self._create_transform()

    def _create_transform(self) -> Callable:
        """创建图像预处理变换"""
        from torchvision import transforms

        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess_image(
        self,
        image: Union[str, Image.Image, torch.Tensor]
    ) -> torch.Tensor:
        """
        预处理图像

        参数:
            image: 图像（路径、PIL图像或张量）

        返回:
            处理后的图像张量
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if isinstance(image, Image.Image):
            image = self.transform(image)

        if image.dim() == 3:
            image = image.unsqueeze(0)

        return image.to(self.device)

    def encode_image(
        self,
        image: Union[str, Image.Image, torch.Tensor],
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        编码图像为特征向量

        参数:
            image: 输入图像
            use_cache: 是否使用缓存

        返回:
            features: 图像特征
        """
        if isinstance(image, str):
            identifier = image
        else:
            identifier = hashlib.sha256(
                str(image).encode()
            ).hexdigest()

        if use_cache:
            cached = self.cache_manager.get(identifier)
            if cached is not None:
                return cached

        processed = self.preprocess_image(image)

        with torch.no_grad():
            features = self.backbone(processed)
            if isinstance(features, dict):
                features = features['pooler_output'] or features['last_hidden_state'][:, 0]

        features = features.cpu()

        if use_cache:
            self.cache_manager.put(identifier, features)

        return features

    def precompute(
        self,
        image_paths: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        parallel: bool = True
    ):
        """
        批量预计算图像特征

        参数:
            image_paths: 图像路径列表
            batch_size: 批次大小
            show_progress: 是否显示进度
            parallel: 是否并行计算
        """
        batch_size = batch_size or self.config.precompute_batch_size

        if parallel and self.config.num_workers > 1:
            self._precompute_parallel(image_paths, batch_size, show_progress)
        else:
            self._precompute_sequential(image_paths, batch_size, show_progress)

    def _precompute_sequential(
        self,
        image_paths: List[str],
        batch_size: int,
        show_progress: bool
    ):
        """顺序预计算"""
        total = len(image_paths)
        for i in range(0, total, batch_size):
            batch_paths = image_paths[i:i + batch_size]

            for path in batch_paths:
                self.encode_image(path, use_cache=True)

            if show_progress:
                progress = min(i + batch_size, total) / total * 100
                print(f"Progress: {progress:.1f}% ({min(i + batch_size, total)}/{total})")

    def _precompute_parallel(
        self,
        image_paths: List[str],
        batch_size: int,
        show_progress: bool
    ):
        """并行预计算"""
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = []

            for path in image_paths:
                future = executor.submit(self.encode_image, path, True)
                futures.append(future)

            completed = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing image: {e}")

                completed += 1
                if show_progress:
                    progress = completed / len(image_paths) * 100
                    print(f"Progress: {progress:.1f}% ({completed}/{len(image_paths)})")

    def precompute_tiles(
        self,
        image_path: str,
        backbone: Optional[nn.Module] = None
    ) -> List[torch.Tensor]:
        """
        预计算图像瓦片特征

        参数:
            image_path: 图像路径
            backbone: 可选的骨干网络

        返回:
            tile_features: 瓦片特征列表
        """
        backbone = backbone or self.backbone
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        tiles, positions = self.tile_cache_manager._split_into_tiles(image_tensor)

        tile_features = []
        with torch.no_grad():
            for tile in tiles:
                features = backbone(tile)
                if isinstance(features, dict):
                    features = features['last_hidden_state']
                tile_features.append(features.cpu())

        self.tile_cache_manager.cache_tiles(
            image_path,
            tiles,
            positions,
            scale=1.0
        )

        return tile_features

    def get_cached_features(
        self,
        identifier: str
    ) -> Optional[torch.Tensor]:
        """
        获取缓存的特征

        参数:
            identifier: 缓存标识符

        返回:
            cached_features: 缓存的特征张量
        """
        return self.cache_manager.get(identifier)

    def clear_cache(self):
        """清空所有缓存"""
        self.cache_manager.clear()

    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        return self.cache_manager.get_stats()

    def export_cache(self, export_path: str):
        """
        导出缓存到指定路径

        参数:
            export_path: 导出路径
        """
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)

        metadata = self.cache_manager.metadata

        with open(export_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        cache_entries = metadata.get("entries", {})
        for cache_key, entry in cache_entries.items():
            src_path = Path(entry["path"])
            if src_path.exists():
                dst_path = export_dir / f"{cache_key}.{self.config.cache_format}"
                shutil.copy2(src_path, dst_path)

    def import_cache(self, import_path: str):
        """
        从指定路径导入缓存

        参数:
            import_path: 导入路径
        """
        import_dir = Path(import_path)

        metadata_file = import_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            for cache_key, entry in metadata.get("entries", {}).items():
                src_path = import_dir / f"{cache_key}.{self.config.cache_format}"
                if src_path.exists():
                    dst_path = self.cache_manager._get_cache_path(cache_key)
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dst_path)

                    self.cache_manager.metadata["entries"][cache_key] = entry

            self.cache_manager._save_metadata()


class AsyncCachePrecompute:
    """
    异步缓存预计算器

    支持异步预计算和增量更新
    """

    def __init__(
        self,
        precompute: VisionCachePrecompute,
        max_queue_size: int = 100
    ):
        self.precompute = precompute
        self.task_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue()
        self.worker_thread = None
        self.running = False

    def start(self):
        """启动异步工作线程"""
        if self.running:
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def stop(self):
        """停止异步工作线程"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)

    def _worker_loop(self):
        """异步工作线程主循环"""
        while self.running:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:
                    break

                image_path = task
                features = self.precompute.encode_image(image_path, use_cache=True)
                self.result_queue.put((image_path, features))

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Async precompute error: {e}")

    def submit(self, image_path: str):
        """
        提交预计算任务

        参数:
            image_path: 图像路径
        """
        self.task_queue.put(image_path)

    def get_result(self, timeout: float = None) -> Tuple[str, torch.Tensor]:
        """
        获取预计算结果

        参数:
            timeout: 超时时间

        返回:
            (image_path, features): 图像路径和特征
        """
        return self.result_queue.get(timeout=timeout)

    def wait_all(self):
        """等待所有任务完成"""
        self.task_queue.join()


def create_cache_precompute(
    backbone: nn.Module,
    cache_dir: str = "vision_cache",
    device: str = None,
    **kwargs
) -> VisionCachePrecompute:
    """
    创建缓存预计算器的工厂函数

    参数:
        backbone: 视觉骨干网络
        cache_dir: 缓存目录
        device: 计算设备
        **kwargs: 其他配置参数

    返回:
        VisionCachePrecompute 实例
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = VisionCacheConfig(
        cache_dir=cache_dir,
        **kwargs
    )

    return VisionCachePrecompute(backbone, config, device)
