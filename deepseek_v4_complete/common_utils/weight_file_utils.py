"""
模型权重保存加载工具模块

提供完整的模型权重保存和加载功能，支持：
- PyTorch 模型权重保存/加载
- SafeTensors 格式支持
- 分布式训练权重处理
- 权重分片保存
- 检查点管理
- 权重兼容性检查
"""

import os
import sys
import json
import shutil
import hashlib
import warnings
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from copy import deepcopy
from contextlib import contextmanager


try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from safetensors.torch import save_file as safe_save_file
    from safetensors.torch import load_file as safe_load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


@dataclass
class CheckpointMetadata:
    """检查点元数据"""
    model_name: str
    version: str
    created_at: str
    epoch: Optional[int] = None
    global_step: Optional[int] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    device: Optional[str] = None
    torch_version: Optional[str] = None
    file_hash: Optional[str] = None
    file_size: Optional[int] = None


class WeightFileUtils:
    """
    权重文件工具类

    提供统一的权重保存和加载接口
    """

    @staticmethod
    def save_weights(
        model: Any,
        path: str,
        save_dtype: Optional[str] = None,
        use_safetensors: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        保存模型权重

        Args:
            model: PyTorch 模型
            path: 保存路径
            save_dtype: 保存时的数据类型（如 'float16', 'float32'）
            use_safetensors: 是否使用 SafeTensors 格式
            metadata: 额外的元数据

        Returns:
            保存的文件路径
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 未安装，无法保存权重")

        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        state_dict = model.state_dict()

        if save_dtype:
            state_dict = WeightFileUtils._convert_dtype(state_dict, save_dtype)

        if use_safetensors and SAFETENSORS_AVAILABLE:
            return WeightFileUtils._save_safetensors(state_dict, path, metadata)
        else:
            return WeightFileUtils._save_torch(state_dict, path, metadata)

    @staticmethod
    def _save_torch(
        state_dict: Dict[str, Any],
        path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """使用 PyTorch 格式保存"""
        if metadata:
            checkpoint = {
                'model_state_dict': state_dict,
                'metadata': metadata,
                'saved_at': datetime.now().isoformat()
            }
            torch.save(checkpoint, path)
        else:
            torch.save(state_dict, path)

        return path

    @staticmethod
    def _save_safetensors(
        state_dict: Dict[str, Any],
        path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """使用 SafeTensors 格式保存"""
        safe_metadata = metadata.copy() if metadata else {}
        safe_metadata['saved_at'] = datetime.now().isoformat()

        safe_save_file(state_dict, path, safe_metadata)
        return path

    @staticmethod
    def load_weights(
        model: Any,
        path: str,
        device: Optional[str] = None,
        strict: bool = True,
        load_to_cpu: bool = False
    ) -> Tuple[Any, Optional[CheckpointMetadata]]:
        """
        加载模型权重

        Args:
            model: PyTorch 模型
            path: 权重文件路径
            device: 加载到的设备
            strict: 是否严格匹配权重键
            load_to_cpu: 是否强制加载到 CPU

        Returns:
            (模型, 检查点元数据)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 未安装，无法加载权重")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if load_to_cpu:
            device = "cpu"

        metadata = None

        if path.endswith('.safetensors') and SAFETENSORS_AVAILABLE:
            state_dict = safe_load_file(path, device=device)
        else:
            checkpoint = torch.load(path, map_location=device)

            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    metadata = checkpoint.get('metadata')
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    metadata = checkpoint.get('metadata')
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

        model.load_state_dict(state_dict, strict=strict)
        return model, metadata

    @staticmethod
    def _convert_dtype(state_dict: Dict[str, Any], dtype: str) -> Dict[str, Any]:
        """转换权重数据类型"""
        dtype_map = {
            'float16': torch.float16,
            'half': torch.float16,
            'fp16': torch.float16,
            'float32': torch.float32,
            'float': torch.float32,
            'fp32': torch.float32,
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
            'int8': torch.int8,
            'int32': torch.int32,
            'int64': torch.int64,
        }

        target_dtype = dtype_map.get(dtype.lower())
        if target_dtype is None:
            raise ValueError(f"不支持的数据类型: {dtype}")

        return {k: v.to(target_dtype) for k, v in state_dict.items()}

    @staticmethod
    def load_partial_weights(
        model: Any,
        path: str,
        device: Optional[str] = None,
        skip_mismatch: bool = True
    ) -> Tuple[Any, List[str], List[str]]:
        """
        加载部分权重（用于预训练模型微调）

        Args:
            model: PyTorch 模型
            path: 权重文件路径
            device: 加载到的设备
            skip_mismatch: 是否跳过不匹配的权重

        Returns:
            (模型, 加载的键列表, 跳过的键列表)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 未安装")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model_state = model.state_dict()

        if path.endswith('.safetensors') and SAFETENSORS_AVAILABLE:
            loaded_state = safe_load_file(path, device=device)
        else:
            checkpoint = torch.load(path, map_location=device)
            if isinstance(checkpoint, dict):
                loaded_state = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            else:
                loaded_state = checkpoint

        loaded_keys = []
        skipped_keys = []

        for key in loaded_state:
            if key in model_state:
                try:
                    model_state[key] = loaded_state[key]
                    loaded_keys.append(key)
                except Exception:
                    skipped_keys.append(key)
            else:
                if not skip_mismatch:
                    warnings.warn(f"权重键不存在于模型中: {key}")
                skipped_keys.append(key)

        model.load_state_dict(model_state, strict=False)
        return model, loaded_keys, skipped_keys


class CheckpointManager:
    """
    检查点管理器

    管理模型的检查点保存和恢复
    """

    def __init__(
        self,
        save_dir: str,
        max_checkpoints: int = 5,
        save_fn: Optional[Callable] = None,
        load_fn: Optional[Callable] = None
    ):
        """
        初始化检查点管理器

        Args:
            save_dir: 保存目录
            max_checkpoints: 最大保留的检查点数量
            save_fn: 自定义保存函数
            load_fn: 自定义加载函数
        """
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self._save_fn = save_fn or WeightFileUtils.save_weights
        self._load_fn = load_fn or WeightFileUtils.load_weights
        os.makedirs(save_dir, exist_ok=True)

    def save(
        self,
        model: Any,
        epoch: int,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None
    ) -> str:
        """
        保存检查点

        Args:
            model: PyTorch 模型
            epoch: 当前轮次
            step: 当前步数
            metrics: 当前指标
            metadata: 额外元数据
            filename: 自定义文件名

        Returns:
            保存的文件路径
        """
        if filename is None:
            filename = f"checkpoint_epoch{epoch}_step{step}.pt"

        filepath = os.path.join(self.save_dir, filename)

        full_metadata = {
            'epoch': epoch,
            'global_step': step,
            'metrics': metrics or {},
            'config': metadata or {},
            'model_name': 'model',
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'file_hash': None,
            'file_size': None
        }

        if TORCH_AVAILABLE:
            full_metadata['torch_version'] = torch.__version__
            full_metadata['device'] = str(next(model.parameters(), None))

        self._save_fn(model, filepath, metadata=full_metadata)

        self._update_checkpoint_list(epoch, step, filepath, metrics)
        self._cleanup_old_checkpoints()

        return filepath

    def load(
        self,
        model: Any,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False
    ) -> Tuple[Any, CheckpointMetadata]:
        """
        加载检查点

        Args:
            model: PyTorch 模型
            checkpoint_path: 检查点路径，None 表示加载最新的
            load_best: 是否加载最佳模型

        Returns:
            (模型, 检查点元数据)
        """
        if checkpoint_path is None and load_best:
            checkpoint_path = self.get_best_checkpoint()
        elif checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()

        if checkpoint_path is None:
            raise FileNotFoundError("未找到检查点文件")

        model, metadata = self._load_fn(model, checkpoint_path)

        if isinstance(metadata, dict):
            checkpoint_meta = CheckpointMetadata(
                model_name=metadata.get('model_name', 'unknown'),
                version=metadata.get('version', '1.0'),
                created_at=metadata.get('created_at', ''),
                epoch=metadata.get('epoch'),
                global_step=metadata.get('global_step'),
                metrics=metadata.get('metrics', {}),
                config=metadata.get('config', {}),
                device=metadata.get('device'),
                torch_version=metadata.get('torch_version'),
                file_hash=metadata.get('file_hash'),
                file_size=metadata.get('file_size')
            )
        else:
            checkpoint_meta = CheckpointMetadata(
                model_name='unknown',
                version='1.0',
                created_at='',
                epoch=None,
                global_step=None
            )

        return model, checkpoint_meta

    def _update_checkpoint_list(
        self,
        epoch: int,
        step: int,
        filepath: str,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """更新检查点列表"""
        list_path = os.path.join(self.save_dir, 'checkpoints.json')

        try:
            if os.path.exists(list_path):
                with open(list_path, 'r') as f:
                    checkpoint_list = json.load(f)
            else:
                checkpoint_list = []
        except (json.JSONDecodeError, FileNotFoundError):
            checkpoint_list = []

        checkpoint_list.append({
            'filepath': filepath,
            'epoch': epoch,
            'step': step,
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat()
        })

        checkpoint_list.sort(key=lambda x: x['timestamp'], reverse=True)

        with open(list_path, 'w') as f:
            json.dump(checkpoint_list, f, indent=2)

    def _cleanup_old_checkpoints(self) -> None:
        """清理旧的检查点"""
        list_path = os.path.join(self.save_dir, 'checkpoints.json')

        try:
            if os.path.exists(list_path):
                with open(list_path, 'r') as f:
                    checkpoint_list = json.load(f)
            else:
                return
        except (json.JSONDecodeError, FileNotFoundError):
            return

        if len(checkpoint_list) <= self.max_checkpoints:
            return

        to_remove = checkpoint_list[self.max_checkpoints:]
        checkpoint_list = checkpoint_list[:self.max_checkpoints]

        for checkpoint in to_remove:
            filepath = checkpoint['filepath']
            safetensors_path = filepath.replace('.pt', '.safetensors')

            if os.path.exists(filepath):
                os.remove(filepath)
            if os.path.exists(safetensors_path):
                os.remove(safetensors_path)

        with open(list_path, 'w') as f:
            json.dump(checkpoint_list, f, indent=2)

    def get_latest_checkpoint(self) -> Optional[str]:
        """获取最新的检查点路径"""
        list_path = os.path.join(self.save_dir, 'checkpoints.json')

        try:
            if os.path.exists(list_path):
                with open(list_path, 'r') as f:
                    checkpoint_list = json.load(f)
                if checkpoint_list:
                    return checkpoint_list[0]['filepath']
        except (json.JSONDecodeError, FileNotFoundError):
            pass

        checkpoints = sorted(
            Path(self.save_dir).glob('checkpoint_*.pt'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        return str(checkpoints[0]) if checkpoints else None

    def get_best_checkpoint(self) -> Optional[str]:
        """获取最佳检查点路径（基于验证损失）"""
        list_path = os.path.join(self.save_dir, 'checkpoints.json')

        try:
            if os.path.exists(list_path):
                with open(list_path, 'r') as f:
                    checkpoint_list = json.load(f)

                best_checkpoint = None
                best_loss = float('inf')

                for checkpoint in checkpoint_list:
                    val_loss = checkpoint.get('metrics', {}).get('val_loss', float('inf'))
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_checkpoint = checkpoint['filepath']

                return best_checkpoint
        except (json.JSONDecodeError, FileNotFoundError):
            pass

        return None

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """列出所有检查点"""
        list_path = os.path.join(self.save_dir, 'checkpoints.json')

        try:
            if os.path.exists(list_path):
                with open(list_path, 'r') as f:
                    return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass

        checkpoints = list(Path(self.save_dir).glob('checkpoint_*.pt'))
        return [{'filepath': str(p), 'filename': p.name} for p in checkpoints]


class WeightSharding:
    """
    权重分片工具

    用于大模型的权重分片保存和加载
    """

    @staticmethod
    def save_sharded(
        model: Any,
        save_dir: str,
        max_shard_size: int = 5 * 1024 * 1024 * 1024,
        save_fn: Optional[Callable] = None
    ) -> List[str]:
        """
        分片保存模型权重

        Args:
            model: PyTorch 模型
            save_dir: 保存目录
            max_shard_size: 每个分片的最大大小（字节）
            save_fn: 保存函数

        Returns:
            分片文件路径列表
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 未安装")

        os.makedirs(save_dir, exist_ok=True)

        state_dict = model.state_dict()
        shards = []
        current_shard = {}
        current_size = 0
        shard_index = 0

        for key, tensor in state_dict.items():
            tensor_size = tensor.nelement() * tensor.element_size()

            if current_size + tensor_size > max_shard_size and current_shard:
                shard_path = os.path.join(save_dir, f"model_shard_{shard_index}.pt")
                torch.save(current_shard, shard_path)
                shards.append(shard_path)
                current_shard = {}
                current_size = 0
                shard_index += 1

            current_shard[key] = tensor
            current_size += tensor_size

        if current_shard:
            shard_path = os.path.join(save_dir, f"model_shard_{shard_index}.pt")
            torch.save(current_shard, shard_path)
            shards.append(shard_path)

        index_data = {
            'metadata': {
                'total_size': sum(
                    os.path.getsize(s) for s in shards
                ),
                'num_shards': len(shards)
            },
            'weight_map': {}
        }

        shard_index = 0
        for key in state_dict:
            tensor_size = state_dict[key].nelement() * state_dict[key].element_size()
            cumulative_size = sum(
                os.path.getsize(shards[i]) for i in range(shard_index + 1)
            )

            if cumulative_size > max_shard_size:
                shard_index += 1

            index_data['weight_map'][key] = f"model_shard_{shard_index}.pt"

        index_path = os.path.join(save_dir, "model.safetensors.index.json")
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)

        return shards

    @staticmethod
    def load_sharded(
        model: Any,
        save_dir: str,
        device: Optional[str] = None
    ) -> Any:
        """
        加载分片模型权重

        Args:
            model: PyTorch 模型
            save_dir: 分片保存目录
            device: 加载设备

        Returns:
            加载了权重的模型
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 未安装")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        index_path = os.path.join(save_dir, "model.safetensors.index.json")

        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                index_data = json.load(f)

            loaded_state = {}
            for key, shard_file in index_data['weight_map'].items():
                if shard_file not in loaded_state:
                    shard_path = os.path.join(save_dir, shard_file)
                    if shard_path.endswith('.safetensors') and SAFETENSORS_AVAILABLE:
                        loaded_state[shard_file] = safe_load_file(shard_path, device=device)
                    else:
                        shard_data = torch.load(shard_path, map_location=device)
                        loaded_state[shard_file] = shard_data

            state_dict = {}
            for key, shard_file in index_data['weight_map'].items():
                state_dict[key] = loaded_state[shard_file][key]

        else:
            shard_files = sorted(Path(save_dir).glob("model_shard_*.pt"))
            state_dict = {}

            for shard_file in shard_files:
                shard_data = torch.load(shard_file, map_location=device)
                if isinstance(shard_data, dict):
                    state_dict.update(shard_data)

        model.load_state_dict(state_dict, strict=False)
        return model


def save_weights(
    model: Any,
    path: str,
    use_safetensors: bool = False,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    便捷函数：保存模型权重

    Args:
        model: PyTorch 模型
        path: 保存路径
        use_safetensors: 是否使用 SafeTensors 格式
        metadata: 额外元数据

    Returns:
        保存的文件路径
    """
    return WeightFileUtils.save_weights(model, path, use_safetensors=use_safetensors, metadata=metadata)


def load_weights(
    model: Any,
    path: str,
    device: Optional[str] = None,
    strict: bool = True
) -> Tuple[Any, Optional[CheckpointMetadata]]:
    """
    便捷函数：加载模型权重

    Args:
        model: PyTorch 模型
        path: 权重文件路径
        device: 加载到的设备
        strict: 是否严格匹配权重键

    Returns:
        (模型, 检查点元数据)
    """
    return WeightFileUtils.load_weights(model, path, device=device, strict=strict)


def compute_file_hash(filepath: str, algorithm: str = 'sha256') -> str:
    """
    计算文件的哈希值

    Args:
        filepath: 文件路径
        algorithm: 哈希算法（'md5', 'sha1', 'sha256'）

    Returns:
        文件哈希值
    """
    hash_obj = hashlib.new(algorithm)

    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_obj.update(chunk)

    return hash_obj.hexdigest()


def verify_weights_integrity(
    filepath: str,
    expected_hash: str,
    algorithm: str = 'sha256'
) -> bool:
    """
    验证权重文件完整性

    Args:
        filepath: 权重文件路径
        expected_hash: 期望的哈希值
        algorithm: 哈希算法

    Returns:
        是否验证通过
    """
    actual_hash = compute_file_hash(filepath, algorithm)
    return actual_hash == expected_hash


def get_model_size(model: Any) -> int:
    """
    获取模型的参数量和大小

    Args:
        model: PyTorch 模型

    Returns:
        模型大小（字节）
    """
    if not TORCH_AVAILABLE:
        return 0

    state_dict = model.state_dict()
    total_size = sum(t.nelement() * t.element_size() for t in state_dict.values())
    return total_size


def print_model_size(model: Any) -> None:
    """
    打印模型大小信息

    Args:
        model: PyTorch 模型
    """
    if not TORCH_AVAILABLE:
        print("PyTorch 未安装")
        return

    state_dict = model.state_dict()
    total_size = 0
    param_info = []

    for key, tensor in state_dict.items():
        size = tensor.nelement() * tensor.element_size()
        total_size += size
        param_info.append({
            'name': key,
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'size_mb': size / (1024 ** 2)
        })

    param_info.sort(key=lambda x: x['size_mb'], reverse=True)

    print(f"模型总大小: {total_size / (1024 ** 3):.2f} GB")
    print(f"参数总数: {sum(p.numel() for p in model.parameters())}")
    print(f"层数: {len(state_dict)}")
    print("\n最大的10个参数:")
    for info in param_info[:10]:
        print(f"  {info['name']}: {info['shape']} - {info['size_mb']:.2f} MB")
