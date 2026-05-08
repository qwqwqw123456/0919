"""
数据备份恢复模块
提供完整的数据备份、恢复、导出、导入功能
支持 MySQL 数据库、Redis 缓存、向量数据库的备份
"""

import os
import json
import shutil
import tarfile
import zipfile
import hashlib
import tempfile
import subprocess
from typing import Optional, List, Dict, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
from pathlib import Path
import logging

from mysql_connect import MySQLClient, DatabaseConfig
from redis_cache import RedisCache, RedisConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BackupStatus(Enum):
    """备份状态枚举"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RESTORED = "restored"


class BackupType(Enum):
    """备份类型枚举"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class CompressionType(Enum):
    """压缩类型枚举"""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    XZ = "xz"
    ZIP = "zip"


@dataclass
class BackupMetadata:
    """备份元数据"""
    backup_id: str
    backup_name: str
    backup_type: str
    compression: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    file_path: Optional[str] = None
    file_size: int = 0
    checksum: Optional[str] = None
    tables: List[str] = field(default_factory=list)
    record_counts: Dict[str, int] = field(default_factory=dict)
    error_message: Optional[str] = None
    mysql_version: Optional[str] = None
    redis_data_size: int = 0
    vector_data_size: int = 0
    metadata_size: int = 0
    total_size: int = 0
    includes_mysql: bool = True
    includes_redis: bool = False
    includes_vectors: bool = False
    retention_days: int = 30

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "backup_id": self.backup_id,
            "backup_name": self.backup_name,
            "backup_type": self.backup_type,
            "compression": self.compression,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "checksum": self.checksum,
            "tables": self.tables,
            "record_counts": self.record_counts,
            "error_message": self.error_message,
            "mysql_version": self.mysql_version,
            "redis_data_size": self.redis_data_size,
            "vector_data_size": self.vector_data_size,
            "metadata_size": self.metadata_size,
            "total_size": self.total_size,
            "includes_mysql": self.includes_mysql,
            "includes_redis": self.includes_redis,
            "includes_vectors": self.includes_vectors,
            "retention_days": self.retention_days
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackupMetadata":
        """从字典创建"""
        data = data.copy()
        if "start_time" in data and data["start_time"]:
            data["start_time"] = datetime.fromisoformat(data["start_time"])
        if "end_time" in data and data["end_time"]:
            data["end_time"] = datetime.fromisoformat(data["end_time"])
        return cls(**data)


@dataclass
class RestoreMetadata:
    """恢复元数据"""
    restore_id: str
    backup_id: str
    restore_type: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    tables_restored: List[str] = field(default_factory=list)
    records_restored: Dict[str, int] = field(default_factory=dict)
    error_message: Optional[str] = None
    tables_failed: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "restore_id": self.restore_id,
            "backup_id": self.backup_id,
            "restore_type": self.restore_type,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "tables_restored": self.tables_restored,
            "records_restored": self.records_restored,
            "error_message": self.error_message,
            "tables_failed": self.tables_failed
        }


class BackupConfig:
    """备份配置"""

    def __init__(
        self,
        backup_dir: str = "./backups",
        compression: str = "gzip",
        includes_mysql: bool = True,
        includes_redis: bool = False,
        includes_vectors: bool = False,
        retention_days: int = 30,
        max_backups: int = 10,
        parallel_workers: int = 4,
        chunk_size: int = 10000,
        exclude_tables: Optional[List[str]] = None,
        include_only_tables: Optional[List[str]] = None,
        compress_level: int = 6
    ):
        self.backup_dir = backup_dir
        self.compression = compression
        self.includes_mysql = includes_mysql
        self.includes_redis = includes_redis
        self.includes_vectors = includes_vectors
        self.retention_days = retention_days
        self.max_backups = max_backups
        self.parallel_workers = parallel_workers
        self.chunk_size = chunk_size
        self.exclude_tables = exclude_tables or []
        self.include_only_tables = include_only_tables
        self.compress_level = compress_level


class BackupRestore:
    """
    数据备份恢复管理器
    提供完整的数据库备份和恢复功能
    """

    def __init__(
        self,
        mysql_client: Optional[MySQLClient] = None,
        redis_cache: Optional[RedisCache] = None,
        config: Optional[BackupConfig] = None
    ):
        """
        初始化备份恢复管理器

        Args:
            mysql_client: MySQL 客户端
            redis_cache: Redis 缓存客户端
            config: 备份配置
        """
        self.mysql_client = mysql_client
        self.redis_cache = redis_cache
        self.config = config or BackupConfig()
        self._ensure_backup_dir()

    def _ensure_backup_dir(self):
        """确保备份目录存在"""
        os.makedirs(self.config.backup_dir, exist_ok=True)

    def _generate_backup_id(self) -> str:
        """生成备份ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"backup_{timestamp}"

    def _calculate_checksum(self, file_path: str) -> str:
        """计算文件校验和"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _compress_file(
        self,
        source_path: str,
        target_path: str,
        compression: str = "gzip"
    ) -> str:
        """
        压缩文件

        Args:
            source_path: 源文件路径
            target_path: 目标文件路径
            compression: 压缩类型

        Returns:
            str: 压缩后的文件路径
        """
        if compression == "gzip" or compression == "gz":
            import gzip
            with open(source_path, "rb") as f_in:
                with gzip.open(target_path, "wb", compresslevel=self.config.compress_level) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return target_path

        elif compression == "bzip2" or compression == "bz2":
            import bz2
            with open(source_path, "rb") as f_in:
                with bz2.open(target_path, "wb", compresslevel=self.config.compress_level) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return target_path

        elif compression == "xz":
            import lzma
            with open(source_path, "rb") as f_in:
                with lzma.open(target_path, "wb", compresslevel=self.config.compress_level) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return target_path

        elif compression == "zip":
            with zipfile.ZipFile(target_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(source_path, os.path.basename(source_path))
            return target_path

        else:
            shutil.copy2(source_path, target_path)
            return target_path

    def _decompress_file(
        self,
        source_path: str,
        target_path: str,
        compression: str = "gzip"
    ) -> str:
        """
        解压文件

        Args:
            source_path: 源文件路径
            target_path: 目标文件路径
            compression: 压缩类型

        Returns:
            str: 解压后的文件路径
        """
        if compression == "gzip" or compression == "gz":
            import gzip
            with gzip.open(source_path, "rb") as f_in:
                with open(target_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return target_path

        elif compression == "bzip2" or compression == "bz2":
            import bz2
            with bz2.open(source_path, "rb") as f_in:
                with open(target_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return target_path

        elif compression == "xz":
            import lzma
            with lzma.open(source_path, "rb") as f_in:
                with open(target_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return target_path

        elif compression == "zip":
            with zipfile.ZipFile(source_path, "r") as zf:
                zf.extractall(os.path.dirname(target_path))
                return os.path.join(os.path.dirname(target_path), zf.namelist()[0])
            return target_path

        else:
            shutil.copy2(source_path, target_path)
            return target_path

    def backup_mysql(
        self,
        backup_id: str,
        tables: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        备份 MySQL 数据库

        Args:
            backup_id: 备份ID
            tables: 要备份的表列表，None 表示全部

        Returns:
            Dict: 备份结果
        """
        if not self.mysql_client:
            return {"success": False, "error": "MySQL client not configured"}

        try:
            backup_dir = os.path.join(self.config.backup_dir, backup_id)
            os.makedirs(backup_dir, exist_ok=True)

            all_tables = self.mysql_client.get_table_names()

            if self.config.include_only_tables:
                tables = [t for t in all_tables if t in self.config.include_only_tables]
            elif tables:
                tables = [t for t in tables if t not in self.config.exclude_tables]
            else:
                tables = [t for t in all_tables if t not in self.config.exclude_tables]

            record_counts = {}
            table_schemas = {}

            for table in tables:
                schema_file = os.path.join(backup_dir, f"{table}_schema.sql")
                data_file = os.path.join(backup_dir, f"{table}_data.json")

                columns = self.mysql_client.get_table_columns(table)
                table_schemas[table] = columns

                with open(schema_file, "w", encoding="utf-8") as f:
                    f.write(f"-- Schema for table: {table}\n")
                    f.write(f"-- Generated at: {datetime.now().isoformat()}\n\n")

                records = self.mysql_client.select(table, limit=100000)
                record_counts[table] = len(records)

                with open(data_file, "w", encoding="utf-8") as f:
                    json.dump(records, f, ensure_ascii=False, default=str)

                logger.info(f"Backed up table: {table} ({record_counts[table]} records)")

            metadata = {
                "backup_id": backup_id,
                "tables": tables,
                "record_counts": record_counts,
                "schemas": table_schemas,
                "backup_time": datetime.now().isoformat(),
                "mysql_version": self.mysql_client.execute_scalar("SELECT VERSION()")
            }

            metadata_file = os.path.join(backup_dir, "backup_metadata.json")
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            return {
                "success": True,
                "backup_dir": backup_dir,
                "tables": tables,
                "record_counts": record_counts
            }

        except Exception as e:
            logger.error(f"MySQL backup failed: {e}")
            return {"success": False, "error": str(e)}

    def backup_redis(self, backup_id: str) -> Dict[str, Any]:
        """
        备份 Redis 数据

        Args:
            backup_id: 备份ID

        Returns:
            Dict: 备份结果
        """
        if not self.redis_cache:
            return {"success": False, "error": "Redis client not configured"}

        try:
            backup_dir = os.path.join(self.config.backup_dir, backup_id)
            os.makedirs(backup_dir, exist_ok=True)

            redis_file = os.path.join(backup_dir, "redis_data.json")
            keys = self.redis_cache.scan(count=10000)

            all_data = {}
            for key in keys:
                value = self.redis_cache.get(key)
                ttl = self.redis_cache.ttl(key)
                all_data[key] = {
                    "value": value,
                    "ttl": ttl if ttl > 0 else None
                }

            with open(redis_file, "w", encoding="utf-8") as f:
                json.dump(all_data, f, ensure_ascii=False, default=str)

            info = self.redis_cache.get_info()
            data_size = os.path.getsize(redis_file)

            logger.info(f"Backed up Redis: {len(keys)} keys, {data_size} bytes")

            return {
                "success": True,
                "keys_count": len(keys),
                "data_size": data_size,
                "memory_used": info.get("used_memory_human", "N/A")
            }

        except Exception as e:
            logger.error(f"Redis backup failed: {e}")
            return {"success": False, "error": str(e)}

    def backup_vectors(self, backup_id: str, vector_index_path: str) -> Dict[str, Any]:
        """
        备份向量数据库

        Args:
            backup_id: 备份ID
            vector_index_path: 向量索引文件路径

        Returns:
            Dict: 备份结果
        """
        try:
            backup_dir = os.path.join(self.config.backup_dir, backup_id)
            os.makedirs(backup_dir, exist_ok=True)

            target_path = os.path.join(backup_dir, "vector_index")

            if os.path.exists(vector_index_path):
                shutil.copy2(vector_index_path, target_path)
                data_size = os.path.getsize(target_path)
                logger.info(f"Backed up vector index: {data_size} bytes")
                return {
                    "success": True,
                    "data_size": data_size
                }
            else:
                return {"success": False, "error": "Vector index file not found"}

        except Exception as e:
            logger.error(f"Vector backup failed: {e}")
            return {"success": False, "error": str(e)}

    def create_backup(
        self,
        name: str = None,
        backup_type: str = "full",
        tables: Optional[List[str]] = None,
        vector_index_path: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> BackupMetadata:
        """
        创建完整备份

        Args:
            name: 备份名称
            backup_type: 备份类型
            tables: 要备份的表
            vector_index_path: 向量索引路径
            progress_callback: 进度回调函数

        Returns:
            BackupMetadata: 备份元数据
        """
        backup_id = self._generate_backup_id()
        name = name or backup_id

        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_name=name,
            backup_type=backup_type,
            compression=self.config.compression,
            status=BackupStatus.IN_PROGRESS.value,
            start_time=datetime.now(),
            includes_mysql=self.config.includes_mysql,
            includes_redis=self.config.includes_redis,
            includes_vectors=self.config.includes_vectors,
            retention_days=self.config.retention_days
        )

        try:
            if progress_callback:
                progress_callback("Starting backup", 0.0)

            if self.config.includes_mysql:
                if progress_callback:
                    progress_callback("Backing up MySQL", 0.1)
                result = self.backup_mysql(backup_id, tables)
                if result.get("success"):
                    metadata.tables = result.get("tables", [])
                    metadata.record_counts = result.get("record_counts", {})
                    metadata.mysql_version = result.get("mysql_version")

            if self.config.includes_redis:
                if progress_callback:
                    progress_callback("Backing up Redis", 0.4)
                result = self.backup_redis(backup_id)
                if result.get("success"):
                    metadata.redis_data_size = result.get("data_size", 0)

            if self.config.includes_vectors and vector_index_path:
                if progress_callback:
                    progress_callback("Backing up vectors", 0.7)
                result = self.backup_vectors(backup_id, vector_index_path)
                if result.get("success"):
                    metadata.vector_data_size = result.get("data_size", 0)

            if progress_callback:
                progress_callback("Compressing backup", 0.85)

            backup_dir = os.path.join(self.config.backup_dir, backup_id)
            tar_path = os.path.join(self.config.backup_dir, f"{backup_id}.tar")

            with tarfile.open(tar_path, "w") as tar:
                tar.add(backup_dir, arcname=backup_id)

            if self.config.compression != "none":
                compressed_path = tar_path + f".{self.config.compression}"
                self._compress_file(tar_path, compressed_path, self.config.compression)
                os.remove(tar_path)
                tar_path = compressed_path

            shutil.rmtree(backup_dir)

            metadata.file_path = tar_path
            metadata.file_size = os.path.getsize(tar_path)
            metadata.checksum = self._calculate_checksum(tar_path)
            metadata.end_time = datetime.now()
            metadata.status = BackupStatus.COMPLETED.value
            metadata.total_size = metadata.file_size + metadata.redis_data_size + metadata.vector_data_size

            metadata_file = tar_path + ".meta"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata.to_dict(), f, ensure_ascii=False, indent=2)

            if progress_callback:
                progress_callback("Backup completed", 1.0)

            logger.info(f"Backup completed: {backup_id}, size: {metadata.file_size} bytes")

        except Exception as e:
            metadata.status = BackupStatus.FAILED.value
            metadata.error_message = str(e)
            metadata.end_time = datetime.now()
            logger.error(f"Backup failed: {e}")

        return metadata

    def list_backups(self) -> List[BackupMetadata]:
        """
        列出所有备份

        Returns:
            List[BackupMetadata]: 备份列表
        """
        backups = []

        for filename in os.listdir(self.config.backup_dir):
            if filename.endswith(".meta"):
                meta_path = os.path.join(self.config.backup_dir, filename)
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    backups.append(BackupMetadata.from_dict(data))
                except Exception as e:
                    logger.warning(f"Failed to read backup metadata: {e}")

        backups.sort(key=lambda x: x.start_time, reverse=True)
        return backups

    def get_backup(self, backup_id: str) -> Optional[BackupMetadata]:
        """获取指定备份的元数据"""
        meta_path = os.path.join(self.config.backup_dir, f"{backup_id}.meta")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                return BackupMetadata.from_dict(json.load(f))
        return None

    def restore_mysql(
        self,
        backup_dir: str,
        tables: Optional[List[str]] = None,
        drop_existing: bool = False
    ) -> Dict[str, Any]:
        """
        恢复 MySQL 数据

        Args:
            backup_dir: 备份目录
            tables: 要恢复的表
            drop_existing: 是否删除现有表

        Returns:
            Dict: 恢复结果
        """
        if not self.mysql_client:
            return {"success": False, "error": "MySQL client not configured"}

        try:
            metadata_file = os.path.join(backup_dir, "backup_metadata.json")
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            available_tables = metadata.get("tables", [])
            target_tables = tables or available_tables

            records_restored = {}

            for table in target_tables:
                if table not in available_tables:
                    continue

                data_file = os.path.join(backup_dir, f"{table}_data.json")

                if not os.path.exists(data_file):
                    continue

                with open(data_file, "r", encoding="utf-8") as f:
                    records = json.load(f)

                if drop_existing:
                    self.mysql_client.execute_raw_sql(f"DELETE FROM `{table}`")

                if records:
                    inserted = self.mysql_client.insert_many(table, records)
                    records_restored[table] = inserted
                    logger.info(f"Restored table: {table} ({inserted} records)")

            return {
                "success": True,
                "tables_restored": list(records_restored.keys()),
                "records_restored": records_restored
            }

        except Exception as e:
            logger.error(f"MySQL restore failed: {e}")
            return {"success": False, "error": str(e)}

    def restore_redis(self, backup_dir: str) -> Dict[str, Any]:
        """
        恢复 Redis 数据

        Args:
            backup_dir: 备份目录

        Returns:
            Dict: 恢复结果
        """
        if not self.redis_cache:
            return {"success": False, "error": "Redis client not configured"}

        try:
            redis_file = os.path.join(backup_dir, "redis_data.json")
            if not os.path.exists(redis_file):
                return {"success": False, "error": "Redis backup file not found"}

            with open(redis_file, "r", encoding="utf-8") as f:
                all_data = json.load(f)

            for key, data in all_data.items():
                value = data["value"]
                ttl = data.get("ttl")

                if ttl:
                    self.redis_cache.set(key, value, ttl=ttl)
                else:
                    self.redis_cache.set(key, value)

            logger.info(f"Restored Redis: {len(all_data)} keys")
            return {"success": True, "keys_restored": len(all_data)}

        except Exception as e:
            logger.error(f"Redis restore failed: {e}")
            return {"success": False, "error": str(e)}

    def restore(
        self,
        backup_id: str,
        restore_target: Optional[str] = None,
        tables: Optional[List[str]] = None,
        drop_existing: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> RestoreMetadata:
        """
        恢复备份

        Args:
            backup_id: 备份ID
            restore_target: 恢复目标目录
            tables: 要恢复的表
            drop_existing: 是否删除现有数据
            progress_callback: 进度回调

        Returns:
            RestoreMetadata: 恢复元数据
        """
        metadata = self.get_backup(backup_id)
        if not metadata:
            raise ValueError(f"Backup not found: {backup_id}")

        restore_id = f"restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        restore_meta = RestoreMetadata(
            restore_id=restore_id,
            backup_id=backup_id,
            restore_type="full" if tables is None else "partial",
            status="in_progress",
            start_time=datetime.now()
        )

        try:
            backup_file = metadata.file_path
            if not os.path.exists(backup_file):
                raise FileNotFoundError(f"Backup file not found: {backup_file}")

            temp_dir = tempfile.mkdtemp()

            try:
                if metadata.compression != "none":
                    tar_path = backup_file.replace(f".{metadata.compression}", "")
                    self._decompress_file(backup_file, tar_path, metadata.compression)
                else:
                    tar_path = backup_file

                with tarfile.open(tar_path, "r") as tar:
                    tar.extractall(temp_dir)

                backup_dir = os.path.join(temp_dir, backup_id)

                if progress_callback:
                    progress_callback("Restoring MySQL", 0.2)

                if metadata.includes_mysql:
                    result = self.restore_mysql(backup_dir, tables, drop_existing)
                    if result.get("success"):
                        restore_meta.tables_restored = result.get("tables_restored", [])
                        restore_meta.records_restored = result.get("records_restored", {})

                if progress_callback:
                    progress_callback("Restoring Redis", 0.6)

                if metadata.includes_redis:
                    result = self.restore_redis(backup_dir)
                    if not result.get("success"):
                        restore_meta.tables_failed.append("redis")

                if progress_callback:
                    progress_callback("Restore completed", 1.0)

                restore_meta.status = "completed"
                restore_meta.end_time = datetime.now()

                logger.info(f"Restore completed: {restore_id}")

            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            restore_meta.status = "failed"
            restore_meta.error_message = str(e)
            restore_meta.end_time = datetime.now()
            logger.error(f"Restore failed: {e}")

        return restore_meta

    def delete_backup(self, backup_id: str) -> bool:
        """
        删除备份

        Args:
            backup_id: 备份ID

        Returns:
            bool: 是否成功
        """
        try:
            backup = self.get_backup(backup_id)
            if not backup:
                return False

            if backup.file_path and os.path.exists(backup.file_path):
                os.remove(backup.file_path)

            meta_file = backup.file_path + ".meta" if backup.file_path else None
            if meta_file and os.path.exists(meta_file):
                os.remove(meta_file)

            backup_dir = os.path.join(self.config.backup_dir, backup_id)
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)

            logger.info(f"Deleted backup: {backup_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete backup: {e}")
            return False

    def cleanup_old_backups(self) -> List[str]:
        """
        清理过期备份

        Returns:
            List[str]: 删除的备份ID列表
        """
        deleted = []
        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)

        backups = self.list_backups()

        for backup in backups:
            if backup.start_time < cutoff_date:
                if self.delete_backup(backup.backup_id):
                    deleted.append(backup.backup_id)

        while len(backups) > self.config.max_backups:
            oldest = backups.pop()
            if self.delete_backup(oldest.backup_id):
                deleted.append(oldest.backup_id)

        logger.info(f"Cleaned up {len(deleted)} old backups")
        return deleted

    def verify_backup(self, backup_id: str) -> Dict[str, Any]:
        """
        验证备份完整性

        Args:
            backup_id: 备份ID

        Returns:
            Dict: 验证结果
        """
        backup = self.get_backup(backup_id)
        if not backup:
            return {"valid": False, "error": "Backup not found"}

        result = {
            "backup_id": backup_id,
            "valid": True,
            "checks": []
        }

        if backup.file_path:
            if os.path.exists(backup.file_path):
                checksum = self._calculate_checksum(backup.file_path)
                if checksum == backup.checksum:
                    result["checks"].append({"name": "checksum", "passed": True})
                else:
                    result["checks"].append({"name": "checksum", "passed": False})
                    result["valid"] = False
            else:
                result["checks"].append({"name": "file_exists", "passed": False})
                result["valid"] = False

        result["checks"].append({
            "name": "mysql_tables",
            "passed": len(backup.tables) > 0 if backup.includes_mysql else True
        })

        return result

    def export_to_sql(
        self,
        backup_id: str,
        output_path: str,
        include_schema: bool = True
    ) -> bool:
        """
        导出备份为 SQL 文件

        Args:
            backup_id: 备份ID
            output_path: 输出路径
            include_schema: 是否包含表结构

        Returns:
            bool: 是否成功
        """
        backup = self.get_backup(backup_id)
        if not backup:
            return False

        try:
            backup_file = backup.file_path
            temp_dir = tempfile.mkdtemp()

            try:
                if backup.compression != "none":
                    tar_path = os.path.join(temp_dir, "backup.tar")
                    self._decompress_file(backup_file, tar_path, backup.compression)
                else:
                    tar_path = backup_file

                with tarfile.open(tar_path, "r") as tar:
                    tar.extractall(temp_dir)

                backup_dir = os.path.join(temp_dir, backup_id)
                metadata_file = os.path.join(backup_dir, "backup_metadata.json")

                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                with open(output_path, "w", encoding="utf-8") as out:
                    out.write("-- Database Backup Export\n")
                    out.write(f"-- Backup ID: {backup_id}\n")
                    out.write(f"-- Export Time: {datetime.now().isoformat()}\n\n")

                    for table in metadata.get("tables", []):
                        data_file = os.path.join(backup_dir, f"{table}_data.json")
                        if os.path.exists(data_file):
                            out.write(f"\n-- Table: {table}\n")

                            with open(data_file, "r") as f:
                                records = json.load(f)

                            if records and include_schema:
                                out.write(f"INSERT INTO `{table}` ({', '.join(records[0].keys())}) VALUES\n")
                                values = []
                                for record in records:
                                    vals = [f"'{v}'" if v is not None else "NULL" for v in record.values()]
                                    values.append(f"({', '.join(vals)})")
                                out.write(",\n".join(values) + ";\n")

                return True

            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            logger.error(f"Export to SQL failed: {e}")
            return False


def create_mysql_backup(
    config: DatabaseConfig,
    backup_path: str,
    tables: Optional[List[str]] = None
) -> bool:
    """
    创建 MySQL 备份的便捷函数

    Args:
        config: 数据库配置
        backup_path: 备份路径
        tables: 要备份的表

    Returns:
        bool: 是否成功
    """
    try:
        client = MySQLClient(config)
        manager = BackupRestore(mysql_client=client)

        os.makedirs(os.path.dirname(backup_path) or ".", exist_ok=True)

        result = subprocess.run(
            [
                "mysqldump",
                f"--host={config.host}",
                f"--port={config.port}",
                f"--user={config.user}",
                f"--password={config.password}",
                "--single-transaction",
                "--quick",
                "--lock-tables=false",
                config.database
            ] + (tables or []),
            capture_output=True,
            check=True
        )

        with open(backup_path, "wb") as f:
            f.write(result.stdout)

        return True

    except Exception as e:
        logger.error(f"MySQL backup failed: {e}")
        return False


def restore_mysql_backup(
    config: DatabaseConfig,
    backup_path: str
) -> bool:
    """
    恢复 MySQL 备份的便捷函数

    Args:
        config: 数据库配置
        backup_path: 备份路径

    Returns:
        bool: 是否成功
    """
    try:
        with open(backup_path, "rb") as f:
            sql_content = f.read().decode("utf-8")

        client = MySQLClient(config)
        with client.get_connection() as conn:
            for statement in sql_content.split(";"):
                statement = statement.strip()
                if statement:
                    conn.execute(statement)

        return True

    except Exception as e:
        logger.error(f"MySQL restore failed: {e}")
        return False
