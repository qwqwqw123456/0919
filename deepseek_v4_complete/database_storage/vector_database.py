"""
向量数据库模块
支持 FAISS、Milvus、Qdrant 等主流向量数据库
提供向量存储、相似度搜索、聚类等操作
"""

import os
import json
import pickle
import hashlib
import time
from typing import Optional, List, Dict, Any, Union, Tuple, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class VectorIndexType(Enum):
    """向量索引类型"""
    FLAT = "flat"
    IVF_FLAT = "ivf_flat"
    IVF_PQ = "ivf_pq"
    HNSW = "hnsw"
    GPU_FLAT = "gpu_flat"
    GPU_IVF_FLAT = "gpu_ivf_flat"
    GPU_IVF_PQ = "gpu_ivf_pq"


class DistanceMetric(Enum):
    """距离度量类型"""
    L2 = "l2"
    IP = "ip"
    COSINE = "cosine"


@dataclass
class VectorRecord:
    """向量记录"""
    id: str
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "vector": self.vector.tolist() if isinstance(self.vector, np.ndarray) else self.vector,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorRecord":
        """从字典创建"""
        return cls(
            id=data["id"],
            vector=np.array(data["vector"]) if isinstance(data["vector"], list) else data["vector"],
            metadata=data.get("metadata", {})
        )


@dataclass
class SearchResult:
    """搜索结果"""
    id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "score": self.score,
            "metadata": self.metadata
        }


@dataclass
class IndexStats:
    """索引统计信息"""
    total_vectors: int
    dimension: int
    index_type: str
    metric_type: str
    size_bytes: Optional[int] = None
    build_time: Optional[float] = None


class VectorDBConfig:
    """向量数据库配置"""

    def __init__(
        self,
        provider: str = "faiss",
        dimension: int = 768,
        index_type: str = "flat",
        metric: str = "l2",
        index_params: Optional[Dict[str, Any]] = None,
        search_params: Optional[Dict[str, Any]] = None,
        nlist: int = 100,
        nprobe: int = 10,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        cache_size: int = 1000
    ):
        """
        初始化配置

        Args:
            provider: 提供商 (faiss/milvus/qdrant)
            dimension: 向量维度
            index_type: 索引类型
            metric: 距离度量
            index_params: 索引参数
            search_params: 搜索参数
            nlist: IVF 簇数量
            nprobe: IVF 搜索的探针数
            m: HNSW 的连接数
            ef_construction: HNSW 构建参数
            ef_search: HNSW 搜索参数
            cache_size: 缓存大小
        """
        self.provider = provider
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.index_params = index_params or {}
        self.search_params = search_params or {}
        self.nlist = nlist
        self.nprobe = nprobe
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.cache_size = cache_size

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "VectorDBConfig":
        """从字典加载"""
        return cls(**config)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "provider": self.provider,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "index_params": self.index_params,
            "search_params": self.search_params,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "m": self.m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "cache_size": self.cache_size
        }


class BaseVectorDB(ABC):
    """向量数据库基类"""

    def __init__(self, config: VectorDBConfig):
        self.config = config
        self._is_initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """初始化向量数据库"""
        pass

    @abstractmethod
    def add(self, records: List[VectorRecord]) -> bool:
        """添加向量"""
        pass

    @abstractmethod
    def search(
        self,
        query: Union[np.ndarray, List[float]],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """搜索向量"""
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """删除向量"""
        pass

    @abstractmethod
    def get(self, ids: List[str]) -> List[VectorRecord]:
        """获取向量"""
        pass

    @abstractmethod
    def update(self, records: List[VectorRecord]) -> bool:
        """更新向量"""
        pass

    @abstractmethod
    def count(self) -> int:
        """获取向量数量"""
        pass

    @abstractmethod
    def save(self, path: str) -> bool:
        """保存索引到文件"""
        pass

    @abstractmethod
    def load(self, path: str) -> bool:
        """从文件加载索引"""
        pass

    @abstractmethod
    def reset(self):
        """重置索引"""
        pass

    @abstractmethod
    def get_stats(self) -> IndexStats:
        """获取索引统计"""
        pass


class FAISSVectorDB(BaseVectorDB):
    """
    FAISS 向量数据库实现
    基于 Facebook AI Similarity Search
    """

    def __init__(
        self,
        config: Optional[VectorDBConfig] = None,
        dimension: int = 768,
        index_type: str = "flat",
        metric: str = "l2"
    ):
        if config is None:
            config = VectorDBConfig(
                provider="faiss",
                dimension=dimension,
                index_type=index_type,
                metric=metric
            )
        super().__init__(config)

        import faiss
        self.faiss = faiss
        self._index = None
        self._id_map: Dict[str, int] = {}
        self._reverse_id_map: Dict[int, str] = {}
        self._metadata_store: Dict[str, Dict[str, Any]] = {}
        self._vector_store: Dict[str, np.ndarray] = {}
        self._current_id = 0

    def initialize(self) -> bool:
        """初始化 FAISS 索引"""
        try:
            dim = self.config.dimension

            if self.config.index_type == "flat":
                if self.config.metric == "l2":
                    self._index = self.faiss.IndexFlatL2(dim)
                elif self.config.metric in ("ip", "cosine"):
                    self._index = self.faiss.IndexFlatIP(dim)
                else:
                    self._index = self.faiss.IndexFlatL2(dim)

            elif self.config.index_type == "ivf_flat":
                quantizer = self.faiss.IndexFlatL2(dim)
                self._index = self.faiss.IndexIVFFlat(
                    quantizer, dim, self.config.nlist,
                    self.faiss.METRIC_L2 if self.config.metric == "l2" else self.faiss.METRIC_INNER_PRODUCT
                )

            elif self.config.index_type == "ivf_pq":
                m = self.config.m.get("m", 16)
                bits = self.config.m.get("bits", 8)
                quantizer = self.faiss.IndexFlatL2(dim)
                self._index = self.faiss.IndexIVFPQ(
                    quantizer, dim, self.config.nlist, m, bits,
                    self.faiss.METRIC_L2 if self.config.metric == "l2" else self.faiss.METRIC_INNER_PRODUCT
                )

            elif self.config.index_type == "hnsw":
                self._index = self.faiss.IndexHNSWFlat(
                    dim, self.config.m,
                    self.faiss.METRIC_L2 if self.config.metric == "l2" else self.faiss.METRIC_INNER_PRODUCT
                )
                self._index.hnsw.efConstruction = self.config.ef_construction
                self._index.hnsw.efSearch = self.config.ef_search

            else:
                self._index = self.faiss.IndexFlatL2(dim)

            self._is_initialized = True
            return True

        except Exception as e:
            print(f"FAISS 初始化失败: {e}")
            return False

    def add(self, records: List[VectorRecord]) -> bool:
        """添加向量到索引"""
        if not self._is_initialized:
            self.initialize()

        try:
            vectors = []
            for record in records:
                vec = record.vector.astype(np.float32)
                if vec.ndim == 1:
                    vec = vec.reshape(1, -1)

                internal_id = self._current_id
                self._id_map[record.id] = internal_id
                self._reverse_id_map[internal_id] = record.id
                self._metadata_store[record.id] = record.metadata
                self._vector_store[record.id] = record.vector

                vectors.append(vec)
                self._current_id += 1

            if vectors:
                vectors_array = np.vstack(vectors)
                if self._index.is_trained:
                    self._index.add(vectors_array)
                else:
                    self._index.train(vectors_array)
                    self._index.add(vectors_array)

            return True

        except Exception as e:
            print(f"添加向量失败: {e}")
            return False

    def search(
        self,
        query: Union[np.ndarray, List[float]],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """搜索相似向量"""
        if not self._is_initialized or self._index is None:
            return []

        try:
            if isinstance(query, list):
                query = np.array(query).astype(np.float32)

            if query.ndim == 1:
                query = query.reshape(1, -1)

            k = min(top_k, self._index.ntotal)
            if k == 0:
                return []

            distances, indices = self._index.search(query, k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    continue

                original_id = self._reverse_id_map.get(int(idx))
                if original_id:
                    metadata = self._metadata_store.get(original_id, {})

                    if filters:
                        match = all(
                            metadata.get(k) == v
                            for k, v in filters.items()
                        )
                        if not match:
                            continue

                    results.append(SearchResult(
                        id=original_id,
                        score=float(dist),
                        metadata=metadata,
                        vector=self._vector_store.get(original_id)
                    ))

            return results

        except Exception as e:
            print(f"搜索失败: {e}")
            return []

    def delete(self, ids: List[str]) -> bool:
        """删除向量(FAISS 不支持直接删除，需要重建索引)"""
        try:
            for record_id in ids:
                if record_id in self._id_map:
                    del self._id_map[record_id]
                if record_id in self._metadata_store:
                    del self._metadata_store[record_id]
                if record_id in self._vector_store:
                    del self._vector_store[record_id]

            self._rebuild_index()
            return True

        except Exception as e:
            print(f"删除向量失败: {e}")
            return False

    def _rebuild_index(self):
        """重建索引"""
        if not self._vector_store:
            self._index = self.faiss.IndexFlatL2(self.config.dimension)
            self._id_map.clear()
            self._reverse_id_map.clear()
            self._current_id = 0
            return

        records = [
            VectorRecord(id=k, vector=v, metadata=self._metadata_store.get(k, {}))
            for k, v in self._vector_store.items()
        ]

        temp_index = self._index
        self._index = None
        self._id_map.clear()
        self._reverse_id_map.clear()
        self._current_id = 0

        self.initialize()
        self.add(records)

    def get(self, ids: List[str]) -> List[VectorRecord]:
        """获取向量"""
        records = []
        for record_id in ids:
            if record_id in self._vector_store:
                records.append(VectorRecord(
                    id=record_id,
                    vector=self._vector_store[record_id],
                    metadata=self._metadata_store.get(record_id, {})
                ))
        return records

    def update(self, records: List[VectorRecord]) -> bool:
        """更新向量"""
        try:
            for record in records:
                self._vector_store[record.id] = record.vector
                self._metadata_store[record.id] = record.metadata

            self._rebuild_index()
            return True

        except Exception as e:
            print(f"更新向量失败: {e}")
            return False

    def count(self) -> int:
        """获取向量数量"""
        if self._index:
            return self._index.ntotal
        return 0

    def save(self, path: str) -> bool:
        """保存索引到文件"""
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

            self.faiss.write_index(
                self._index,
                f"{path}.index"
            )

            metadata = {
                "id_map": self._id_map,
                "reverse_id_map": {str(k): v for k, v in self._reverse_id_map.items()},
                "metadata_store": self._metadata_store,
                "config": self.config.to_dict(),
                "current_id": self._current_id
            }

            with open(f"{path}.metadata", "wb") as f:
                pickle.dump(metadata, f)

            return True

        except Exception as e:
            print(f"保存索引失败: {e}")
            return False

    def load(self, path: str) -> bool:
        """从文件加载索引"""
        try:
            index_path = f"{path}.index"
            metadata_path = f"{path}.metadata"

            if not os.path.exists(index_path) or not os.path.exists(metadata_path):
                return False

            self._index = self.faiss.read_index(index_path)

            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)

            self._id_map = metadata["id_map"]
            self._reverse_id_map = {int(k): v for k, v in metadata["reverse_id_map"].items()}
            self._metadata_store = metadata["metadata_store"]
            self.config = VectorDBConfig.from_dict(metadata["config"])
            self._current_id = metadata["current_id"]

            self._vector_store = {}
            for record_id in self._id_map.keys():
                pass

            self._is_initialized = True
            return True

        except Exception as e:
            print(f"加载索引失败: {e}")
            return False

    def reset(self):
        """重置索引"""
        self._index = None
        self._id_map.clear()
        self._reverse_id_map.clear()
        self._metadata_store.clear()
        self._vector_store.clear()
        self._current_id = 0
        self._is_initialized = False

    def get_stats(self) -> IndexStats:
        """获取索引统计"""
        return IndexStats(
            total_vectors=self.count(),
            dimension=self.config.dimension,
            index_type=self.config.index_type,
            metric_type=self.config.metric
        )


class MilvusVectorDB(BaseVectorDB):
    """
    Milvus 向量数据库实现
    基于 Zilliz Cloud / Milvus
    """

    def __init__(
        self,
        config: Optional[VectorDBConfig] = None,
        uri: str = "http://localhost:19530",
        token: str = "",
        collection_name: str = "default",
        dimension: int = 768
    ):
        if config is None:
            config = VectorDBConfig(provider="milvus", dimension=dimension)
        super().__init__(config)

        self.uri = uri
        self.token = token
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    def _get_client(self):
        """获取 Milvus 客户端"""
        try:
            from pymilvus import connections, Collection

            connections.connect(
                uri=self.uri,
                token=self.token if self.token else None,
                alias="default"
            )

            self._client = connections

            try:
                self._collection = Collection(self.collection_name)
                self._collection.load()
            except Exception:
                self._collection = None

            return True

        except ImportError:
            print("请安装 pymilvus: pip install pymilvus")
            return False
        except Exception as e:
            print(f"Milvus 连接失败: {e}")
            return False

    def initialize(self) -> bool:
        """初始化 Milvus 连接和集合"""
        return self._get_client()

    def add(self, records: List[VectorRecord]) -> bool:
        """添加向量"""
        if not self._get_client():
            return False

        try:
            from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, utility

            if not utility.has_collection(self.collection_name):
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.config.dimension)
                ]
                schema = CollectionSchema(fields=fields, description="Vector collection")
                self._collection = Collection(name=self.collection_name, schema=schema)

            entities = [
                [r.id for r in records],
                [r.vector.tolist() for r in records]
            ]

            if self._collection:
                self._collection.insert(entities)
                self._collection.flush()
                return True

            return False

        except Exception as e:
            print(f"添加向量失败: {e}")
            return False

    def search(
        self,
        query: Union[np.ndarray, List[float]],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """搜索向量"""
        if not self._get_client() or self._collection is None:
            return []

        try:
            from pymilvus import Collection

            if isinstance(query, list):
                query = np.array(query)

            query_vector = query.flatten().tolist()

            search_params = {
                "metric_type": self.config.metric.upper(),
                "params": {"nprobe": self.config.nprobe}
            }

            results = self._collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["id"]
            )

            search_results = []
            for hits in results:
                for hit in hits:
                    search_results.append(SearchResult(
                        id=hit.entity.get("id", hit.id),
                        score=hit.distance,
                        metadata={}
                    ))

            return search_results

        except Exception as e:
            print(f"搜索失败: {e}")
            return []

    def delete(self, ids: List[str]) -> bool:
        """删除向量"""
        if not self._get_client() or self._collection is None:
            return False

        try:
            expr = f'id in [{", ".join(f"\"{i}\"" for i in ids)}]'
            self._collection.delete(expr)
            return True

        except Exception as e:
            print(f"删除向量失败: {e}")
            return False

    def get(self, ids: List[str]) -> List[VectorRecord]:
        """获取向量"""
        if not self._get_client() or self._collection is None:
            return []

        try:
            from pymilvus import Collection

            expr = f'id in [{", ".join(f"\"{i}\"" for i in ids)}]'
            results = self._collection.query(
                expr=expr,
                output_fields=["id", "vector"]
            )

            records = []
            for item in results:
                records.append(VectorRecord(
                    id=item["id"],
                    vector=np.array(item["vector"]),
                    metadata={}
                ))

            return records

        except Exception as e:
            print(f"获取向量失败: {e}")
            return []

    def update(self, records: List[VectorRecord]) -> bool:
        """更新向量"""
        self.delete([r.id for r in records])
        return self.add(records)

    def count(self) -> int:
        """获取向量数量"""
        if not self._get_client() or self._collection is None:
            return 0

        try:
            return self._collection.num_entities
        except Exception:
            return 0

    def save(self, path: str) -> bool:
        """Milvus 不需要手动保存"""
        return True

    def load(self, path: str) -> bool:
        """加载集合"""
        return self._get_client()

    def reset(self):
        """重置"""
        if self._get_client():
            try:
                from pymilvus import utility
                if utility.has_collection(self.collection_name):
                    utility.drop_collection(self.collection_name)
            except Exception:
                pass

    def get_stats(self) -> IndexStats:
        """获取统计"""
        return IndexStats(
            total_vectors=self.count(),
            dimension=self.config.dimension,
            index_type="milvus",
            metric_type=self.config.metric
        )


class VectorDBManager:
    """
    向量数据库管理器
    统一管理多种向量数据库
    """

    _instances: Dict[str, BaseVectorDB] = {}

    @classmethod
    def get_instance(
        cls,
        name: str = "default",
        provider: str = "faiss",
        **kwargs
    ) -> BaseVectorDB:
        """
        获取向量数据库实例

        Args:
            name: 实例名称
            provider: 提供商
            **kwargs: 其他参数

        Returns:
            BaseVectorDB: 向量数据库实例
        """
        if name in cls._instances:
            return cls._instances[name]

        if provider == "faiss":
            instance = FAISSVectorDB(**kwargs)
        elif provider == "milvus":
            instance = MilvusVectorDB(**kwargs)
        else:
            instance = FAISSVectorDB(**kwargs)

        instance.initialize()
        cls._instances[name] = instance
        return instance

    @classmethod
    def register_instance(cls, name: str, instance: BaseVectorDB):
        """注册实例"""
        cls._instances[name] = instance

    @classmethod
    def list_instances(cls) -> List[str]:
        """列出所有实例"""
        return list(cls._instances.keys())

    @classmethod
    def remove_instance(cls, name: str):
        """移除实例"""
        if name in cls._instances:
            del cls._instances[name]


class VectorOperations:
    """向量操作工具类"""

    @staticmethod
    def normalize(vector: np.ndarray) -> np.ndarray:
        """L2 归一化"""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    @staticmethod
    def batch_normalize(vectors: np.ndarray) -> np.ndarray:
        """批量 L2 归一化"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """计算余弦相似度"""
        v1_norm = VectorOperations.normalize(v1)
        v2_norm = VectorOperations.normalize(v2)
        return float(np.dot(v1_norm, v2_norm))

    @staticmethod
    def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        """计算欧几里得距离"""
        return float(np.linalg.norm(v1 - v2))

    @staticmethod
    def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
        """计算点积"""
        return float(np.dot(v1, v2))

    @staticmethod
    def batch_cosine_similarity(
        query: np.ndarray,
        vectors: np.ndarray
    ) -> np.ndarray:
        """批量计算余弦相似度"""
        query_norm = VectorOperations.normalize(query)
        vectors_norm = VectorOperations.batch_normalize(vectors)
        return np.dot(vectors_norm, query_norm)


def get_vector_db(
    provider: str = "faiss",
    dimension: int = 768,
    **kwargs
) -> BaseVectorDB:
    """
    获取向量数据库实例的工厂函数

    Args:
        provider: 提供商
        dimension: 向量维度
        **kwargs: 其他参数

    Returns:
        BaseVectorDB: 向量数据库实例
    """
    if provider == "faiss":
        return FAISSVectorDB(dimension=dimension, **kwargs)
    elif provider == "milvus":
        return MilvusVectorDB(dimension=dimension, **kwargs)
    else:
        return FAISSVectorDB(dimension=dimension, **kwargs)
