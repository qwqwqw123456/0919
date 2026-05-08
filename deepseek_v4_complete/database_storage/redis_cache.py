"""
Redis 缓存模块
提供完整的 Redis 连接、缓存操作、分布式锁、消息队列等功能
"""

import os
import json
import time
import uuid
import hashlib
import pickle
from typing import Optional, List, Dict, Any, Union, Callable, TypeVar, Generic
from contextlib import contextmanager
from datetime import timedelta, datetime
from functools import wraps
from abc import ABC, abstractmethod

import redis
from redis import Redis, ConnectionPool, RedisCluster
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
from redis.lock import Lock


T = TypeVar("T")


class RedisConfig:
    """Redis 配置管理类"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 50,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        socket_keepalive: bool = True,
        health_check_interval: int = 30,
        decode_responses: bool = True,
        encoding: str = "utf-8",
        encoding_errors: str = "strict"
    ):
        """
        初始化 Redis 配置

        Args:
            host: Redis 主机地址
            port: Redis 端口
            db: 数据库编号
            password: 密码
            max_connections: 最大连接数
            socket_timeout: socket 超时时间
            socket_connect_timeout: 连接超时时间
            socket_keepalive: 是否保持连接
            health_check_interval: 健康检查间隔
            decode_responses: 是否自动解码响应
            encoding: 编码格式
            encoding_errors: 编码错误处理
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.max_connections = max_connections
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.socket_keepalive = socket_keepalive
        self.health_check_interval = health_check_interval
        self.decode_responses = decode_responses
        self.encoding = encoding
        self.encoding_errors = encoding_errors

    @classmethod
    def from_env(cls) -> "RedisConfig":
        """从环境变量加载配置"""
        return cls(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
            max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "50")),
            decode_responses=os.getenv("REDIS_DECODE_RESPONSES", "true").lower() == "true"
        )

    @classmethod
    def from_url(cls, url: str) -> "RedisConfig":
        """从 URL 解析配置"""
        import re
        pattern = r"redis://(?::([^@]+)@)?([^:]+):(\d+)(?:/(\d+))?"
        match = re.match(pattern, url)
        if match:
            password, host, port, db = match.groups()
            return cls(
                host=host,
                port=int(port),
                db=int(db) if db else 0,
                password=password
            )
        raise ValueError(f"无效的 Redis URL: {url}")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "password": self.password,
            "max_connections": self.max_connections,
            "socket_timeout": self.socket_timeout,
            "socket_connect_timeout": self.socket_connect_timeout,
            "socket_keepalive": self.socket_keepalive,
            "health_check_interval": self.health_check_interval,
            "decode_responses": self.decode_responses,
            "encoding": self.encoding,
            "encoding_errors": self.encoding_errors
        }


class SerializationType:
    """序列化类型"""
    JSON = "json"
    PICKLE = "pickle"
    STRING = "string"


class Serializer:
    """序列化工具类"""

    @staticmethod
    def serialize(value: Any, serializer: str = SerializationType.JSON) -> str:
        """
        序列化值

        Args:
            value: 要序列化的值
            serializer: 序列化类型

        Returns:
            str: 序列化后的字符串
        """
        if serializer == SerializationType.JSON:
            return json.dumps(value, ensure_ascii=False, default=str)
        elif serializer == SerializationType.PICKLE:
            return pickle.dumps(value)
        else:
            return str(value)

    @staticmethod
    def deserialize(value: Union[str, bytes], serializer: str = SerializationType.JSON) -> Any:
        """
        反序列化值

        Args:
            value: 要反序列化的值
            serializer: 序列化类型

        Returns:
            Any: 反序列化后的值
        """
        if value is None:
            return None
        if serializer == SerializationType.JSON:
            return json.loads(value)
        elif serializer == SerializationType.PICKLE:
            return pickle.loads(value)
        else:
            return value


class CacheStats:
    """缓存统计信息"""

    def __init__(self):
        self.hits: int = 0
        self.misses: int = 0
        self.sets: int = 0
        self.deletes: int = 0
        self.errors: int = 0

    @property
    def total_requests(self) -> int:
        """总请求数"""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """命中率"""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": self.total_requests,
            "hit_rate": round(self.hit_rate, 4),
            "sets": self.sets,
            "deletes": self.deletes,
            "errors": self.errors
        }


class RedisCache:
    """
    Redis 缓存客户端
    提供完整的缓存操作、分布式锁、消息发布/订阅等功能
    """

    def __init__(
        self,
        config: Optional[RedisConfig] = None,
        redis_client: Optional[Redis] = None,
        default_ttl: int = 3600,
        default_serializer: str = SerializationType.JSON,
        key_prefix: str = "cache:"
    ):
        """
        初始化 Redis 缓存客户端

        Args:
            config: Redis 配置
            redis_client: 已有的 Redis 客户端
            default_ttl: 默认过期时间(秒)
            default_serializer: 默认序列化类型
            key_prefix: 键前缀
        """
        self.config = config or RedisConfig()
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[Redis] = redis_client
        self.default_ttl = default_ttl
        self.default_serializer = default_serializer
        self.key_prefix = key_prefix
        self.stats = CacheStats()

    @property
    def client(self) -> Redis:
        """获取或创建 Redis 客户端"""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_pool(self) -> ConnectionPool:
        """创建连接池"""
        return ConnectionPool(
            host=self.config.host,
            port=self.config.port,
            db=self.config.db,
            password=self.config.password,
            max_connections=self.config.max_connections,
            socket_timeout=self.config.socket_timeout,
            socket_connect_timeout=self.config.socket_connect_timeout,
            socket_keepalive=self.config.socket_keepalive,
            health_check_interval=self.config.health_check_interval,
            decode_responses=self.config.decode_responses,
            encoding=self.config.encoding,
            encoding_errors=self.config.encoding_errors
        )

    def _create_client(self) -> Redis:
        """创建 Redis 客户端"""
        if self._pool is None:
            self._pool = self._create_pool()
        return Redis(connection_pool=self._pool)

    def _make_key(self, key: str) -> str:
        """生成带前缀的键"""
        if key.startswith(self.key_prefix):
            return key
        return f"{self.key_prefix}{key}"

    def _parse_key(self, key: str) -> str:
        """移除前缀获取原始键"""
        if key.startswith(self.key_prefix):
            return key[len(self.key_prefix):]
        return key

    @contextmanager
    def pipeline(self):
        """获取管道上下文管理器"""
        pipe = self.client.pipeline()
        try:
            yield pipe
        finally:
            pipe.close()

    def get(self, key: str, serializer: Optional[str] = None) -> Optional[Any]:
        """
        获取缓存值

        Args:
            key: 缓存键
            serializer: 序列化类型

        Returns:
            缓存值，不存在返回 None
        """
        try:
            value = self.client.get(self._make_key(key))
            if value is None:
                self.stats.misses += 1
                return None
            self.stats.hits += 1
            return Serializer.deserialize(value, serializer or self.default_serializer)
        except RedisError:
            self.stats.errors += 1
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serializer: Optional[str] = None
    ) -> bool:
        """
        设置缓存值

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间(秒)
            serializer: 序列化类型

        Returns:
            bool: 是否成功
        """
        try:
            serialized = Serializer.serialize(value, serializer or self.default_serializer)
            ttl = ttl or self.default_ttl
            result = self.client.set(self._make_key(key), serialized, ex=ttl)
            self.stats.sets += 1
            return bool(result)
        except RedisError:
            self.stats.errors += 1
            return False

    def setnx(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        设置值(仅当键不存在时)

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间(秒)

        Returns:
            bool: 是否设置成功
        """
        try:
            serialized = Serializer.serialize(value, self.default_serializer)
            result = self.client.setnx(self._make_key(key), serialized)
            if result and ttl:
                self.client.expire(self._make_key(key), ttl)
            return bool(result)
        except RedisError:
            self.stats.errors += 1
            return False

    def setex(self, key: str, value: Any, ttl: int) -> bool:
        """
        设置值并指定过期时间

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间(秒)

        Returns:
            bool: 是否成功
        """
        try:
            serialized = Serializer.serialize(value, self.default_serializer)
            result = self.client.setex(self._make_key(key), ttl, serialized)
            self.stats.sets += 1
            return bool(result)
        except RedisError:
            self.stats.errors += 1
            return False

    def delete(self, *keys: str) -> int:
        """
        删除缓存

        Args:
            keys: 要删除的键

        Returns:
            int: 删除的数量
        """
        try:
            full_keys = [self._make_key(k) for k in keys]
            result = self.client.delete(*full_keys)
            self.stats.deletes += result
            return result
        except RedisError:
            self.stats.errors += 1
            return 0

    def exists(self, *keys: str) -> int:
        """检查键是否存在"""
        try:
            full_keys = [self._make_key(k) for k in keys]
            return self.client.exists(*full_keys)
        except RedisError:
            self.stats.errors += 1
            return 0

    def expire(self, key: str, ttl: int) -> bool:
        """设置过期时间"""
        try:
            return bool(self.client.expire(self._make_key(key), ttl))
        except RedisError:
            self.stats.errors += 1
            return False

    def ttl(self, key: str) -> int:
        """获取键的剩余过期时间"""
        try:
            return self.client.ttl(self._make_key(key))
        except RedisError:
            return -2

    def rename(self, old_key: str, new_key: str) -> bool:
        """重命名键"""
        try:
            return bool(self.client.rename(
                self._make_key(old_key),
                self._make_key(new_key)
            ))
        except RedisError:
            return False

    def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """递增"""
        try:
            return self.client.incr(self._make_key(key), amount)
        except RedisError:
            self.stats.errors += 1
            return None

    def decrement(self, key: str, amount: int = 1) -> Optional[int]:
        """递减"""
        try:
            return self.client.decr(self._make_key(key), amount)
        except RedisError:
            self.stats.errors += 1
            return None

    def get_many(self, *keys: str) -> Dict[str, Any]:
        """批量获取"""
        try:
            full_keys = [self._make_key(k) for k in keys]
            values = self.client.mget(full_keys)
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = Serializer.deserialize(
                        value, self.default_serializer
                    )
                else:
                    self.stats.misses += 1
            self.stats.hits += len(result)
            return result
        except RedisError:
            self.stats.errors += 1
            return {}

    def set_many(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> int:
        """批量设置"""
        try:
            pipe = self.client.pipeline()
            for key, value in mapping.items():
                serialized = Serializer.serialize(value, self.default_serializer)
                full_key = self._make_key(key)
                if ttl:
                    pipe.setex(full_key, ttl, serialized)
                else:
                    pipe.set(full_key, serialized)
            results = pipe.execute()
            self.stats.sets += sum(1 for r in results if r)
            return sum(1 for r in results if r)
        except RedisError:
            self.stats.errors += 1
            return 0

    def get_or_set(
        self,
        key: str,
        default: Callable[[], Any],
        ttl: Optional[int] = None
    ) -> Any:
        """
        获取缓存值，不存在则调用 default 并缓存

        Args:
            key: 缓存键
            default: 默认值生成函数
            ttl: 过期时间

        Returns:
            缓存值
        """
        value = self.get(key)
        if value is None:
            value = default()
            if value is not None:
                self.set(key, value, ttl)
        return value

    def clear_pattern(self, pattern: str) -> int:
        """清除匹配模式的所有键"""
        try:
            full_pattern = self._make_key(pattern)
            keys = self.client.keys(full_pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except RedisError:
            self.stats.errors += 1
            return 0

    def keys(self, pattern: str = "*") -> List[str]:
        """获取匹配模式的所有键"""
        try:
            full_pattern = self._make_key(pattern)
            keys = self.client.keys(full_pattern)
            return [self._parse_key(k) for k in keys]
        except RedisError:
            return []

    def scan(self, pattern: str = "*", count: int = 100) -> List[str]:
        """扫描键(比 keys 更高效)"""
        try:
            full_pattern = self._make_key(pattern)
            cursor = 0
            all_keys = []
            while True:
                cursor, keys = self.client.scan(cursor, match=full_pattern, count=count)
                all_keys.extend([self._parse_key(k) for k in keys])
                if cursor == 0:
                    break
            return all_keys
        except RedisError:
            return []

    def get_info(self) -> Dict[str, Any]:
        """获取 Redis 服务器信息"""
        try:
            return self.client.info()
        except RedisError:
            return {}

    def ping(self) -> bool:
        """检测连接"""
        try:
            return self.client.ping()
        except RedisError:
            return False

    def close(self):
        """关闭连接"""
        if self._client:
            self._client.close()
            self._client = None
        if self._pool:
            self._pool.disconnect()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class DistributedLock:
    """
    分布式锁
    基于 Redis 实现的分布式锁机制
    """

    def __init__(
        self,
        cache: RedisCache,
        lock_name: str,
        timeout: int = 30,
        blocking: bool = True,
        blocking_timeout: int = 10
    ):
        """
        初始化分布式锁

        Args:
            cache: RedisCache 实例
            lock_name: 锁名称
            timeout: 锁超时时间(秒)
            blocking: 是否阻塞等待
            blocking_timeout: 阻塞超时时间(秒)
        """
        self.cache = cache
        self.lock_name = f"lock:{lock_name}"
        self.timeout = timeout
        self.blocking = blocking
        self.blocking_timeout = blocking_timeout
        self.lock_id = str(uuid.uuid4())
        self._acquired = False

    def acquire(self) -> bool:
        """
        获取锁

        Returns:
            bool: 是否获取成功
        """
        start_time = time.time()
        while True:
            if self.cache.client.set(
                self.lock_name,
                self.lock_id,
                nx=True,
                ex=self.timeout
            ):
                self._acquired = True
                return True

            if not self.blocking:
                return False

            if time.time() - start_time >= self.blocking_timeout:
                return False

            time.sleep(0.01)

    def release(self) -> bool:
        """释放锁"""
        if not self._acquired:
            return False

        try:
            lua_script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            result = self.cache.client.eval(lua_script, 1, self.lock_name, self.lock_id)
            self._acquired = False
            return bool(result)
        except RedisError:
            return False

    def extend(self, additional_time: int) -> bool:
        """延长锁的过期时间"""
        if not self._acquired:
            return False

        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("expire", KEYS[1], ARGV[2])
        else
            return 0
        end
        """
        try:
            result = self.cache.client.eval(
                lua_script, 1,
                self.lock_name,
                self.lock_id,
                additional_time
            )
            return bool(result)
        except RedisError:
            return False

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError(f"无法获取锁: {self.lock_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


class RateLimiter:
    """
    限流器
    基于 Redis 实现的滑动窗口限流
    """

    def __init__(
        self,
        cache: RedisCache,
        key: str,
        max_requests: int,
        window_seconds: int
    ):
        """
        初始化限流器

        Args:
            cache: RedisCache 实例
            key: 限流键
            max_requests: 最大请求数
            window_seconds: 时间窗口(秒)
        """
        self.cache = cache
        self.key = f"ratelimit:{key}"
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def is_allowed(self) -> bool:
        """
        检查请求是否允许

        Returns:
            bool: 是否允许请求
        """
        try:
            now = time.time()
            window_start = now - self.window_seconds

            pipe = self.cache.client.pipeline()
            pipe.zremrangebyscore(self.key, 0, window_start)
            pipe.zcard(self.key)
            pipe.zadd(self.key, {str(now): now})
            pipe.expire(self.key, self.window_seconds)
            results = pipe.execute()

            current_count = results[1]
            return current_count < self.max_requests
        except RedisError:
            return True

    def get_remaining(self) -> int:
        """获取剩余请求数"""
        try:
            now = time.time()
            window_start = now - self.window_seconds
            self.cache.client.zremrangebyscore(self.key, 0, window_start)
            current_count = self.cache.client.zcard(self.key)
            return max(0, self.max_requests - current_count)
        except RedisError:
            return self.max_requests

    def reset(self):
        """重置限流器"""
        try:
            self.cache.client.delete(self.key)
        except RedisError:
            pass


class PubSubManager:
    """
    发布订阅管理器
    """

    def __init__(self, cache: RedisCache):
        self.cache = cache
        self._pubsub = None
        self._subscriptions: Dict[str, Callable] = {}

    def publish(self, channel: str, message: Any) -> int:
        """
        发布消息

        Args:
            channel: 频道名称
            message: 消息内容

        Returns:
            int: 订阅者数量
        """
        try:
            serialized = Serializer.serialize(message, self.cache.default_serializer)
            return self.cache.client.publish(channel, serialized)
        except RedisError:
            return 0

    def subscribe(self, channel: str, callback: Callable[[Any], None]):
        """订阅频道"""
        if self._pubsub is None:
            self._pubsub = self.cache.client.pubsub()
        self._pubsub.subscribe(channel)
        self._subscriptions[channel] = callback

    def psubscribe(self, pattern: str, callback: Callable[[str, Any], None]):
        """订阅模式"""
        if self._pubsub is None:
            self._pubsub = self.cache.client.pubsub()
        self._pubsub.psubscribe(pattern)
        self._subscriptions[pattern] = callback

    def listen(self, timeout: float = 1.0):
        """监听消息"""
        if self._pubsub is None:
            return

        for message in self._pubsub.listen():
            if message["type"] == "message":
                channel = message["channel"]
                if channel in self._subscriptions:
                    data = Serializer.deserialize(
                        message["data"],
                        self.cache.default_serializer
                    )
                    self._subscriptions[channel](data)

    def close(self):
        """关闭发布订阅"""
        if self._pubsub:
            self._pubsub.close()
            self._pubsub = None
            self._subscriptions.clear()


class CacheDecorator:
    """缓存装饰器"""

    @staticmethod
    def cached(
        cache: RedisCache,
        key_func: Optional[Callable[..., str]] = None,
        ttl: Optional[int] = None,
        unless: Optional[Callable[..., bool]] = None
    ):
        """
        缓存装饰器

        Args:
            cache: RedisCache 实例
            key_func: 键生成函数
            ttl: 过期时间
            unless: 条件函数，返回 True 时跳过缓存
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if unless and unless(*args, **kwargs):
                    return func(*args, **kwargs)

                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    key_parts = [func.__module__, func.__name__]
                    key_parts.extend(str(arg) for arg in args)
                    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                    cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()

                result = cache.get(cache_key)
                if result is not None:
                    return result

                result = func(*args, **kwargs)
                if result is not None:
                    cache.set(cache_key, result, ttl)
                return result

            return wrapper
        return decorator


def get_redis_cache(config: Optional[RedisConfig] = None) -> RedisCache:
    """
    获取 RedisCache 实例的工厂函数

    Args:
        config: Redis 配置

    Returns:
        RedisCache: Redis 缓存客户端实例
    """
    return RedisCache(config)
