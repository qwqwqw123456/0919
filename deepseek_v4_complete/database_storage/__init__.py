"""
数据库存储模块
整合 MySQL、Redis、向量数据库、备份恢复等功能
"""

from mysql_connect import (
    MySQLClient,
    DatabaseConfig,
    TransactionManager,
    QueryBuilder,
    Base,
    get_mysql_client
)

from redis_cache import (
    RedisCache,
    RedisConfig,
    DistributedLock,
    RateLimiter,
    PubSubManager,
    CacheDecorator,
    SerializationType,
    Serializer,
    CacheStats,
    get_redis_cache
)

from vector_database import (
    VectorDBConfig,
    FAISSVectorDB,
    MilvusVectorDB,
    VectorDBManager,
    VectorOperations,
    VectorRecord,
    SearchResult,
    IndexStats,
    VectorIndexType,
    DistanceMetric,
    BaseVectorDB,
    get_vector_db
)

from data_backup_restore import (
    BackupRestore,
    BackupConfig,
    BackupMetadata,
    RestoreMetadata,
    BackupStatus,
    BackupType,
    CompressionType,
    create_mysql_backup,
    restore_mysql_backup
)

from table_struct import (
    User,
    Token,
    Session,
    Message,
    VectorStore,
    AuditLog,
    Config,
    Model,
    FileStorage,
    RateLimit,
    Webhook,
    WebhookDelivery,
    Collection,
    Tool,
    UserStatus,
    MessageRole,
    SessionStatus,
    TokenType,
    AuditAction,
    create_all_tables,
    drop_all_tables,
    get_table_names
)


__version__ = "1.0.0"
__all__ = [
    "MySQLClient",
    "DatabaseConfig",
    "TransactionManager",
    "QueryBuilder",
    "Base",
    "get_mysql_client",
    "RedisCache",
    "RedisConfig",
    "DistributedLock",
    "RateLimiter",
    "PubSubManager",
    "CacheDecorator",
    "SerializationType",
    "Serializer",
    "CacheStats",
    "get_redis_cache",
    "VectorDBConfig",
    "FAISSVectorDB",
    "MilvusVectorDB",
    "VectorDBManager",
    "VectorOperations",
    "VectorRecord",
    "SearchResult",
    "IndexStats",
    "VectorIndexType",
    "DistanceMetric",
    "BaseVectorDB",
    "get_vector_db",
    "BackupRestore",
    "BackupConfig",
    "BackupMetadata",
    "RestoreMetadata",
    "BackupStatus",
    "BackupType",
    "CompressionType",
    "create_mysql_backup",
    "restore_mysql_backup",
    "User",
    "Token",
    "Session",
    "Message",
    "VectorStore",
    "AuditLog",
    "Config",
    "Model",
    "FileStorage",
    "RateLimit",
    "Webhook",
    "WebhookDelivery",
    "Collection",
    "Tool",
    "UserStatus",
    "MessageRole",
    "SessionStatus",
    "TokenType",
    "AuditAction",
    "create_all_tables",
    "drop_all_tables",
    "get_table_names"
]
