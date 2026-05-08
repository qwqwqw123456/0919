"""
MySQL 数据库连接和操作模块
使用 SQLAlchemy 实现完整的数据库连接、CRUD 操作、事务管理和连接池
"""

import os
import json
from typing import Optional, List, Dict, Any, Union, Tuple
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    create_engine, Engine, Connection, MetaData, Table, Column,
    Integer, String, Text, Float, Boolean, DateTime, Date,
    JSON, Enum as SQLEnum, ForeignKey, Index, UniqueConstraint,
    and_, or_, not_, func, case, text, inspect
)
from sqlalchemy.orm import (
    Session, sessionmaker, DeclarativeBase, relationship,
    mapped_column, Mapped
)
from sqlalchemy.pool import QueuePool, NullPool, Pool
from sqlalchemy.exc import (
    SQLAlchemyError, IntegrityError, OperationalError,
    DataError, ProgrammingError
)


class Base(DeclarativeBase):
    """SQLAlchemy ORM 基类"""
    pass


class DatabaseConfig:
    """数据库配置管理类"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        user: str = "root",
        password: str = "",
        database: str = "deepseek_v4",
        charset: str = "utf8mb4",
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_recycle: int = 3600,
        pool_pre_ping: bool = True,
        echo: bool = False
    ):
        """
        初始化数据库配置

        Args:
            host: 数据库主机地址
            port: 数据库端口
            user: 数据库用户名
            password: 数据库密码
            database: 数据库名称
            charset: 字符编码
            pool_size: 连接池大小
            max_overflow: 最大溢出连接数
            pool_recycle: 连接回收时间(秒)
            pool_pre_ping: 使用前检测连接
            echo: 是否打印 SQL 语句
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.charset = charset
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_recycle = pool_recycle
        self.pool_pre_ping = pool_pre_ping
        self.echo = echo

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """从环境变量加载配置"""
        return cls(
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DATABASE", "deepseek_v4"),
            charset=os.getenv("MYSQL_CHARSET", "utf8mb4"),
            pool_size=int(os.getenv("MYSQL_POOL_SIZE", "5")),
            max_overflow=int(os.getenv("MYSQL_MAX_OVERFLOW", "10")),
            echo=os.getenv("MYSQL_ECHO", "false").lower() == "true"
        )

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "DatabaseConfig":
        """从字典加载配置"""
        return cls(**config)

    @classmethod
    def from_json(cls, json_path: str) -> "DatabaseConfig":
        """从 JSON 文件加载配置"""
        with open(json_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return cls.from_dict(config)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "database": self.database,
            "charset": self.charset,
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_recycle": self.pool_recycle,
            "pool_pre_ping": self.pool_pre_ping,
            "echo": self.echo
        }

    def get_connection_url(self) -> str:
        """生成 SQLAlchemy 连接 URL"""
        return (
            f"mysql+pymysql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
            f"?charset={self.charset}"
        )


class MySQLClient:
    """
    MySQL 数据库客户端
    提供完整的数据库连接、CRUD 操作、事务管理和连接池功能
    """

    def __init__(
        self,
        config: Optional[DatabaseConfig] = None,
        engine: Optional[Engine] = None
    ):
        """
        初始化 MySQL 客户端

        Args:
            config: 数据库配置对象
            engine: 已有的 SQLAlchemy 引擎(优先使用)
        """
        self.config = config or DatabaseConfig()
        self._engine: Optional[Engine] = engine
        self._session_factory: Optional[sessionmaker] = None
        self._metadata: Optional[MetaData] = None

    @property
    def engine(self) -> Engine:
        """获取或创建数据库引擎"""
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine

    @property
    def session_factory(self) -> sessionmaker:
        """获取或创建会话工厂"""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
        return self._session_factory

    @property
    def metadata(self) -> MetaData:
        """获取或创建元数据对象"""
        if self._metadata is None:
            self._metadata = MetaData(bind=self.engine)
        return self._metadata

    def _create_engine(self) -> Engine:
        """创建数据库引擎"""
        return create_engine(
            self.config.get_connection_url(),
            poolclass=QueuePool,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_recycle=self.config.pool_recycle,
            pool_pre_ping=self.config.pool_pre_ping,
            echo=self.config.echo
        )

    @contextmanager
    def get_session(self) -> Session:
        """
        获取数据库会话的上下文管理器
        自动处理提交和回滚

        Yields:
            Session: 数据库会话对象
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    @contextmanager
    def get_connection(self) -> Connection:
        """
        获取原始连接的上下文管理器

        Yields:
            Connection: 数据库连接对象
        """
        with self.engine.connect() as conn:
            yield conn

    def create_database_if_not_exists(self):
        """创建数据库(如果不存在)"""
        temp_url = (
            f"mysql+pymysql://{self.config.user}:{self.config.password}"
            f"@{self.config.host}:{self.config.port}/"
        )
        temp_engine = create_engine(temp_url)
        with temp_engine.connect() as conn:
            conn.execute(
                text(f"CREATE DATABASE IF NOT EXISTS {self.config.database}")
            )
            conn.commit()
        temp_engine.dispose()

    def create_all_tables(self, base: type = Base):
        """
        创建所有表

        Args:
            base: SQLAlchemy 声明基类
        """
        base.metadata.create_all(self.engine)

    def drop_all_tables(self, base: type = Base):
        """删除所有表"""
        base.metadata.drop_all(self.engine)

    def create_table(self, table: Table):
        """创建单个表"""
        table.create(self.engine, checkfirst=True)

    def drop_table(self, table: Table):
        """删除单个表"""
        table.drop(self.engine, checkfirst=True)

    def execute_raw_sql(self, sql: str, params: Optional[Dict] = None) -> ResultProxy:
        """
        执行原生 SQL 语句

        Args:
            sql: SQL 语句
            params: 参数字典

        Returns:
            ResultProxy: 查询结果
        """
        with self.get_connection() as conn:
            result = conn.execute(text(sql), params or {})
            conn.commit()
            return result

    def execute_scalar(self, sql: str, params: Optional[Dict] = None) -> Any:
        """执行 SQL 并返回标量值"""
        with self.get_connection() as conn:
            result = conn.execute(text(sql), params or {})
            row = result.fetchone()
            return row[0] if row else None

    def get_table_names(self) -> List[str]:
        """获取所有表名"""
        with self.get_connection() as conn:
            result = conn.execute(
                text("SHOW TABLES")
            )
            return [row[0] for row in result]

    def get_table_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """获取表字段信息"""
        with self.get_connection() as conn:
            result = conn.execute(
                text(f"SHOW FULL COLUMNS FROM `{table_name}`")
            )
            columns = []
            for row in result:
                columns.append({
                    "field": row[0],
                    "type": row[1],
                    "collation": row[2],
                    "null": row[3],
                    "key": row[4],
                    "default": row[5],
                    "extra": row[6],
                    "privileges": row[7],
                    "comment": row[8]
                })
            return columns

    def table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        with self.get_connection() as conn:
            result = conn.execute(
                text("SHOW TABLES LIKE :table_name"),
                {"table_name": table_name}
            )
            return result.fetchone() is not None

    def get_row_count(self, table_name: str) -> int:
        """获取表的行数"""
        return self.execute_scalar(f"SELECT COUNT(*) FROM `{table_name}`")

    def truncate_table(self, table_name: str):
        """清空表数据"""
        self.execute_raw_sql(f"TRUNCATE TABLE `{table_name}`")

    def insert(self, table_name: str, data: Dict[str, Any]) -> int:
        """
        插入单条数据

        Args:
            table_name: 表名
            data: 要插入的数据字典

        Returns:
            int: 插入的 ID
        """
        columns = list(data.keys())
        placeholders = [f":{col}" for col in columns]
        sql = f"INSERT INTO `{table_name}` ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        result = self.execute_raw_sql(sql, data)
        return result.lastrowid

    def insert_many(self, table_name: str, data_list: List[Dict[str, Any]]) -> int:
        """
        批量插入数据

        Args:
            table_name: 表名
            data_list: 要插入的数据列表

        Returns:
            int: 影响的行数
        """
        if not data_list:
            return 0

        columns = list(data_list[0].keys())
        placeholders = [f":{col}" for col in columns]
        sql = f"INSERT INTO `{table_name}` ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"

        with self.get_connection() as conn:
            result = conn.execute(text(sql), data_list)
            conn.commit()
            return result.rowcount

    def update(
        self,
        table_name: str,
        data: Dict[str, Any],
        where: Dict[str, Any]
    ) -> int:
        """
        更新数据

        Args:
            table_name: 表名
            data: 要更新的数据
            where: 更新条件

        Returns:
            int: 影响的行数
        """
        set_clause = ", ".join([f"{col} = :{col}" for col in data.keys()])
        where_clause = " AND ".join([f"{col} = :where_{col}" for col in where.keys()])

        params = {**data, **{f"where_{k}": v for k, v in where.items()}}
        sql = f"UPDATE `{table_name}` SET {set_clause} WHERE {where_clause}"

        result = self.execute_raw_sql(sql, params)
        return result.rowcount

    def delete(self, table_name: str, where: Dict[str, Any]) -> int:
        """
        删除数据

        Args:
            table_name: 表名
            where: 删除条件

        Returns:
            int: 影响的行数
        """
        where_clause = " AND ".join([f"{col} = :{col}" for col in where.keys()])
        sql = f"DELETE FROM `{table_name}` WHERE {where_clause}"

        result = self.execute_raw_sql(sql, where)
        return result.rowcount

    def select(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        order_by: Optional[Dict[str, str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        查询数据

        Args:
            table_name: 表名
            columns: 要查询的字段列表
            where: 查询条件
            order_by: 排序字段 {"field": "asc"} 或 {"field": "desc"}
            limit: 返回记录数
            offset: 偏移量

        Returns:
            List[Dict]: 查询结果列表
        """
        col_str = ", ".join(columns) if columns else "*"
        sql = f"SELECT {col_str} FROM `{table_name}`"
        params = {}

        if where:
            where_clause = " AND ".join([f"{col} = :{col}" for col in where.keys()])
            sql += f" WHERE {where_clause}"
            params.update(where)

        if order_by:
            order_parts = [f"{field} {direction}" for field, direction in order_by.items()]
            sql += f" ORDER BY {', '.join(order_parts)}"

        if limit is not None:
            sql += f" LIMIT :limit"
            params["limit"] = limit

        if offset is not None:
            sql += f" OFFSET :offset"
            params["offset"] = offset

        with self.get_connection() as conn:
            result = conn.execute(text(sql), params)
            columns_names = result.keys()
            return [dict(zip(columns_names, row)) for row in result]

    def select_one(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """查询单条数据"""
        results = self.select(table_name, columns, where, limit=1)
        return results[0] if results else None

    def select_by_sql(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        执行查询 SQL 并返回结果

        Args:
            sql: SQL 查询语句
            params: 查询参数

        Returns:
            List[Dict]: 查询结果
        """
        with self.get_connection() as conn:
            result = conn.execute(text(sql), params or {})
            columns_names = result.keys()
            return [dict(zip(columns_names, row)) for row in result]

    def count(self, table_name: str, where: Optional[Dict[str, Any]] = None) -> int:
        """统计记录数"""
        sql = f"SELECT COUNT(*) FROM `{table_name}`"
        params = {}

        if where:
            where_clause = " AND ".join([f"{col} = :{col}" for col in where.keys()])
            sql += f" WHERE {where_clause}"
            params.update(where)

        return self.execute_scalar(sql, params) or 0

    def exists(self, table_name: str, where: Dict[str, Any]) -> bool:
        """检查记录是否存在"""
        return self.count(table_name, where) > 0

    def begin_transaction(self) -> Connection:
        """开始事务"""
        conn = self.engine.connect()
        conn.begin()
        return conn

    def commit(self, conn: Connection):
        """提交事务"""
        conn.commit()

    def rollback(self, conn: Connection):
        """回滚事务"""
        conn.rollback()

    def ping(self) -> bool:
        """检测数据库连接是否正常"""
        try:
            self.execute_scalar("SELECT 1")
            return True
        except Exception:
            return False

    def close(self):
        """关闭引擎"""
        if self._engine:
            self._engine.dispose()
            self._engine = None

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close()


class TransactionManager:
    """事务管理器"""

    def __init__(self, client: MySQLClient):
        self.client = client
        self._conn: Optional[Connection] = None
        self._in_transaction = False

    def begin(self):
        """开始事务"""
        self._conn = self.client.begin_transaction()
        self._in_transaction = True

    def commit(self):
        """提交事务"""
        if self._conn and self._in_transaction:
            self.client.commit(self._conn)
            self._in_transaction = False

    def rollback(self):
        """回滚事务"""
        if self._conn and self._in_transaction:
            self.client.rollback(self._conn)
            self._in_transaction = False

    def execute(self, sql: str, params: Optional[Dict] = None):
        """在事务中执行 SQL"""
        if not self._in_transaction:
            raise RuntimeError("事务未开始")
        return self._conn.execute(text(sql), params or {})

    def __enter__(self):
        self.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.rollback()
        else:
            self.commit()


class QueryBuilder:
    """
    SQL 查询构建器
    提供链式调用的查询构建方式
    """

    def __init__(self, client: MySQLClient, table_name: str):
        self.client = client
        self.table_name = table_name
        self._columns: Optional[List[str]] = None
        self._where_clauses: List[str] = []
        self._where_params: Dict[str, Any] = {}
        self._order_by: List[str] = []
        self._limit_val: Optional[int] = None
        self._offset_val: Optional[int] = None
        self._joins: List[str] = []
        self._group_by: List[str] = []
        self._having_clauses: List[str] = []

    def select(self, *columns: str) -> "QueryBuilder":
        """选择字段"""
        self._columns = list(columns)
        return self

    def join(
        self,
        table: str,
        on: str,
        join_type: str = "INNER"
    ) -> "QueryBuilder":
        """添加 JOIN"""
        self._joins.append(f"{join_type} JOIN {table} ON {on}")
        return self

    def left_join(self, table: str, on: str) -> "QueryBuilder":
        """添加 LEFT JOIN"""
        return self.join(table, on, "LEFT")

    def right_join(self, table: str, on: str) -> "QueryBuilder":
        """添加 RIGHT JOIN"""
        return self.join(table, on, "RIGHT")

    def where(self, condition: str, **params) -> "QueryBuilder":
        """添加 WHERE 条件"""
        self._where_clauses.append(condition)
        self._where_params.update(params)
        return self

    def where_in(self, field: str, values: List) -> "QueryBuilder":
        """添加 WHERE IN 条件"""
        placeholders = ", ".join([f":{field}_{i}" for i in range(len(values))])
        self._where_clauses.append(f"{field} IN ({placeholders})")
        for i, val in enumerate(values):
            self._where_params[f"{field}_{i}"] = val
        return self

    def where_between(
        self,
        field: str,
        low: Any,
        high: Any
    ) -> "QueryBuilder":
        """添加 WHERE BETWEEN 条件"""
        self._where_clauses.append(f"{field} BETWEEN :{field}_low AND :{field}_high")
        self._where_params[f"{field}_low"] = low
        self._where_params[f"{field}_high"] = high
        return self

    def order_by(self, field: str, direction: str = "ASC") -> "QueryBuilder":
        """添加排序"""
        self._order_by.append(f"{field} {direction}")
        return self

    def group_by(self, *fields: str) -> "QueryBuilder":
        """添加分组"""
        self._group_by.extend(fields)
        return self

    def having(self, condition: str, **params) -> "QueryBuilder":
        """添加 HAVING 条件"""
        self._having_clauses.append(condition)
        self._where_params.update(params)
        return self

    def limit(self, count: int) -> "QueryBuilder":
        """限制返回数量"""
        self._limit_val = count
        return self

    def offset(self, count: int) -> "QueryBuilder":
        """设置偏移量"""
        self._offset_val = count
        return self

    def build_select(self) -> Tuple[str, Dict]:
        """构建 SELECT SQL"""
        col_str = ", ".join(self._columns) if self._columns else "*"
        sql_parts = [f"SELECT {col_str} FROM `{self.table_name}`"]

        if self._joins:
            sql_parts.extend(self._joins)

        if self._where_clauses:
            sql_parts.append(f"WHERE {' AND '.join(self._where_clauses)}")

        if self._group_by:
            sql_parts.append(f"GROUP BY {', '.join(self._group_by)}")

        if self._having_clauses:
            sql_parts.append(f"HAVING {' AND '.join(self._having_clauses)}")

        if self._order_by:
            sql_parts.append(f"ORDER BY {', '.join(self._order_by)}")

        if self._limit_val is not None:
            sql_parts.append(f"LIMIT {self._limit_val}")

        if self._offset_val is not None:
            sql_parts.append(f"OFFSET {self._offset_val}")

        return " ".join(sql_parts), self._where_params

    def all(self) -> List[Dict[str, Any]]:
        """执行查询并返回所有结果"""
        sql, params = self.build_select()
        return self.client.select_by_sql(sql, params)

    def first(self) -> Optional[Dict[str, Any]]:
        """执行查询并返回第一条结果"""
        results = self.limit(1).all()
        return results[0] if results else None

    def count_all(self) -> int:
        """统计总数"""
        original_columns = self._columns
        self._columns = ["COUNT(*) as cnt"]
        sql, params = self.build_select()
        self._columns = original_columns

        with self.client.get_connection() as conn:
            result = conn.execute(text(sql), params)
            row = result.fetchone()
            return row[0] if row else 0


def get_mysql_client(config: Optional[DatabaseConfig] = None) -> MySQLClient:
    """
    获取 MySQL 客户端实例的工厂函数

    Args:
        config: 数据库配置

    Returns:
        MySQLClient: 数据库客户端实例
    """
    return MySQLClient(config)
