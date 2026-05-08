"""
搜索引擎基类模块
提供所有搜索引擎实现的抽象基类和通用功能
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, AsyncIterator
import asyncio
import logging
from datetime import datetime
import re


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """搜索结果数据结构
    
    用于存储单个搜索结果的所有相关信息，包括标题、URL、摘要等。
    
    Attributes:
        title: 结果标题
        url: 结果链接
        snippet: 结果摘要/描述
        source: 来源搜索引擎名称
        score: 相关性评分 (0-1)
        published_date: 发布日期 (如果可获取)
        author: 作者 (如果可获取)
        language: 语言
        metadata: 其他元数据信息
    """
    title: str
    url: str
    snippet: str
    source: str = "unknown"
    score: float = 0.0
    published_date: Optional[datetime] = None
    author: Optional[str] = None
    language: str = "zh"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """数据验证和清理"""
        self.title = self.title.strip() if self.title else ""
        self.url = self.url.strip() if self.url else ""
        self.snippet = self.snippet.strip() if self.snippet else ""
        self.score = max(0.0, min(1.0, self.score))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "score": self.score,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "author": self.author,
            "language": self.language,
            "metadata": self.metadata
        }
    
    def is_valid(self) -> bool:
        """检查结果是否有效"""
        return bool(self.title and self.url and len(self.snippet) > 10)


@dataclass
class SearchResponse:
    """搜索响应数据结构
    
    包含搜索请求的完整响应信息，包括状态、结果列表和错误信息。
    
    Attributes:
        query: 搜索查询词
        results: 搜索结果列表
        total_results: 总结果数
        page: 当前页码
        per_page: 每页结果数
        response_time: 响应时间(毫秒)
        error: 错误信息
        cached: 是否从缓存获取
    """
    query: str
    results: List[SearchResult] = field(default_factory=list)
    total_results: int = 0
    page: int = 1
    per_page: int = 10
    response_time: float = 0.0
    error: Optional[str] = None
    cached: bool = False
    
    def __post_init__(self):
        """初始化后处理"""
        if self.total_results == 0:
            self.total_results = len(self.results)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "total_results": self.total_results,
            "page": self.page,
            "per_page": self.per_page,
            "response_time": self.response_time,
            "error": self.error,
            "cached": self.cached
        }
    
    def filter_valid_results(self) -> "SearchResponse":
        """过滤出有效结果"""
        valid_results = [r for r in self.results if r.is_valid()]
        return SearchResponse(
            query=self.query,
            results=valid_results,
            total_results=len(valid_results),
            page=self.page,
            per_page=self.per_page,
            response_time=self.response_time,
            cached=self.cached
        )


class BaseSearchEngine(ABC):
    """搜索引擎抽象基类
    
    所有具体搜索引擎实现都应继承此类。
    提供同步和异步搜索接口，以及通用配置和错误处理。
    
    Example:
        class MySearchEngine(BaseSearchEngine):
            def _execute_search(self, query, max_results, **kwargs):
                # 实现具体的搜索逻辑
                pass
        
        engine = MySearchEngine()
        response = engine.search("python教程")
        async_response = await engine.async_search("python教程")
    """
    
    ENGINE_NAME: str = "base"
    DEFAULT_MAX_RESULTS: int = 10
    MAX_RETRY_ATTEMPTS: int = 3
    REQUEST_TIMEOUT: int = 30
    
    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        default_language: str = "zh-CN",
        enable_cache: bool = True,
        rate_limit_delay: float = 1.0
    ):
        """初始化搜索引擎
        
        Args:
            timeout: 请求超时时间(秒)
            max_retries: 最大重试次数
            default_language: 默认语言
            enable_cache: 是否启用缓存
            rate_limit_delay: 速率限制延迟(秒)
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_language = default_language
        self.enable_cache = enable_cache
        self.rate_limit_delay = rate_limit_delay
        self._cache: Dict[str, SearchResponse] = {}
        self._last_request_time: float = 0
    
    @abstractmethod
    def _execute_search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """执行实际搜索的核心抽象方法
        
        子类必须实现此方法来提供具体的搜索功能。
        
        Args:
            query: 搜索查询词
            max_results: 最大结果数
            **kwargs: 其他搜索引擎特定参数
            
        Returns:
            原始搜索结果列表，每个结果为字典格式
        """
        pass
    
    def _parse_result(self, raw_result: Dict[str, Any]) -> SearchResult:
        """解析原始搜索结果为SearchResult对象
        
        可在子类中重写以处理特定格式。
        
        Args:
            raw_result: 原始搜索结果字典
            
        Returns:
            解析后的SearchResult对象
        """
        return SearchResult(
            title=raw_result.get("title", ""),
            url=raw_result.get("url", raw_result.get("link", "")),
            snippet=raw_result.get("snippet", raw_result.get("description", raw_result.get("content", ""))),
            source=self.ENGINE_NAME,
            score=raw_result.get("score", 0.0),
            metadata=raw_result.get("metadata", {})
        )
    
    def _normalize_query(self, query: str) -> str:
        """规范化搜索查询
        
        清理和标准化用户输入的查询词。
        
        Args:
            query: 原始查询词
            
        Returns:
            规范化后的查询词
        """
        query = query.strip()
        query = re.sub(r'\s+', ' ', query)
        query = re.sub(r'[^\w\s\u4e00-\u9fff\-.,!?，。！？]', '', query)
        return query
    
    def _rate_limit(self):
        """速率限制控制
        
        确保请求之间有足够的延迟，避免被限流。
        """
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self._last_request_time = asyncio.get_event_loop().time()
    
    def _get_from_cache(self, cache_key: str) -> Optional[SearchResponse]:
        """从缓存获取结果
        
        Args:
            cache_key: 缓存键
            
        Returns:
            缓存的响应或None
        """
        if self.enable_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            cached.cached = True
            return cached
        return None
    
    def _save_to_cache(self, cache_key: str, response: SearchResponse):
        """保存结果到缓存
        
        Args:
            cache_key: 缓存键
            response: 要缓存的响应
        """
        if self.enable_cache:
            self._cache[cache_key] = response
            if len(self._cache) > 1000:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
    
    def _create_cache_key(self, query: str, max_results: int, **kwargs) -> str:
        """创建缓存键
        
        Args:
            query: 查询词
            max_results: 最大结果数
            **kwargs: 其他参数
            
        Returns:
            缓存键字符串
        """
        params = f"{query}:{max_results}"
        for k, v in sorted(kwargs.items()):
            params += f":{k}={v}"
        return params
    
    def search(
        self,
        query: str,
        max_results: int = 10,
        page: int = 1,
        language: Optional[str] = None,
        **kwargs
    ) -> SearchResponse:
        """执行同步搜索
        
        主要的同步搜索接口，自动处理重试、缓存和错误处理。
        
        Args:
            query: 搜索查询词
            max_results: 最大结果数 (默认10)
            page: 页码 (默认1)
            language: 搜索语言 (默认使用引擎配置)
            **kwargs: 其他搜索引擎特定参数
            
        Returns:
            SearchResponse 对象，包含搜索结果和元数据
        """
        start_time = datetime.now()
        cache_key = self._create_cache_key(query, max_results, page=page, **kwargs)
        
        cached_response = self._get_from_cache(cache_key)
        if cached_response:
            logger.info(f"返回缓存结果: {query}")
            return cached_response
        
        query = self._normalize_query(query)
        language = language or self.default_language
        
        try:
            raw_results = self._execute_search_with_retry(
                query, max_results, page, language, **kwargs
            )
            
            results = [self._parse_result(r) for r in raw_results]
            response = SearchResponse(
                query=query,
                results=results,
                total_results=len(results),
                page=page,
                per_page=max_results,
                response_time=(datetime.now() - start_time).total_seconds() * 1000
            )
            
            self._save_to_cache(cache_key, response)
            logger.info(f"搜索完成: {query}, 结果数: {len(results)}")
            
            return response
            
        except Exception as e:
            logger.error(f"搜索出错: {query}, 错误: {str(e)}")
            return SearchResponse(
                query=query,
                error=str(e),
                response_time=(datetime.now() - start_time).total_seconds() * 1000
            )
    
    def _execute_search_with_retry(
        self,
        query: str,
        max_results: int,
        page: int,
        language: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """带重试机制的搜索执行
        
        Args:
            query: 查询词
            max_results: 最大结果数
            page: 页码
            language: 语言
            **kwargs: 其他参数
            
        Returns:
            原始搜索结果列表
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                return self._execute_search(
                    query, max_results, page=page, language=language, **kwargs
                )
            except Exception as e:
                last_error = e
                logger.warning(f"搜索尝试 {attempt + 1}/{self.max_retries} 失败: {str(e)}")
                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)
        
        raise last_error or Exception("搜索失败")
    
    async def async_search(
        self,
        query: str,
        max_results: int = 10,
        page: int = 1,
        language: Optional[str] = None,
        **kwargs
    ) -> SearchResponse:
        """执行异步搜索
        
        异步版本的搜索接口，用于非阻塞操作。
        
        Args:
            query: 搜索查询词
            max_results: 最大结果数
            page: 页码
            language: 搜索语言
            **kwargs: 其他参数
            
        Returns:
            SearchResponse 对象
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.search,
            query,
            max_results,
            page,
            language
        )
    
    async def async_search_stream(
        self,
        query: str,
        max_results: int = 10,
        **kwargs
    ) -> AsyncIterator[SearchResult]:
        """异步流式搜索
        
        返回一个异步迭代器，用于逐个获取搜索结果。
        
        Args:
            query: 搜索查询词
            max_results: 最大结果数
            **kwargs: 其他参数
            
        Yields:
            SearchResult 对象
        """
        response = await self.async_search(query, max_results, **kwargs)
        for result in response.results:
            yield result
    
    def get_engine_info(self) -> Dict[str, Any]:
        """获取搜索引擎信息
        
        Returns:
            包含引擎配置的字典
        """
        return {
            "name": self.ENGINE_NAME,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "default_language": self.default_language,
            "enable_cache": self.enable_cache,
            "rate_limit_delay": self.rate_limit_delay,
            "cache_size": len(self._cache)
        }
    
    def clear_cache(self):
        """清空搜索缓存"""
        self._cache.clear()
        logger.info(f"{self.ENGINE_NAME} 缓存已清空")
    
    def validate_url(self, url: str) -> bool:
        """验证URL格式
        
        Args:
            url: 要验证的URL
            
        Returns:
            是否为有效URL
        """
        url_pattern = re.compile(
            r'^https?://'
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
            r'localhost|'
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
            r'(?::\d+)?'
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return bool(url_pattern.match(url))


class SearchEngineRegistry:
    """搜索引擎注册表
    
    用于管理和访问多个搜索引擎实例。
    支持注册、获取和自动选择最优搜索引擎。
    """
    
    _engines: Dict[str, BaseSearchEngine] = {}
    _default_engine: Optional[str] = None
    
    @classmethod
    def register(cls, name: str, engine: BaseSearchEngine, set_default: bool = False):
        """注册搜索引擎
        
        Args:
            name: 引擎名称
            engine: 引擎实例
            set_default: 是否设为默认引擎
        """
        cls._engines[name] = engine
        if set_default or not cls._default_engine:
            cls._default_engine = name
        logger.info(f"注册搜索引擎: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[BaseSearchEngine]:
        """获取指定搜索引擎
        
        Args:
            name: 引擎名称
            
        Returns:
            引擎实例或None
        """
        return cls._engines.get(name)
    
    @classmethod
    def get_default(cls) -> Optional[BaseSearchEngine]:
        """获取默认搜索引擎"""
        if cls._default_engine:
            return cls._engines.get(cls._default_engine)
        return None
    
    @classmethod
    def list_engines(cls) -> List[str]:
        """列出所有已注册的搜索引擎"""
        return list(cls._engines.keys())
    
    @classmethod
    def unregister(cls, name: str):
        """取消注册搜索引擎"""
        if name in cls._engines:
            del cls._engines[name]
            if cls._default_engine == name:
                cls._default_engine = next(iter(cls._engines), None)
            logger.info(f"取消注册搜索引擎: {name}")


class SearchEngineFactory:
    """搜索引擎工厂类
    
    提供创建不同类型搜索引擎实例的工厂方法。
    """
    
    @staticmethod
    def create(engine_type: str, **kwargs) -> BaseSearchEngine:
        """创建搜索引擎实例
        
        Args:
            engine_type: 引擎类型 ("duckduckgo", "searxng", 等)
            **kwargs: 引擎配置参数
            
        Returns:
            搜索引擎实例
            
        Raises:
            ValueError: 不支持的引擎类型
        """
        from .duckduckgo_api import DuckDuckGoAPI
        from .searxng_engine import SearXNGEngine
        
        engines = {
            "duckduckgo": DuckDuckGoAPI,
            "searxng": SearXNGEngine,
        }
        
        engine_class = engines.get(engine_type.lower())
        if not engine_class:
            raise ValueError(f"不支持的搜索引擎类型: {engine_type}")
        
        return engine_class(**kwargs)
