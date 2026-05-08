"""
SearXNG 搜索引擎引擎集成模块

SearXNG 是一个隐私友好的元搜索引擎，支持多种搜索类别。
本模块提供对 SearXNG 实例的完整封装，支持异步调用和错误处理。

依赖安装:
    pip install requests aiohttp

使用示例:
    # 基本使用
    engine = SearXNGEngine()
    results = engine.search("Python 教程")

    # 自定义实例
    engine = SearXNGEngine(url="https://searx.example.com")

    # 异步使用
    results = await engine.async_search("Python 教程")

    # 搜索特定类别
    results = engine.search("Python", categories=["news", "it"])
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Union

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from .search_engine_base import (
    BaseSearchEngine,
    SearchEngineFactory,
    SearchEngineRegistry,
    SearchResponse,
    SearchResult,
)

logger = logging.getLogger(__name__)


class SearXNGCategory(Enum):
    """SearXNG 支持的搜索类别枚举"""

    GENERAL = "general"
    NEWS = "news"
    MAPS = "maps"
    MUSIC = "music"
    SOCIAL_MEDIA = "social media"
    VIDEOS = "videos"
    IMAGES = "images"
    FILES = "files"
    IT = "it"
    SCIENCE = "science"


class SearXNGEngineType(Enum):
    """SearXNG 引擎类型"""

    WEB = "web"
    NEWS = "news"
    IMAGES = "images"
    VIDEOS = "videos"
    MAPS = "maps"
    FILES = "files"


@dataclass
class SearXNGConfig:
    """SearXNG 配置类

    Attributes:
        url: SearXNG 实例的基础 URL
        timeout: 请求超时时间（秒）
        max_retries: 最大重试次数
        retry_delay: 重试延迟时间（秒）
        verify_ssl: 是否验证 SSL 证书
        use_proxy: 是否使用代理
        proxy_url: 代理 URL
        categories: 默认搜索类别
        engines: 指定使用的搜索引擎列表
        language: 搜索语言 (如 "zh-CN", "en-US", "auto")
        safe_search: 安全搜索级别 (0=关闭, 1=适中, 2=严格)
        time_range: 时间范围 ("day", "week", "month", "year")
    """

    url: str = "http://localhost:8080"
    timeout: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0
    verify_ssl: bool = True
    use_proxy: bool = False
    proxy_url: Optional[str] = None
    categories: List[str] = field(default_factory=lambda: ["general"])
    engines: Optional[List[str]] = None
    language: str = "auto"
    safe_search: int = 0
    time_range: Optional[str] = None


@dataclass
class SearXNGResult:
    """SearXNG 单条搜索结果

    Attributes:
        title: 结果标题
        url: 结果链接
        snippet: 结果摘要/内容
        engine: 来源搜索引擎名称
        engine_type: 引擎类型 (web, news, images, videos 等)
        thumbnail: 缩略图 URL（图片/视频结果）
        img_src: 图片源 URL
        video_src: 视频源 URL
        iframe_src: iframe 嵌入 URL
        published_date: 发布时间
        author: 作者
        latitude: 纬度（地图结果）
        longitude: 经度（地图结果）
        template: 结果模板类型
    """

    title: str
    url: str
    snippet: str = ""
    engine: Optional[str] = None
    engine_type: Optional[str] = None
    thumbnail: Optional[str] = None
    img_src: Optional[str] = None
    video_src: Optional[str] = None
    iframe_src: Optional[str] = None
    published_date: Optional[str] = None
    author: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    template: Optional[str] = None

    def to_search_result(self) -> SearchResult:
        """转换为通用 SearchResult 格式"""
        return SearchResult(
            engine=f"searxng_{self.engine}" if self.engine else "searxng",
            title=self.title,
            url=self.url,
            snippet=self.snippet,
            source=self.engine,
        )


@SearchEngineRegistry.register("searxng")
class SearXNGEngine(BaseSearchEngine):
    """
    SearXNG 搜索引擎引擎实现类

    继承自 BaseSearchEngine，提供完整的 SearXNG 搜索功能封装。
    SearXNG 是一个开源的隐私友好型元搜索引擎，支持聚合多个搜索引擎的结果。

    Features:
        - 多类别搜索支持（新闻、图片、视频、地图等）
        - 异步搜索支持
        - 自动重试机制
        - 灵活的搜索引擎选择
        - 结果缓存
        - SSL/TLS 配置
        - 代理支持
        - 时间范围过滤

    Example:
        >>> engine = SearXNGEngine()
        >>> # 基本搜索
        >>> results = engine.search("人工智能")
        >>> # 类别搜索
        >>> news = engine.search("科技新闻", categories=["news"])
        >>> # 异步搜索
        >>> results = await engine.async_search("Python 教程")
    """

    _SUPPORTED_CATEGORIES: Set[str] = {
        "general", "news", "maps", "music", "social media",
        "videos", "images", "files", "it", "science"
    }

    _SUPPORTED_ENGINES: Set[str] = {
        "google", "bing", "duckduckgo", "yandex", "baidu",
        "wikipedia", "arxiv", "github", "youtube", "twitter"
    }

    def __init__(
        self,
        config: Optional[SearXNGConfig] = None,
        enable_cache: bool = True,
        cache_size: int = 100,
        enable_rate_limit: bool = True,
    ):
        """
        初始化 SearXNG 引擎客户端

        Args:
            config: SearXNG 配置对象，若为 None 则使用默认配置
            enable_cache: 是否启用结果缓存
            cache_size: 缓存最大容量
            enable_rate_limit: 是否启用速率限制保护
        """
        super().__init__(
            engine_name="searxng",
            enable_cache=enable_cache,
            cache_size=cache_size,
            enable_rate_limit=enable_rate_limit,
        )

        self.config = config or SearXNGConfig()
        self._last_request_time = 0.0
        self._min_request_interval = 0.5

        self._session = None
        self._aiohttp_session = None

    def _get_session(self) -> requests.Session:
        """
        获取或创建 HTTP 会话

        使用会话可以复用连接，提高性能

        Returns:
            requests.Session: HTTP 会话对象
        """
        if self._session is None:
            self._session = requests.Session()
            if self.config.use_proxy and self.config.proxy_url:
                self._session.proxies = {
                    "http": self.config.proxy_url,
                    "https": self.config.proxy_url,
                }

        return self._session

    def _check_rate_limit(self) -> None:
        """
        检查并实施速率限制
        """
        if not self.enable_rate_limit:
            return

        current_time = time.time()
        elapsed = current_time - self._last_request_time

        if elapsed < self._min_request_interval:
            sleep_time = self._min_request_interval - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self._last_request_time = time.time()

    async def _async_check_rate_limit(self) -> None:
        """
        异步检查并实施速率限制
        """
        if not self.enable_rate_limit:
            return

        current_time = time.time()
        elapsed = current_time - self._last_request_time

        if elapsed < self._min_request_interval:
            sleep_time = self._min_request_interval - elapsed
            logger.debug(f"Async rate limiting: sleeping {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)

        self._last_request_time = time.time()

    def _parse_results(
        self,
        raw_results: List[Dict[str, Any]],
        response_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearXNGResult]:
        """
        解析 SearXNG 原始搜索结果

        Args:
            raw_results: SearXNG API 返回的原始结果列表
            response_metadata: 响应的元数据

        Returns:
            解析后的 SearXNGResult 列表
        """
        results = []

        for item in raw_results:
            try:
                result = SearXNGResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", item.get("snippet", "")),
                    engine=item.get("engine", ""),
                    engine_type=item.get("engine_type", item.get("typename", "")),
                    thumbnail=item.get("thumbnail", ""),
                    img_src=item.get("img_src", item.get("thumbnail", "")),
                    video_src=item.get("video_src", ""),
                    iframe_src=item.get("iframe_src", ""),
                    published_date=item.get("publishedDate", item.get("published_date", "")),
                    author=item.get("author", ""),
                    latitude=item.get("geo", {}).get("lat") if isinstance(item.get("geo"), dict) else item.get("latitude"),
                    longitude=item.get("geo", {}).get("lon") if isinstance(item.get("geo"), dict) else item.get("longitude"),
                    template=item.get("template", ""),
                )
                results.append(result)

            except Exception as e:
                logger.warning(f"Failed to parse search result item: {e}")
                continue

        return results

    def _validate_categories(self, categories: List[str]) -> List[str]:
        """
        验证并规范化类别列表

        Args:
            categories: 输入的类别列表

        Returns:
            验证后的类别列表
        """
        valid_categories = []
        for cat in categories:
            cat_lower = cat.lower().strip()
            if cat_lower in self._SUPPORTED_CATEGORIES:
                valid_categories.append(cat_lower)
            else:
                logger.warning(f"Unsupported category '{cat}', skipping")

        if not valid_categories:
            valid_categories = ["general"]

        return valid_categories

    def _validate_engines(self, engines: List[str]) -> List[str]:
        """
        验证并规范化引擎列表

        Args:
            engines: 输入的引擎列表

        Returns:
            验证后的引擎列表
        """
        if not engines:
            return []

        valid_engines = []
        for eng in engines:
            eng_lower = eng.lower().strip()
            if eng_lower in self._SUPPORTED_ENGINES:
                valid_engines.append(eng_lower)
            else:
                logger.warning(f"Unsupported engine '{eng}', skipping")

        return valid_engines

    def _build_search_params(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        engines: Optional[List[str]] = None,
        language: Optional[str] = None,
        safe_search: Optional[int] = None,
        time_range: Optional[str] = None,
        page: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        构建搜索请求参数

        Args:
            query: 搜索查询词
            categories: 搜索类别
            engines: 指定引擎
            language: 搜索语言
            safe_search: 安全搜索级别
            time_range: 时间范围
            page: 页码
            **kwargs: 额外参数

        Returns:
            搜索参数字典
        """
        params = {
            "q": query,
            "format": "json",
            "pageno": page,
        }

        valid_categories = self._validate_categories(
            categories or self.config.categories
        )
        if valid_categories:
            params["categories"] = ",".join(valid_categories)

        valid_engines = self._validate_engines(
            engines or (self.config.engines or [])
        )
        if valid_engines:
            params["engines"] = ",".join(valid_engines)

        if language or self.config.language:
            params["language"] = language or self.config.language

        if safe_search is not None or self.config.safe_search:
            params["safesearch"] = safe_search if safe_search is not None else self.config.safe_search

        if time_range or self.config.time_range:
            params["time_range"] = time_range or self.config.time_range

        params.update(kwargs)

        return params

    def _execute_search_with_retry(
        self,
        query: str,
        max_results: int,
        categories: Optional[List[str]] = None,
        engines: Optional[List[str]] = None,
        language: Optional[str] = None,
        safe_search: Optional[int] = None,
        time_range: Optional[str] = None,
        page: int = 1,
        **kwargs,
    ) -> tuple[List[SearXNGResult], Dict[str, Any]]:
        """
        带重试机制的搜索执行

        Args:
            query: 搜索查询词
            max_results: 最大结果数
            categories: 搜索类别
            engines: 指定引擎
            language: 搜索语言
            safe_search: 安全搜索级别
            time_range: 时间范围
            page: 页码
            **kwargs: 额外参数

        Returns:
            (搜索结果列表, 响应元数据) 元组

        Raises:
            Exception: 当所有重试都失败时
        """
        last_exception = None

        for attempt in range(self.config.max_retries):
            try:
                self._check_rate_limit()

                params = self._build_search_params(
                    query=query,
                    categories=categories,
                    engines=engines,
                    language=language,
                    safe_search=safe_search,
                    time_range=time_range,
                    page=page,
                    **kwargs,
                )

                session = self._get_session()
                response = session.get(
                    f"{self.config.url}/search",
                    params=params,
                    timeout=self.config.timeout,
                    verify=self.config.verify_ssl,
                )

                response.raise_for_status()
                data = response.json()

                results = self._parse_results(
                    data.get("results", []),
                    data.get("infoboxes", []),
                )

                metadata = {
                    "total_results": data.get("number_of_results", len(results)),
                    "page": page,
                    "query": query,
                    "categories": categories or self.config.categories,
                    "engines": engines or self.config.engines or [],
                    "result_count": len(data.get("results", [])),
                }

                return results[:max_results], metadata

            except requests.exceptions.Timeout as e:
                last_exception = e
                logger.warning(
                    f"Timeout (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2**attempt))

            except requests.exceptions.HTTPError as e:
                last_exception = e
                if e.response.status_code == 429:
                    wait_time = self.config.retry_delay * (2**attempt)
                    logger.warning(
                        f"Rate limit (attempt {attempt + 1}/{self.config.max_retries}), "
                        f"waiting {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)
                else:
                    logger.warning(
                        f"HTTP error (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                    )
                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay)

            except requests.exceptions.RequestException as e:
                last_exception = e
                logger.warning(
                    f"Request error (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2**attempt))

            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error during search: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)

        logger.error(f"All {self.config.max_retries} attempts failed")
        raise Exception(
            f"Search failed after {self.config.max_retries} attempts: {last_exception}"
        )

    def search(
        self,
        query: str,
        max_results: int = 10,
        categories: Optional[List[str]] = None,
        engines: Optional[List[str]] = None,
        language: Optional[str] = None,
        safe_search: Optional[int] = None,
        time_range: Optional[str] = None,
        page: int = 1,
        **kwargs,
    ) -> SearchResponse:
        """
        执行同步搜索

        Args:
            query: 搜索查询词
            max_results: 最大返回结果数，默认为 10
            categories: 搜索类别列表
            engines: 指定使用的搜索引擎列表
            language: 搜索语言
            safe_search: 安全搜索级别 (0=关闭, 1=适中, 2=严格)
            time_range: 时间范围 ("day", "week", "month", "year")
            page: 页码
            **kwargs: 传递给底层搜索API的额外参数

        Returns:
            SearchResponse: 搜索响应对象，包含结果列表和元数据

        Raises:
            Exception: 搜索失败时抛出

        Example:
            >>> engine = SearXNGEngine()
            >>> response = engine.search("Python 教程")
            >>> for result in response.results:
            ...     print(f"{result.title}: {result.url}")
        """
        cache_key = self._get_cache_key(
            query, max_results, str(categories), str(engines)
        )
        cached_response = self._get_from_cache(cache_key)
        if cached_response:
            logger.debug(f"Cache hit for query: {query}")
            return cached_response

        start_time = time.time()

        try:
            results, metadata = self._execute_search_with_retry(
                query=query,
                max_results=max_results,
                categories=categories,
                engines=engines,
                language=language,
                safe_search=safe_search,
                time_range=time_range,
                page=page,
                **kwargs,
            )

            search_results = [r.to_search_result() for r in results]
            response = SearchResponse(
                query=query,
                results=search_results,
                total_results=len(search_results),
                engine=self.engine_name,
                metadata=metadata,
            )

            self._add_to_cache(cache_key, response)
            response.elapsed_time = time.time() - start_time

            return response

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                engine=self.engine_name,
                error=str(e),
            )

    async def async_search(
        self,
        query: str,
        max_results: int = 10,
        categories: Optional[List[str]] = None,
        engines: Optional[List[str]] = None,
        language: Optional[str] = None,
        safe_search: Optional[int] = None,
        time_range: Optional[str] = None,
        page: int = 1,
        **kwargs,
    ) -> SearchResponse:
        """
        执行异步搜索

        Args:
            query: 搜索查询词
            max_results: 最大返回结果数，默认为 10
            categories: 搜索类别列表
            engines: 指定使用的搜索引擎列表
            language: 搜索语言
            safe_search: 安全搜索级别
            time_range: 时间范围
            page: 页码
            **kwargs: 额外参数

        Returns:
            SearchResponse: 搜索响应对象

        Example:
            >>> engine = SearXNGEngine()
            >>> response = await engine.async_search("Python 教程")
            >>> print(f"Found {response.total_results} results")
        """
        cache_key = self._get_cache_key(
            query, max_results, str(categories), str(engines)
        )
        cached_response = self._get_from_cache(cache_key)
        if cached_response:
            logger.debug(f"Async cache hit for query: {query}")
            return cached_response

        start_time = time.time()

        try:
            await self._async_check_rate_limit()

            params = self._build_search_params(
                query=query,
                categories=categories,
                engines=engines,
                language=language,
                safe_search=safe_search,
                time_range=time_range,
                page=page,
                **kwargs,
            )

            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config.url}/search",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                    ssl=None if self.config.verify_ssl else False,
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

            results = self._parse_results(data.get("results", []))
            metadata = {
                "total_results": data.get("number_of_results", len(results)),
                "page": page,
                "query": query,
                "categories": categories or self.config.categories,
            }

            search_results = [r.to_search_result() for r in results[:max_results]]
            response = SearchResponse(
                query=query,
                results=search_results,
                total_results=len(search_results),
                engine=self.engine_name,
                metadata=metadata,
            )

            self._add_to_cache(cache_key, response)
            response.elapsed_time = time.time() - start_time

            return response

        except Exception as e:
            logger.error(f"Async search failed: {e}")
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                engine=self.engine_name,
                error=str(e),
            )

    async def async_search_stream(
        self,
        query: str,
        max_results: int = 10,
        categories: Optional[List[str]] = None,
        engines: Optional[List[str]] = None,
        language: Optional[str] = None,
        safe_search: Optional[int] = None,
        time_range: Optional[str] = None,
        page: int = 1,
        **kwargs,
    ) -> AsyncGenerator[SearchResult, None]:
        """
        异步流式搜索

        Args:
            query: 搜索查询词
            max_results: 最大返回结果数
            categories: 搜索类别列表
            engines: 指定使用的搜索引擎列表
            language: 搜索语言
            safe_search: 安全搜索级别
            time_range: 时间范围
            page: 页码
            **kwargs: 额外参数

        Yields:
            SearchResult: 逐个返回搜索结果

        Example:
            >>> engine = SearXNGEngine()
            >>> async for result in engine.async_search_stream("Python"):
            ...     print(result.title)
        """
        try:
            await self._async_check_rate_limit()

            params = self._build_search_params(
                query=query,
                categories=categories,
                engines=engines,
                language=language,
                safe_search=safe_search,
                time_range=time_range,
                page=page,
                **kwargs,
            )

            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config.url}/search",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                    ssl=None if self.config.verify_ssl else False,
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

            results = self._parse_results(data.get("results", []))

            for searxng_result in results[:max_results]:
                yield searxng_result.to_search_result()

        except Exception as e:
            logger.error(f"Async stream search failed: {e}")

    def search_news(
        self,
        query: str,
        max_results: int = 10,
        time_range: Optional[str] = None,
        **kwargs,
    ) -> SearchResponse:
        """
        执行新闻搜索（快捷方法）

        Args:
            query: 搜索查询词
            max_results: 最大返回结果数
            time_range: 时间范围
            **kwargs: 额外参数

        Returns:
            SearchResponse: 搜索响应对象
        """
        return self.search(
            query,
            max_results=max_results,
            categories=["news"],
            time_range=time_range,
            **kwargs,
        )

    def search_images(
        self,
        query: str,
        max_results: int = 10,
        **kwargs,
    ) -> SearchResponse:
        """
        执行图片搜索（快捷方法）

        Args:
            query: 搜索查询词
            max_results: 最大返回结果数
            **kwargs: 额外参数

        Returns:
            SearchResponse: 搜索响应对象
        """
        return self.search(
            query,
            max_results=max_results,
            categories=["images"],
            **kwargs,
        )

    def search_videos(
        self,
        query: str,
        max_results: int = 10,
        **kwargs,
    ) -> SearchResponse:
        """
        执行视频搜索（快捷方法）

        Args:
            query: 搜索查询词
            max_results: 最大返回结果数
            **kwargs: 额外参数

        Returns:
            SearchResponse: 搜索响应对象
        """
        return self.search(
            query,
            max_results=max_results,
            categories=["videos"],
            **kwargs,
        )

    def get_instance_info(self) -> Dict[str, Any]:
        """
        获取 SearXNG 实例信息

        Returns:
            实例信息字典
        """
        try:
            self._check_rate_limit()
            session = self._get_session()
            response = session.get(
                f"{self.config.url}/",
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
            )
            response.raise_for_status()

            return {
                "status": "online",
                "url": self.config.url,
                "status_code": response.status_code,
            }

        except Exception as e:
            logger.error(f"Failed to get instance info: {e}")
            return {
                "status": "offline",
                "url": self.config.url,
                "error": str(e),
            }

    def get_available_engines(self) -> List[str]:
        """
        获取可用的搜索引擎列表

        Returns:
            可用引擎名称列表
        """
        try:
            self._check_rate_limit()
            session = self._get_session()
            response = session.get(
                f"{self.config.url}/search",
                params={"q": "", "format": "json"},
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
            )
            response.raise_for_status()
            data = response.json()

            return list(data.get("engines", {}).keys())

        except Exception as e:
            logger.error(f"Failed to get available engines: {e}")
            return []

    def clear_cache(self) -> None:
        """
        清空搜索结果缓存
        """
        self._clear_cache()
        logger.info("SearXNG search cache cleared")

    def close(self) -> None:
        """
        关闭 HTTP 会话
        """
        if self._session:
            self._session.close()
            self._session = None

        if self._aiohttp_session:
            asyncio.run(self._aiohttp_session.close())
            self._aiohttp_session = None

    def __enter__(self) -> "SearXNGEngine":
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口"""
        self.close()

    def __del__(self) -> None:
        """析构函数"""
        self.close()


SearchEngineFactory.register_engine_class("searxng", SearXNGEngine)
