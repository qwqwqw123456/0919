"""
DuckDuckGo 搜索引擎 API 集成模块

提供对 DuckDuckGo 搜索功能的完整封装，支持文本新闻、图片、视频等多种搜索类型。
所有搜索操作都支持异步调用和错误重试机制。

依赖安装:
    pip install duckduckgo-search

使用示例:
    # 同步使用
    api = DuckDuckGoAPI()
    results = api.search("Python 教程")

    # 异步使用
    api = DuckDuckGoAPI()
    results = await api.async_search("Python 教程")

    # 使用工厂创建
    factory = SearchEngineFactory()
    engine = factory.create_engine("duckduckgo")
    results = engine.search("Python 教程")
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import (
    DuckDuckGoSearchException,
    RatelimitException,
    TimeoutException,
)

from .search_engine_base import (
    BaseSearchEngine,
    SearchEngineFactory,
    SearchEngineRegistry,
    SearchResponse,
    SearchResult,
)

logger = logging.getLogger(__name__)


class SearchType(Enum):
    """DuckDuckGo 支持的搜索类型枚举"""

    TEXT = "text"  # 文本搜索
    NEWS = "news"  # 新闻搜索
    IMAGES = "images"  # 图片搜索
    VIDEOS = "videos"  # 视频搜索
    ANSWERS = "answers"  # 即时答案
    MAPS = "maps"  # 地图搜索
    RECIPES = "recipes"  # 食谱搜索


@dataclass
class DuckDuckGoConfig:
    """DuckDuckGo API 配置类

    Attributes:
        timeout: 请求超时时间（秒）
        max_retries: 最大重试次数
        retry_delay: 重试延迟时间（秒）
        safe_search: 安全搜索级别 (0=关闭, 1=适中, 2=严格)
        region: 搜索区域 (如 "cn-zh" 表示中国)
        source: 数据来源
    """

    timeout: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0
    safe_search: int = 1
    region: str = "wt-wt"
    source: str = "web"


@dataclass
class DuckDuckGoResult:
    """DuckDuckGo 单条搜索结果

    Attributes:
        title: 结果标题
        url: 结果链接
        snippet: 结果摘要/描述
        image_url: 图片URL（仅图片搜索）
        video_url: 视频URL（仅视频搜索）
        source: 数据来源网站
        published_date: 发布时间（仅新闻搜索）
    """

    title: str
    url: str
    snippet: str = ""
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    source: Optional[str] = None
    published_date: Optional[str] = None

    def to_search_result(self) -> SearchResult:
        """转换为通用 SearchResult 格式"""
        return SearchResult(
            engine="duckduckgo",
            title=self.title,
            url=self.url,
            snippet=self.snippet,
            source=self.source,
        )


@SearchEngineRegistry.register("duckduckgo")
class DuckDuckGoAPI(BaseSearchEngine):
    """
    DuckDuckGo 搜索引擎 API 实现类

    继承自 BaseSearchEngine，提供完整的 DuckDuckGo 搜索功能封装。
    支持文本、新闻、图片、视频等多种搜索类型，所有方法都支持同步和异步调用。

    Features:
        - 多种搜索类型支持（文本、新闻、图片、视频等）
        - 异步搜索支持
        - 自动重试机制
        - 速率限制保护
        - 结果缓存
        - 完整的错误处理

    Example:
        >>> api = DuckDuckGoAPI()
        >>> # 文本搜索
        >>> results = api.search("人工智能发展")
        >>> # 新闻搜索
        >>> news = api.search("科技新闻", search_type=SearchType.NEWS)
        >>> # 异步搜索
        >>> results = await api.async_search("机器学习")
    """

    def __init__(
        self,
        config: Optional[DuckDuckGoConfig] = None,
        enable_cache: bool = True,
        cache_size: int = 100,
        enable_rate_limit: bool = True,
    ):
        """
        初始化 DuckDuckGo API 客户端

        Args:
            config: DuckDuckGo 配置对象，若为 None 则使用默认配置
            enable_cache: 是否启用结果缓存
            cache_size: 缓存最大容量
            enable_rate_limit: 是否启用速率限制保护
        """
        super().__init__(
            engine_name="duckduckgo",
            enable_cache=enable_cache,
            cache_size=cache_size,
            enable_rate_limit=enable_rate_limit,
        )

        self.config = config or DuckDuckGoConfig()
        self._last_request_time = 0.0
        self._min_request_interval = 0.5  # 最小请求间隔（秒）

    def _check_rate_limit(self) -> None:
        """
        检查并实施速率限制

        确保连续请求之间有足够的间隔时间，防止被限流
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

        异步版本的速率限制检查
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
        search_type: SearchType = SearchType.TEXT,
    ) -> List[DuckDuckGoResult]:
        """
        解析 DuckDuckGo 原始搜索结果

        Args:
            raw_results: DuckDuckGo API 返回的原始结果列表
            search_type: 搜索类型

        Returns:
            解析后的 DuckDuckGoResult 列表
        """
        results = []

        for item in raw_results:
            try:
                if search_type == SearchType.TEXT:
                    result = DuckDuckGoResult(
                        title=item.get("title", ""),
                        url=item.get("href", item.get("url", "")),
                        snippet=item.get("body", ""),
                        source=item.get("source", ""),
                    )
                elif search_type == SearchType.NEWS:
                    result = DuckDuckGoResult(
                        title=item.get("title", ""),
                        url=item.get("url", item.get("href", "")),
                        snippet=item.get("description", item.get("body", "")),
                        source=item.get("source", ""),
                        published_date=item.get("published_date", ""),
                    )
                elif search_type == SearchType.IMAGES:
                    result = DuckDuckGoResult(
                        title=item.get("title", ""),
                        url=item.get("image", item.get("url", item.get("href", ""))),
                        snippet=item.get("title", ""),
                        image_url=item.get("image", ""),
                        source=item.get("source", ""),
                    )
                elif search_type == SearchType.VIDEOS:
                    result = DuckDuckGoResult(
                        title=item.get("title", ""),
                        url=item.get("url", item.get("href", "")),
                        snippet=item.get("description", ""),
                        video_url=item.get("content", item.get("url", "")),
                        source=item.get("source", ""),
                        published_date=item.get("published_date", ""),
                    )
                else:
                    result = DuckDuckGoResult(
                        title=item.get("title", ""),
                        url=item.get("url", item.get("href", "")),
                        snippet=item.get("body", item.get("text", "")),
                        source=item.get("source", ""),
                    )

                results.append(result)

            except Exception as e:
                logger.warning(f"Failed to parse search result item: {e}")
                continue

        return results

    def _execute_search_with_retry(
        self,
        query: str,
        max_results: int,
        search_type: SearchType,
        **kwargs,
    ) -> List[DuckDuckGoResult]:
        """
        带重试机制的搜索执行

        Args:
            query: 搜索查询词
            max_results: 最大结果数
            search_type: 搜索类型
            **kwargs: 传递给底层搜索API的额外参数

        Returns:
            搜索结果列表

        Raises:
            DuckDuckGoSearchException: 当所有重试都失败时
        """
        last_exception = None

        for attempt in range(self.config.max_retries):
            try:
                self._check_rate_limit()

                with DDGS() as ddgs:
                    if search_type == SearchType.TEXT:
                        raw_results = list(
                            ddgs.text(
                                query,
                                max_results=max_results,
                                safe_search=self.config.safe_search,
                                region=self.config.region,
                                timeout=self.config.timeout,
                                **kwargs,
                            )
                        )
                    elif search_type == SearchType.NEWS:
                        raw_results = list(
                            ddgs.news(
                                query,
                                max_results=max_results,
                                region=self.config.region,
                                timeout=self.config.timeout,
                                **kwargs,
                            )
                        )
                    elif search_type == SearchType.IMAGES:
                        raw_results = list(
                            ddgs.images(
                                query,
                                max_results=max_results,
                                safe_search=self.config.safe_search,
                                region=self.config.region,
                                timeout=self.config.timeout,
                                **kwargs,
                            )
                        )
                    elif search_type == SearchType.VIDEOS:
                        raw_results = list(
                            ddgs.videos(
                                query,
                                max_results=max_results,
                                safe_search=self.config.safe_search,
                                region=self.config.region,
                                timeout=self.config.timeout,
                                **kwargs,
                            )
                        )
                    elif search_type == SearchType.ANSWERS:
                        raw_results = list(
                            ddgs.answers(
                                query,
                                timeout=self.config.timeout,
                                **kwargs,
                            )
                        )
                    elif search_type == SearchType.MAPS:
                        raw_results = list(
                            ddgs.maps(
                                query,
                                max_results=max_results,
                                region=self.config.region,
                                timeout=self.config.timeout,
                                **kwargs,
                            )
                        )
                    elif search_type == SearchType.RECIPES:
                        raw_results = list(
                            ddgs.recipes(
                                query,
                                max_results=max_results,
                                region=self.config.region,
                                timeout=self.config.timeout,
                                **kwargs,
                            )
                        )
                    else:
                        raw_results = list(
                            ddgs.text(
                                query,
                                max_results=max_results,
                                safe_search=self.config.safe_search,
                                region=self.config.region,
                                timeout=self.config.timeout,
                                **kwargs,
                            )
                        )

                return self._parse_results(raw_results, search_type)

            except RatelimitException as e:
                last_exception = e
                wait_time = self.config.retry_delay * (2**attempt)
                logger.warning(
                    f"Rate limit hit (attempt {attempt + 1}/{self.config.max_retries}), "
                    f"waiting {wait_time:.1f}s"
                )
                time.sleep(wait_time)

            except TimeoutException as e:
                last_exception = e
                logger.warning(
                    f"Timeout (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)

            except DuckDuckGoSearchException as e:
                last_exception = e
                logger.warning(
                    f"Search error (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)

            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error during search: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)

        logger.error(f"All {self.config.max_retries} attempts failed")
        raise DuckDuckGoSearchException(
            f"Search failed after {self.config.max_retries} attempts: {last_exception}"
        )

    def search(
        self,
        query: str,
        max_results: int = 10,
        search_type: Union[SearchType, str] = SearchType.TEXT,
        **kwargs,
    ) -> SearchResponse:
        """
        执行同步搜索

        Args:
            query: 搜索查询词
            max_results: 最大返回结果数，默认为 10
            search_type: 搜索类型，可以是 SearchType 枚举或字符串
            **kwargs: 传递给底层搜索API的额外参数

        Returns:
            SearchResponse: 搜索响应对象，包含结果列表和元数据

        Raises:
            DuckDuckGoSearchException: 搜索失败时抛出
            ValueError: 无效的搜索类型时抛出

        Example:
            >>> api = DuckDuckGoAPI()
            >>> response = api.search("Python 教程")
            >>> for result in response.results:
            ...     print(f"{result.title}: {result.url}")
        """
        if isinstance(search_type, str):
            try:
                search_type = SearchType(search_type.lower())
            except ValueError:
                raise ValueError(
                    f"Invalid search type: {search_type}. "
                    f"Valid types: {[t.value for t in SearchType]}"
                )

        cache_key = self._get_cache_key(query, max_results, search_type.value)
        cached_response = self._get_from_cache(cache_key)
        if cached_response:
            logger.debug(f"Cache hit for query: {query}")
            return cached_response

        start_time = time.time()

        try:
            results = self._execute_search_with_retry(
                query=query,
                max_results=max_results,
                search_type=search_type,
                **kwargs,
            )

            search_results = [r.to_search_result() for r in results]
            response = SearchResponse(
                query=query,
                results=search_results,
                total_results=len(search_results),
                engine=self.engine_name,
                metadata={
                    "search_type": search_type.value,
                    "max_results": max_results,
                },
            )

            self._add_to_cache(cache_key, response)
            response.elapsed_time = time.time() - start_time

            return response

        except DuckDuckGoSearchException:
            raise
        except Exception as e:
            logger.error(f"Search failed with unexpected error: {e}")
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
        search_type: Union[SearchType, str] = SearchType.TEXT,
        **kwargs,
    ) -> SearchResponse:
        """
        执行异步搜索

        使用 asyncio 在线程池中执行搜索操作，避免阻塞事件循环

        Args:
            query: 搜索查询词
            max_results: 最大返回结果数，默认为 10
            search_type: 搜索类型，可以是 SearchType 枚举或字符串
            **kwargs: 传递给底层搜索API的额外参数

        Returns:
            SearchResponse: 搜索响应对象

        Example:
            >>> api = DuckDuckGoAPI()
            >>> response = await api.async_search("Python 教程")
            >>> print(f"Found {response.total_results} results")
        """
        if isinstance(search_type, str):
            try:
                search_type = SearchType(search_type.lower())
            except ValueError:
                raise ValueError(
                    f"Invalid search type: {search_type}. "
                    f"Valid types: {[t.value for t in SearchType]}"
                )

        cache_key = self._get_cache_key(query, max_results, search_type.value)
        cached_response = self._get_from_cache(cache_key)
        if cached_response:
            logger.debug(f"Async cache hit for query: {query}")
            return cached_response

        start_time = time.time()

        try:
            await self._async_check_rate_limit()

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self._execute_search_with_retry,
                query,
                max_results,
                search_type,
            )

            search_results = [r.to_search_result() for r in results]
            response = SearchResponse(
                query=query,
                results=search_results,
                total_results=len(search_results),
                engine=self.engine_name,
                metadata={
                    "search_type": search_type.value,
                    "max_results": max_results,
                },
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
        search_type: Union[SearchType, str] = SearchType.TEXT,
        **kwargs,
    ) -> AsyncGenerator[SearchResult, None]:
        """
        异步流式搜索

        逐步返回搜索结果，适用于需要边获取边处理的场景

        Args:
            query: 搜索查询词
            max_results: 最大返回结果数
            search_type: 搜索类型
            **kwargs: 额外参数

        Yields:
            SearchResult: 逐个返回搜索结果

        Example:
            >>> api = DuckDuckGoAPI()
            >>> async for result in api.async_search_stream("Python"):
            ...     print(result.title)
        """
        if isinstance(search_type, str):
            try:
                search_type = SearchType(search_type.lower())
            except ValueError:
                raise ValueError(f"Invalid search type: {search_type}")

        try:
            await self._async_check_rate_limit()

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self._execute_search_with_retry,
                query,
                max_results,
                search_type,
            )

            for duckduckgo_result in results:
                yield duckduckgo_result.to_search_result()

        except Exception as e:
            logger.error(f"Async stream search failed: {e}")

    def search_text(
        self,
        query: str,
        max_results: int = 10,
        **kwargs,
    ) -> SearchResponse:
        """
        执行文本搜索（快捷方法）

        Args:
            query: 搜索查询词
            max_results: 最大返回结果数
            **kwargs: 额外参数

        Returns:
            SearchResponse: 搜索响应对象
        """
        return self.search(query, max_results, SearchType.TEXT, **kwargs)

    def search_news(
        self,
        query: str,
        max_results: int = 10,
        **kwargs,
    ) -> SearchResponse:
        """
        执行新闻搜索（快捷方法）

        Args:
            query: 搜索查询词
            max_results: 最大返回结果数
            **kwargs: 额外参数

        Returns:
            SearchResponse: 搜索响应对象
        """
        return self.search(query, max_results, SearchType.NEWS, **kwargs)

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
        return self.search(query, max_results, SearchType.IMAGES, **kwargs)

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
        return self.search(query, max_results, SearchType.VIDEOS, **kwargs)

    def search_answers(
        self,
        query: str,
        **kwargs,
    ) -> SearchResponse:
        """
        获取即时答案（快捷方法）

        DuckDuckGo 的即时答案功能，提供直接的答案而非链接列表

        Args:
            query: 搜索查询词
            **kwargs: 额外参数

        Returns:
            SearchResponse: 搜索响应对象
        """
        return self.search(query, max_results=1, search_type=SearchType.ANSWERS, **kwargs)

    def get_trending_searches(self, region: str = "cn-zh") -> List[str]:
        """
        获取热门搜索词

        Args:
            region: 地区代码

        Returns:
            热门搜索词列表
        """
        try:
            self._check_rate_limit()
            with DDGS() as ddgs:
                trends = list(ddgs.trends(max_results=20, region=region))
                return [t.get("name", "") for t in trends if t.get("name")]
        except Exception as e:
            logger.error(f"Failed to get trending searches: {e}")
            return []

    def get_suggestions(self, query: str) -> List[str]:
        """
        获取搜索建议

        Args:
            query: 搜索查询词

        Returns:
            搜索建议列表
        """
        try:
            self._check_rate_limit()
            with DDGS() as ddgs:
                suggestions = list(ddgs.suggestions(query))
                return [s.get("phrase", "") for s in suggestions if s.get("phrase")]
        except Exception as e:
            logger.error(f"Failed to get search suggestions: {e}")
            return []

    def clear_cache(self) -> None:
        """
        清空搜索结果缓存

        Example:
            >>> api = DuckDuckGoAPI()
            >>> api.search("test")
            >>> api.clear_cache()  # 清空缓存
        """
        self._clear_cache()
        logger.info("DuckDuckGo search cache cleared")


SearchEngineFactory.register_engine_class("duckduckgo", DuckDuckGoAPI)
