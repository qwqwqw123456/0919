"""
搜索判断模块

根据用户查询内容判断是否需要进行网络搜索，以及选择合适的搜索策略。
支持多种判断规则：时效性判断、意图分析、领域识别等。

使用示例:
    judge = SearchJudge()
    # 判断是否需要搜索
    if judge.should_search("今天北京的天气"):
        print("需要搜索")

    # 获取搜索策略
    strategy = judge.get_search_strategy("Python 教程")
    print(f"搜索策略: {strategy}")

    # 完整判断结果
    result = judge.judge_query("如何学习机器学习")
    print(f"需要搜索: {result.need_search}")
    print(f"搜索类型: {result.search_type}")
"""

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class SearchType(Enum):
    """搜索类型枚举"""

    GENERAL = "general"  # 通用搜索
    NEWS = "news"  # 新闻搜索
    ACADEMIC = "academic"  # 学术搜索
    TECHNICAL = "technical"  # 技术搜索
    PRODUCT = "product"  # 产品搜索
    LOCAL = "local"  # 本地搜索
    IMAGE = "image"  # 图片搜索
    VIDEO = "video"  # 视频搜索
    NONE = "none"  # 不需要搜索


class QueryIntent(Enum):
    """查询意图枚举"""

    INFORMATION = "information"  # 信息查询
    NAVIGATION = "navigation"  # 导航意图
    TRANSACTIONAL = "transactional"  # 交易意图
    COMMERCIAL = "commercial"  # 商业调查
    KNOWLEDGE = "knowledge"  # 知识问答
    OPINION = "opinion"  # 观点意见
    UNKNOWN = "unknown"  # 未知


class QueryDomain(Enum):
    """查询领域枚举"""

    TECHNOLOGY = "technology"
    SCIENCE = "science"
    BUSINESS = "business"
    HEALTH = "health"
    EDUCATION = "education"
    ENTERTAINMENT = "entertainment"
    SPORTS = "sports"
    POLITICS = "politics"
    CULTURE = "culture"
    LIFE = "life"
    GENERAL = "general"


@dataclass
class JudgeResult:
    """判断结果数据类

    Attributes:
        need_search: 是否需要搜索
        search_type: 推荐的搜索类型
        confidence: 判断置信度 (0.0 - 1.0)
        intent: 查询意图
        domain: 查询领域
        keywords: 提取的关键词
        reasons: 判断原因列表
        suggestions: 搜索建议
    """

    need_search: bool = False
    search_type: SearchType = SearchType.NONE
    confidence: float = 0.0
    intent: QueryIntent = QueryIntent.UNKNOWN
    domain: QueryDomain = QueryDomain.GENERAL
    keywords: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    suggestions: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "need_search": self.need_search,
            "search_type": self.search_type.value,
            "confidence": self.confidence,
            "intent": self.intent.value,
            "domain": self.domain.value,
            "keywords": self.keywords,
            "reasons": self.reasons,
            "suggestions": self.suggestions,
        }


class SearchJudge:
    """
    搜索判断器

    根据查询内容分析是否需要搜索，返回搜索策略建议。
    支持多种判断维度：时效性、意图、领域、关键词等。

    Features:
        - 时效性关键词检测
        - 查询意图分析
        - 领域分类识别
        - 搜索类型推荐
        - 置信度评估
        - 自定义规则扩展

    Example:
        >>> judge = SearchJudge()
        >>> result = judge.judge_query("2024年科技发展趋势")
        >>> print(result.need_search)  # True
        >>> print(result.search_type)   # SearchType.NEWS
    """

    _RECENCY_KEYWORDS: Set[str] = {
        "最新", "最近", "今天", "昨天", "明天", "本周", "本月", "今年",
        "刚刚", "刚才", "现在", "眼下", "当前", "目前", "实时", "即时",
        "今日", "昨日", "明日", "本周", "本月", "本年", "近来",
        "new", "latest", "recent", "today", "yesterday", "tomorrow",
        "now", "current", "just", "breaking",
    }

    _NEWS_KEYWORDS: Set[str] = {
        "新闻", "报道", "资讯", "事件", "动态", "消息", "快讯",
        "发生了什么", "今日要闻", "热点", "头条", "播报",
        "news", "breaking", "report", "update", "headline",
    }

    _TECHNICAL_KEYWORDS: Set[str] = {
        "怎么", "如何", "教程", "方法", "步骤", "技巧", "代码",
        "编程", "开发", "API", "文档", "安装", "配置", "部署",
        "问题", "错误", "解决", "修复", "调试", "优化",
        "how", "tutorial", "guide", "code", "api", "docs",
        "install", "setup", "configure", "error", "fix", "debug",
    }

    _PRODUCT_KEYWORDS: Set[str] = {
        "推荐", "对比", "比较", "评测", "评价", "哪个好", "值得买",
        "产品", "手机", "电脑", "相机", "耳机", "键盘", "鼠标",
        "review", "best", "top", "compare", "versus", "vs",
    }

    _ACADEMIC_KEYWORDS: Set[str] = {
        "论文", "研究", "学术", "论文", "文献", "SCI", "EI",
        "论文", "发表", "期刊", "会议", "博士", "硕士",
        "paper", "research", "academic", "thesis", "journal",
    }

    _LOCAL_KEYWORDS: Set[str] = {
        "附近", "周边", "地址", "怎么走", "在哪里", "位置",
        "餐厅", "酒店", "医院", "银行", "地铁", "公交",
        "nearby", "near", "around", "location", "address",
    }

    _IMAGE_KEYWORDS: Set[str] = {
        "图片", "照片", "图", "壁纸", "头像", "表情包",
        "image", "photo", "picture", "wallpaper",
    }

    _VIDEO_KEYWORDS: Set[str] = {
        "视频", "教程", "演示", "讲解", "直播", "录像",
        "video", "tutorial", "demo", "stream",
    }

    _TRIVIA_KEYWORDS: Set[str] = {
        "什么是", "谁是", "哪有", "哪有", "多少", "多大",
        "定义", "概念", "解释", "说明",
        "what is", "who is", "where is", "how many", "definition",
    }

    _NO_SEARCH_PATTERNS: List[re.Pattern] = [
        re.compile(r"^(好的|好的|OK|好|行|嗯|是的)$", re.IGNORECASE),
        re.compile(r"^(你好|您好|hi|hello|hey)$", re.IGNORECASE),
        re.compile(r"^(谢谢|感谢|多谢)$", re.IGNORECASE),
        re.compile(r"^(对不起|抱歉|不好意思)$", re.IGNORECASE),
        re.compile(r"^再见$", re.IGNORECASE),
    ]

    def __init__(
        self,
        custom_rules: Optional[Dict[str, List[str]]] = None,
        enable_fuzzy_match: bool = True,
        min_confidence_threshold: float = 0.5,
    ):
        """
        初始化搜索判断器

        Args:
            custom_rules: 自定义规则字典，格式为 {"类别": ["关键词1", "关键词2"]}
            enable_fuzzy_match: 是否启用模糊匹配
            min_confidence_threshold: 最小置信度阈值
        """
        self.enable_fuzzy_match = enable_fuzzy_match
        self.min_confidence_threshold = min_confidence_threshold

        self._keywords_map: Dict[str, Set[str]] = {
            "recency": self._RECENCY_KEYWORDS.copy(),
            "news": self._NEWS_KEYWORDS.copy(),
            "technical": self._TECHNICAL_KEYWORDS.copy(),
            "product": self._PRODUCT_KEYWORDS.copy(),
            "academic": self._ACADEMIC_KEYWORDS.copy(),
            "local": self._LOCAL_KEYWORDS.copy(),
            "image": self._IMAGE_KEYWORDS.copy(),
            "video": self._VIDEO_KEYWORDS.copy(),
            "trivia": self._TRIVIA_KEYWORDS.copy(),
        }

        if custom_rules:
            self.add_custom_rules(custom_rules)

    def add_custom_rules(self, rules: Dict[str, List[str]]) -> None:
        """
        添加自定义规则

        Args:
            rules: 自定义规则字典
        """
        for category, keywords in rules.items():
            if category not in self._keywords_map:
                self._keywords_map[category] = set()
            self._keywords_map[category].update(keywords)

    def _normalize_query(self, query: str) -> str:
        """
        规范化查询文本

        Args:
            query: 原始查询

        Returns:
            规范化后的查询
        """
        query = query.strip()
        query = re.sub(r"\s+", " ", query)
        query = query.lower()
        return query

    def _check_no_search_patterns(self, query: str) -> bool:
        """
        检查是否符合不需要搜索的模式

        Args:
            query: 规范化后的查询

        Returns:
            True 如果匹配不需要搜索的模式
        """
        for pattern in self._NO_SEARCH_PATTERNS:
            if pattern.match(query):
                return True
        return False

    def _extract_keywords(self, query: str) -> List[str]:
        """
        从查询中提取关键词

        Args:
            query: 规范化后的查询

        Returns:
            提取的关键词列表
        """
        words = query.split()
        keywords = []

        for word in words:
            if len(word) >= 2:
                keywords.append(word)

        return keywords

    def _calculate_keyword_score(
        self,
        query: str,
        keyword_set: Set[str],
    ) -> float:
        """
        计算关键词匹配得分

        Args:
            query: 规范化后的查询
            keyword_set: 关键词集合

        Returns:
            匹配得分 (0.0 - 1.0)
        """
        if not keyword_set:
            return 0.0

        query_lower = query.lower()
        words = set(query_lower.split())

        matched = 0
        total = len(keyword_set)

        for keyword in keyword_set:
            if keyword.lower() in query_lower:
                matched += 1
            elif self.enable_fuzzy_match:
                for word in words:
                    if word in keyword.lower() or keyword.lower() in word:
                        matched += 0.5
                        break

        return min(1.0, matched / max(1, len(words)))

    def _analyze_intent(self, query: str, scores: Dict[str, float]) -> QueryIntent:
        """
        分析查询意图

        Args:
            query: 规范化后的查询
            scores: 各类型关键词得分

        Returns:
            识别的意图类型
        """
        if scores.get("technical", 0) > 0.3:
            return QueryIntent.INFORMATION

        if scores.get("product", 0) > 0.3:
            return QueryIntent.COMMERCIAL

        if scores.get("trivia", 0) > 0.4:
            return QueryIntent.KNOWLEDGE

        if any(scores.get(k, 0) > 0.5 for k in ["news", "recency"]):
            return QueryIntent.INFORMATION

        return QueryIntent.UNKNOWN

    def _analyze_domain(self, query: str, scores: Dict[str, float]) -> QueryDomain:
        """
        分析查询所属领域

        Args:
            query: 规范化后的查询
            scores: 各类型关键词得分

        Returns:
            识别的领域类型
        """
        domain_keywords = {
            QueryDomain.TECHNOLOGY: {"技术", "编程", "代码", "软件", "科技", "技术", "computer", "tech"},
            QueryDomain.SCIENCE: {"科学", "研究", "实验", "物理", "化学", "生物", "science", "research"},
            QueryDomain.BUSINESS: {"商业", "投资", "股票", "经济", "市场", "business", "finance"},
            QueryDomain.HEALTH: {"健康", "医疗", "疾病", "医院", "医生", "health", "medical"},
            QueryDomain.EDUCATION: {"教育", "学习", "学校", "课程", "考试", "education", "school"},
            QueryDomain.ENTERTAINMENT: {"娱乐", "电影", "音乐", "游戏", "明星", "entertainment", "movie"},
            QueryDomain.SPORTS: {"体育", "足球", "篮球", "比赛", "运动", "sports", "football"},
            QueryDomain.POLITICS: {"政治", "政府", "政策", "国际", "politics", "government"},
            QueryDomain.CULTURE: {"文化", "历史", "艺术", "文学", "culture", "history", "art"},
            QueryDomain.LIFE: {"生活", "家居", "美食", "旅游", "life", "travel", "food"},
        }

        query_lower = query.lower()

        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return domain

        return QueryDomain.GENERAL

    def _determine_search_type(self, scores: Dict[str, float]) -> SearchType:
        """
        确定搜索类型

        Args:
            scores: 各类型关键词得分

        Returns:
            推荐的搜索类型
        """
        max_score = 0.0
        search_type = SearchType.GENERAL

        type_mapping = {
            "news": SearchType.NEWS,
            "academic": SearchType.ACADEMIC,
            "technical": SearchType.TECHNICAL,
            "product": SearchType.PRODUCT,
            "local": SearchType.LOCAL,
            "image": SearchType.IMAGE,
            "video": SearchType.VIDEO,
        }

        for key, st in type_mapping.items():
            if scores.get(key, 0) > max_score:
                max_score = scores.get(key, 0)
                search_type = st

        if max_score < 0.2:
            return SearchType.GENERAL

        return search_type

    def should_search(self, query: str) -> bool:
        """
        判断是否需要搜索

        简单的快捷方法，用于快速判断

        Args:
            query: 用户查询

        Returns:
            True 如果需要搜索
        """
        result = self.judge_query(query)
        return result.need_search

    def get_search_strategy(self, query: str) -> SearchType:
        """
        获取搜索策略

        快捷方法，返回推荐的搜索类型

        Args:
            query: 用户查询

        Returns:
            推荐的搜索类型
        """
        result = self.judge_query(query)
        return result.search_type

    def judge_query(self, query: str) -> JudgeResult:
        """
        完整判断查询

        综合分析查询内容，返回完整的判断结果

        Args:
            query: 用户查询

        Returns:
            JudgeResult: 包含详细判断信息的结果对象

        Example:
            >>> judge = SearchJudge()
            >>> result = judge.judge_query("今天北京的天气怎么样")
            >>> print(f"需要搜索: {result.need_search}")
            >>> print(f"搜索类型: {result.search_type.value}")
            >>> print(f"置信度: {result.confidence:.2f}")
        """
        result = JudgeResult()
        normalized = self._normalize_query(query)

        if self._check_no_search_patterns(normalized):
            result.need_search = False
            result.search_type = SearchType.NONE
            result.confidence = 1.0
            result.reasons.append("查询符合不需要搜索的模式（如问候语、简单回复等）")
            return result

        keywords = self._extract_keywords(normalized)
        result.keywords = keywords

        scores = {}
        for category, keyword_set in self._keywords_map.items():
            scores[category] = self._calculate_keyword_score(normalized, keyword_set)

        recency_score = scores.get("recency", 0)
        news_score = scores.get("news", 0)
        technical_score = scores.get("technical", 0)

        if recency_score > 0 or news_score > 0:
            result.need_search = True
            result.search_type = SearchType.NEWS
            result.confidence = max(recency_score, news_score) * 0.9
            result.reasons.append(f"检测到时效性关键词（得分: {recency_score:.2f}）")
            if news_score > 0.3:
                result.reasons.append(f"检测到新闻相关关键词（得分: {news_score:.2f}）")

        elif technical_score > 0.3:
            result.need_search = True
            result.search_type = SearchType.TECHNICAL
            result.confidence = technical_score * 0.8
            result.reasons.append(f"检测到技术类关键词（得分: {technical_score:.2f}）")

        else:
            search_type = self._determine_search_type(scores)
            max_score = max(scores.values()) if scores else 0

            if max_score > 0.2:
                result.need_search = True
                result.search_type = search_type
                result.confidence = max_score * 0.7
                result.reasons.append(f"检测到{search_type.value}相关关键词")
            else:
                result.need_search = True
                result.search_type = SearchType.GENERAL
                result.confidence = 0.5
                result.reasons.append("未检测到特定类型关键词，使用通用搜索")

        result.intent = self._analyze_intent(normalized, scores)
        result.domain = self._analyze_domain(normalized, scores)

        if result.intent != QueryIntent.UNKNOWN:
            result.reasons.append(f"识别到查询意图: {result.intent.value}")

        if result.domain != QueryDomain.GENERAL:
            result.reasons.append(f"识别到查询领域: {result.domain.value}")

        result.suggestions = {
            "search_engine": "duckduckgo" if result.search_type == SearchType.NEWS else "searxng",
            "max_results": "10" if result.search_type == SearchType.GENERAL else "20",
            "time_range": "day" if recency_score > 0.5 else "week",
        }

        if result.confidence < self.min_confidence_threshold:
            result.need_search = True
            result.search_type = SearchType.GENERAL
            result.confidence = self.min_confidence_threshold

        logger.debug(
            f"Query judgment: {query} -> need_search={result.need_search}, "
            f"type={result.search_type.value}, confidence={result.confidence:.2f}"
        )

        return result

    def batch_judge(self, queries: List[str]) -> List[JudgeResult]:
        """
        批量判断查询

        Args:
            queries: 查询列表

        Returns:
            判断结果列表
        """
        return [self.judge_query(query) for query in queries]

    def get_statistics(self, results: List[JudgeResult]) -> Dict:
        """
        获取批量判断的统计信息

        Args:
            results: 判断结果列表

        Returns:
            统计信息字典
        """
        stats = {
            "total": len(results),
            "need_search": sum(1 for r in results if r.need_search),
            "no_search": sum(1 for r in results if not r.need_search),
            "by_type": {},
            "by_domain": {},
            "avg_confidence": 0.0,
        }

        for result in results:
            type_key = result.search_type.value
            stats["by_type"][type_key] = stats["by_type"].get(type_key, 0) + 1

            domain_key = result.domain.value
            stats["by_domain"][domain_key] = stats["by_domain"].get(domain_key, 0) + 1

        if results:
            stats["avg_confidence"] = sum(r.confidence for r in results) / len(results)

        return stats
