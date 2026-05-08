"""
意图识别模块
负责识别用户输入的意图和目的
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging


class IntentType(Enum):
    """意图类型枚举"""
    CHAT = "chat"                      # 闲聊
    SEARCH = "search"                  # 搜索
    EXECUTE_CODE = "execute_code"       # 执行代码
    QUESTION = "question"              # 问答
    TASK = "task"                      # 任务执行
    TOOL_USE = "tool_use"              # 使用工具
    ANALYSIS = "analysis"              # 分析
    CREATION = "creation"              # 创作
    LEARNING = "learning"              # 学习
    HELP = "help"                      # 请求帮助
    UNKNOWN = "unknown"                # 未知


class IntentConfidence:
    """意图置信度"""
    def __init__(self, intent: IntentType, confidence: float, keywords: List[str] = None):
        self.intent = intent
        self.confidence = confidence
        self.keywords = keywords or []
    
    def to_dict(self) -> Dict:
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "keywords": self.keywords
        }


@dataclass
class IntentResult:
    """意图识别结果"""
    primary_intent: IntentType
    confidence: float
    secondary_intents: List[IntentConfidence] = field(default_factory=list)
    entities: Dict[str, Any] = field(default_factory=dict)
    slots: Dict[str, Any] = field(default_factory=dict)
    raw_intent: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "primary_intent": self.primary_intent.value,
            "confidence": self.confidence,
            "secondary_intents": [i.to_dict() for i in self.secondary_intents],
            "entities": self.entities,
            "slots": self.slots,
            "raw_intent": self.raw_intent
        }


class IntentRecognizer:
    """
    意图识别器
    
    负责从用户输入中识别出意图，包括：
    1. 关键词匹配
    2. 正则表达式匹配
    3. 模式识别
    4. 实体提取
    5. 槽位填充
    """
    
    def __init__(self, use_ml: bool = False):
        """
        初始化意图识别器
        
        Args:
            use_ml: 是否使用机器学习模型（预留接口）
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.use_ml = use_ml
        
        self._init_keyword_patterns()
        self._init_regex_patterns()
        self._init_entity_extractors()
    
    def _init_keyword_patterns(self):
        """初始化关键词模式"""
        self.keyword_patterns = {
            IntentType.SEARCH: [
                "搜索", "查找", "查询", "搜", "寻找", "search", "find", "look up",
                "帮我找", "搜索一下", "查一下", "找一下", "哪里有", "怎么找"
            ],
            IntentType.EXECUTE_CODE: [
                "执行代码", "运行代码", "写代码", "编程", "代码", "execute", "run code",
                "帮我写", "写一个", "开发", "程序", "脚本", "code", "python", "programming"
            ],
            IntentType.QUESTION: [
                "什么是", "是什么", "为什么", "怎么", "如何", "多少", "who", "what", "why", "how",
                "请问", "问一下", "解释", "说明", "告诉我"
            ],
            IntentType.TASK: [
                "帮我", "请帮我", "帮我做", "完成", "执行", "处理", "做一下", "搞定",
                "accomplish", "complete", "do", "handle", "process"
            ],
            IntentType.TOOL_USE: [
                "使用工具", "调用", "启动", "打开", "使用", "use tool", "call", "invoke",
                "工具", "功能", "能力"
            ],
            IntentType.ANALYSIS: [
                "分析", "解析", "比较", "评估", "研究", "analyze", "compare", "evaluate",
                "研究一下", "分析一下", "对比", "评估"
            ],
            IntentType.CREATION: [
                "创作", "生成", "写", "创建", "制作", "generate", "create", "write", "make",
                "写一首", "画", "生成一个", "创作"
            ],
            IntentType.LEARNING: [
                "学习", "了解", "知道", "认识", "learn", "understand", "know",
                "教我", "介绍一下", "了解一下", "解释一下"
            ],
            IntentType.HELP: [
                "帮助", "帮忙", "help", "assist", "support",
                "怎么用", "如何使用", "有什么", "能做什么"
            ],
            IntentType.CHAT: [
                "你好", "嗨", "在吗", "hi", "hello", "hey", "早上好", "晚上好"
            ]
        }
    
    def _init_regex_patterns(self):
        """初始化正则表达式模式"""
        self.regex_patterns = {
            IntentType.SEARCH: [
                r'搜索[^\s]+',
                r'查找[^\s]+',
                r'.*在哪里.*',
                r'.*的位置.*'
            ],
            IntentType.EXECUTE_CODE: [
                r'```[\s\S]*?```',
                r'执行.*代码',
                r'运行.*程序'
            ],
            IntentType.QUESTION: [
                r'.*\?$',
                r'请问.*',
                r'.*是什么.*',
                r'.*为什么.*'
            ],
            IntentType.TOOL_USE: [
                r'使用(\w+)工具',
                r'调用(\w+)',
                r'启动(\w+)功能'
            ]
        }
    
    def _init_entity_extractors(self):
        """初始化实体提取器"""
        self.entity_extractors = {
            "url": r'https?://[^\s]+',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}-?\d{4}-?\d{4}\b',
            "code_block": r'```[\s\S]*?```',
            "filename": r'[a-zA-Z0-9_]+\.[a-zA-Z]+',
            "number": r'\d+(\.\d+)?',
            "date": r'\d{4}[-/]\d{1,2}[-/]\d{1,2}'
        }
    
    def recognize(self, text: str) -> str:
        """
        识别意图（简化版本，返回字符串）
        
        Args:
            text: 用户输入文本
            
        Returns:
            意图类型字符串
        """
        result = self.recognize_full(text)
        return result.primary_intent.value
    
    def recognize_full(self, text: str) -> IntentResult:
        """
        完整意图识别
        
        Args:
            text: 用户输入文本
            
        Returns:
            IntentResult: 完整的意图识别结果
        """
        text_lower = text.lower().strip()
        
        intent_scores = self._calculate_intent_scores(text, text_lower)
        primary_intent, confidence = self._get_primary_intent(intent_scores)
        secondary_intents = self._get_secondary_intents(intent_scores)
        entities = self._extract_entities(text)
        slots = self._extract_slots(text, primary_intent)
        
        return IntentResult(
            primary_intent=primary_intent,
            confidence=confidence,
            secondary_intents=secondary_intents,
            entities=entities,
            slots=slots,
            raw_intent=text
        )
    
    def _calculate_intent_scores(self, text: str, text_lower: str) -> Dict[IntentType, float]:
        """
        计算各意图的得分
        
        Args:
            text: 原始文本
            text_lower: 小写文本
            
        Returns:
            各意图得分字典
        """
        scores = {intent: 0.0 for intent in IntentType}
        
        for intent, keywords in self.keyword_patterns.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    scores[intent] += 1.0
        
        for intent, patterns in self.regex_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    scores[intent] += 2.0
        
        max_score = max(scores.values()) if scores.values() else 1
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}
        
        return scores
    
    def _get_primary_intent(self, scores: Dict[IntentType, float]) -> Tuple[IntentType, float]:
        """
        获取主要意图
        
        Args:
            scores: 意图得分
            
        Returns:
            (意图类型, 置信度)
        """
        if not scores:
            return IntentType.UNKNOWN, 0.0
        
        primary = max(scores.items(), key=lambda x: x[1])
        
        if primary[1] < 0.1:
            return IntentType.CHAT, 0.5
        
        return primary
    
    def _get_secondary_intents(
        self,
        scores: Dict[IntentType, float]
    ) -> List[IntentConfidence]:
        """
        获取次要意图
        
        Args:
            scores: 意图得分
            
        Returns:
            次要意图列表
        """
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        secondary = []
        for intent, score in sorted_scores[1:4]:
            if score > 0.1:
                secondary.append(IntentConfidence(
                    intent=intent,
                    confidence=score
                ))
        
        return secondary
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        提取实体
        
        Args:
            text: 输入文本
            
        Returns:
            实体字典
        """
        entities = {}
        
        for entity_type, pattern in self.entity_extractors.items():
            matches = re.findall(pattern, text)
            if matches:
                entities[entity_type] = matches
        
        return entities
    
    def _extract_slots(self, text: str, intent: IntentType) -> Dict[str, Any]:
        """
        提取槽位
        
        Args:
            text: 输入文本
            intent: 意图类型
            
        Returns:
            槽位字典
        """
        slots = {}
        
        if intent == IntentType.SEARCH:
            search_term = text
            for keyword in self.keyword_patterns[IntentType.SEARCH]:
                search_term = search_term.replace(keyword, "")
            slots["search_term"] = search_term.strip()
        
        elif intent == IntentType.EXECUTE_CODE:
            code_match = re.search(r'```[\s\S]*?```', text)
            if code_match:
                slots["code"] = code_match.group()
            language_match = re.search(r'```(\w+)', text)
            if language_match:
                slots["language"] = language_match.group(1)
        
        elif intent == IntentType.QUESTION:
            slots["question"] = text.strip()
        
        return slots
    
    def add_keyword_pattern(self, intent: IntentType, keywords: List[str]) -> None:
        """
        添加关键词模式
        
        Args:
            intent: 意图类型
            keywords: 关键词列表
        """
        if intent not in self.keyword_patterns:
            self.keyword_patterns[intent] = []
        self.keyword_patterns[intent].extend(keywords)
    
    def add_regex_pattern(self, intent: IntentType, pattern: str) -> None:
        """
        添加正则表达式模式
        
        Args:
            intent: 意图类型
            pattern: 正则表达式
        """
        if intent not in self.regex_patterns:
            self.regex_patterns[intent] = []
        self.regex_patterns[intent].append(pattern)
    
    def add_entity_extractor(self, entity_type: str, pattern: str) -> None:
        """
        添加实体提取器
        
        Args:
            entity_type: 实体类型
            pattern: 正则表达式
        """
        self.entity_extractors[entity_type] = pattern
    
    def batch_recognize(self, texts: List[str]) -> List[IntentResult]:
        """
        批量识别意图
        
        Args:
            texts: 文本列表
            
        Returns:
            意图识别结果列表
        """
        return [self.recognize_full(text) for text in texts]
    
    def get_intent_description(self, intent: IntentType) -> str:
        """
        获取意图描述
        
        Args:
            intent: 意图类型
            
        Returns:
            意图描述
        """
        descriptions = {
            IntentType.CHAT: "闲聊对话",
            IntentType.SEARCH: "搜索查询",
            IntentType.EXECUTE_CODE: "代码执行",
            IntentType.QUESTION: "问答问题",
            IntentType.TASK: "任务执行",
            IntentType.TOOL_USE: "工具使用",
            IntentType.ANALYSIS: "分析任务",
            IntentType.CREATION: "创作生成",
            IntentType.LEARNING: "学习了解",
            IntentType.HELP: "请求帮助",
            IntentType.UNKNOWN: "未知意图"
        }
        return descriptions.get(intent, "未知")


class AdvancedIntentRecognizer(IntentRecognizer):
    """
    高级意图识别器
    
    基于规则的增强版识别，支持更多特征
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_context_patterns()
        self._init_intent_hierarchy()
    
    def _init_context_patterns(self):
        """初始化上下文模式"""
        self.context_patterns = {
            "follow_up": [
                r"然后呢", "接下来", "之后", "继续", "还有呢",
                "what next", "then", "after that", "continue"
            ],
            "clarification": [
                r"你是说", "你的意思是", "等等", "等一下",
                "you mean", "wait", "hold on"
            ],
            "confirmation": [
                r"是的", "对", "没错", "好的", "可以",
                "yes", "correct", "right", "ok", "sure"
            ],
            "negation": [
                r"不是", "不对", "错", "不是这样",
                "no", "not", "wrong", "incorrect"
            ]
        }
    
    def _init_intent_hierarchy(self):
        """初始化意图层级"""
        self.intent_hierarchy = {
            IntentType.TASK: [IntentType.SEARCH, IntentType.EXECUTE_CODE, IntentType.ANALYSIS],
            IntentType.QUESTION: [IntentType.LEARNING, IntentType.HELP],
            IntentType.CHAT: [IntentType.HELP]
        }
    
    def recognize_full(self, text: str, context: Optional[Dict] = None) -> IntentResult:
        """
        完整意图识别（带上下文）
        
        Args:
            text: 用户输入文本
            context: 上下文信息
            
        Returns:
            IntentResult: 完整的意图识别结果
        """
        result = super().recognize_full(text)
        
        if context:
            result = self._incorporate_context(result, context)
        
        return result
    
    def _incorporate_context(self, result: IntentResult, context: Dict) -> IntentResult:
        """
        合并上下文信息
        
        Args:
            result: 原识别结果
            context: 上下文
            
        Returns:
            更新后的结果
        """
        if "last_intent" in context:
            last_intent = context["last_intent"]
            
            if any(re.search(p, result.raw_intent) for p in self.context_patterns["follow_up"]):
                if last_intent in self.intent_hierarchy:
                    result.slots["follow_up_from"] = last_intent.value
            
            if any(re.search(p, result.raw_intent) for p in self.context_patterns["confirmation"]):
                result.slots["is_confirmation"] = True
                result.confidence = min(1.0, result.confidence * 1.2)
            
            if any(re.search(p, result.raw_intent) for p in self.context_patterns["negation"]):
                result.slots["is_negation"] = True
        
        return result
    
    def infer_intent_from_history(self, history: List[Dict]) -> Optional[IntentType]:
        """
        从历史推断意图
        
        Args:
            history: 对话历史
            
        Returns:
            推断的意图
        """
        if not history:
            return None
        
        last_message = history[-1]
        if isinstance(last_message, dict) and "intent" in last_message:
            return IntentType(last_message["intent"])
        
        return None


__all__ = [
    "IntentRecognizer",
    "AdvancedIntentRecognizer",
    "IntentType",
    "IntentConfidence",
    "IntentResult"
]
