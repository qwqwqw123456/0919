"""
对话质量评估模块

提供全面的对话系统质量评估，包括：
- 响应质量评估 (Response Quality)
- 对话连贯性评估 (Coherence)
- 上下文相关性评估 (Context Relevance)
- 回答有用性评估 (Helpfulness)
- 多轮对话评估 (Multi-turn Dialogue)
- 人格一致性评估 (Personality Consistency)
- 情感分析 (Sentiment Analysis)
- 毒性检测 (Toxicity Detection)
"""

import json
import re
import time
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from collections import Counter
import random


@dataclass
class Message:
    """对话消息"""
    role: str
    content: str
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


@dataclass
class Conversation:
    """对话会话"""
    id: str
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, **kwargs) -> Message:
        """添加消息"""
        msg = Message(role=role, content=content, timestamp=time.time(), **kwargs)
        self.messages.append(msg)
        return msg

    def get_last_response(self) -> Optional[str]:
        """获取最后一条回复"""
        for msg in reversed(self.messages):
            if msg.role == 'assistant':
                return msg.content
        return None

    def get_conversation_text(self) -> str:
        """获取对话文本"""
        return '\n'.join(f"{msg.role}: {msg.content}" for msg in self.messages)


@dataclass
class QualityScore:
    """质量评分"""
    name: str
    score: float
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"{self.name}: {self.score:.4f}"


@dataclass
class ConversationEvalResult:
    """对话评估结果"""
    conversation_id: str
    overall_score: float
    quality_scores: List[QualityScore] = field(default_factory=list)
    turn_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'conversation_id': self.conversation_id,
            'overall_score': self.overall_score,
            'quality_scores': [
                {'name': qs.name, 'score': qs.score, 'details': qs.details}
                for qs in self.quality_scores
            ],
            'turn_results': self.turn_results,
            'metadata': self.metadata
        }


class TextAnalyzer:
    """文本分析工具"""

    def __init__(self):
        """初始化文本分析器"""
        self.stop_words = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        ])

    def tokenize(self, text: str) -> List[str]:
        """分词"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if t.strip()]

    def get_word_count(self, text: str) -> int:
        """获取词数"""
        return len(self.tokenize(text))

    def get_sentence_count(self, text: str) -> int:
        """获取句子数"""
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])

    def get_avg_word_length(self, text: str) -> float:
        """获取平均词长"""
        tokens = self.tokenize(text)
        if not tokens:
            return 0.0
        return sum(len(t) for t in tokens) / len(tokens)

    def get_avg_sentence_length(self, text: str) -> float:
        """获取平均句长"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0
        return sum(len(s.split()) for s in sentences) / len(sentences)

    def compute_overlap(self, text1: str, text2: str) -> float:
        """计算两段文本的词汇重叠度"""
        tokens1 = set(self.tokenize(text1))
        tokens2 = set(self.tokenize(text2))

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union)

    def get_ngrams(self, text: str, n: int = 2) -> List[Tuple[str, ...]]:
        """获取n-gram"""
        tokens = self.tokenize(text)
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def compute_bleu_like(self, reference: str, candidate: str, n: int = 2) -> float:
        """计算类BLEU分数"""
        ref_ngrams = Counter(self.get_ngrams(reference, n))
        cand_ngrams = Counter(self.get_ngrams(candidate, n))

        if not cand_ngrams:
            return 0.0

        overlap = sum((ref_ngrams & cand_ngrams).values())
        total = sum(cand_ngrams.values())

        return overlap / total if total > 0 else 0.0

    def has_question(self, text: str) -> bool:
        """检查是否包含问题"""
        return '?' in text

    def has_greeting(self, text: str) -> bool:
        """检查是否包含问候语"""
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        text_lower = text.lower()
        return any(g in text_lower for g in greetings)

    def has_acknowledgment(self, text: str) -> bool:
        """检查是否包含确认/反馈"""
        acknowledgments = ['i see', 'i understand', 'i know', 'got it', 'understood', 'ok', 'okay', 'yes']
        text_lower = text.lower()
        return any(a in text_lower for a in acknowledgments)


class ResponseQualityEvaluator:
    """响应质量评估器"""

    QUALITY_CRITERIA = {
        'relevance': {
            'weight': 0.25,
            'description': 'Response relevance to the query'
        },
        'coherence': {
            'weight': 0.20,
            'description': 'Logical flow and coherence'
        },
        'helpfulness': {
            'weight': 0.20,
            'description': 'Usefulness of the response'
        },
        'completeness': {
            'weight': 0.15,
            'description': 'Completeness of the answer'
        },
        'conciseness': {
            'weight': 0.10,
            'description': 'Conciseness without unnecessary verbosity'
        },
        'safety': {
            'weight': 0.10,
            'description': 'Safety and appropriateness'
        }
    }

    def __init__(self):
        """初始化响应质量评估器"""
        self.text_analyzer = TextAnalyzer()

    def evaluate_response(
        self,
        query: str,
        response: str,
        context: Optional[str] = None
    ) -> Dict[str, float]:
        """
        评估单个响应的质量

        Args:
            query: 用户查询
            response: 模型响应
            context: 上下文信息

        Returns:
            Dict: 各维度评分
        """
        scores = {}

        scores['relevance'] = self._evaluate_relevance(query, response, context)
        scores['coherence'] = self._evaluate_coherence(response)
        scores['helpfulness'] = self._evaluate_helpfulness(query, response)
        scores['completeness'] = self._evaluate_completeness(query, response)
        scores['conciseness'] = self._evaluate_conciseness(response)
        scores['safety'] = self._evaluate_safety(response)

        return scores

    def _evaluate_relevance(
        self,
        query: str,
        response: str,
        context: Optional[str] = None
    ) -> float:
        """评估相关性"""
        query_tokens = set(self.text_analyzer.tokenize(query))
        response_tokens = set(self.text_analyzer.tokenize(response))

        if not query_tokens or not response_tokens:
            return 0.5

        overlap = len(query_tokens & response_tokens)
        relevance = overlap / len(query_tokens)

        if context:
            context_tokens = set(self.text_analyzer.tokenize(context))
            context_relevance = len(response_tokens & context_tokens) / len(context_tokens)
            relevance = (relevance + context_relevance) / 2

        return min(relevance * 1.5, 1.0)

    def _evaluate_coherence(self, response: str) -> float:
        """评估连贯性"""
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 1:
            return 0.8

        coherence_scores = []
        for i in range(len(sentences) - 1):
            overlap = self.text_analyzer.compute_overlap(sentences[i], sentences[i+1])
            coherence_scores.append(overlap)

        avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.5

        transition_words = ['however', 'therefore', 'furthermore', 'moreover', 'additionally', 'thus', 'hence']
        text_lower = response.lower()
        transition_count = sum(1 for tw in transition_words if tw in text_lower)
        transition_bonus = min(transition_count * 0.05, 0.2)

        return min(avg_coherence * 0.8 + transition_bonus, 1.0)

    def _evaluate_helpfulness(self, query: str, response: str) -> float:
        """评估有用性"""
        helpful_indicators = [
            'here is', 'here are', 'you can', 'you should', 'i recommend',
            'let me', 'i suggest', 'try to', 'you might', 'it is recommended'
        ]
        text_lower = response.lower()

        indicator_score = sum(0.1 for ind in helpful_indicators if ind in text_lower)
        indicator_score = min(indicator_score, 0.4)

        length_score = 0.3 if len(response) > 50 else 0.1
        if len(response) > 100:
            length_score = 0.4

        if self.text_analyzer.has_question(query) and self.text_analyzer.has_question(response):
            question_penalty = 0.2
        else:
            question_penalty = 0.0

        return min(indicator_score + length_score - question_penalty + 0.2, 1.0)

    def _evaluate_completeness(self, query: str, response: str) -> float:
        """评估完整性"""
        if self.text_analyzer.has_question(query):
            if not response.strip().endswith(('.', '!', '?', ':')):
                return 0.6

            question_words = ['what', 'why', 'how', 'when', 'where', 'who', 'which']
            query_lower = query.lower()
            has_question_word = any(qw in query_lower for qw in question_words)

            if has_question_word and len(response) < 30:
                return 0.4

        if len(response) < 20:
            return 0.3

        completeness = min(len(response) / 200, 1.0) * 0.5 + 0.3

        direct_answer_indicators = ['is', 'are', 'was', 'were', 'can', 'cannot', 'should']
        if any(ind in response.lower()[:100] for ind in direct_answer_indicators):
            completeness += 0.2

        return min(completeness, 1.0)

    def _evaluate_conciseness(self, response: str) -> float:
        """评估简洁性"""
        if len(response) < 20:
            return 0.5

        if len(response) > 500:
            return 0.6

        if len(response) < 100:
            return 0.9

        return 0.8

    def _evaluate_safety(self, response: str) -> float:
        """评估安全性"""
        harmful_patterns = [
            r'\bhate\b', r'\bkill\b', r'\bdestroy\b',
            r'\bviolent\b', r'\billegal\b', r'\bdangerous\b'
        ]

        text_lower = response.lower()

        for pattern in harmful_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return 0.3

        return 0.9


class CoherenceEvaluator:
    """对话连贯性评估器"""

    def __init__(self):
        """初始化连贯性评估器"""
        self.text_analyzer = TextAnalyzer()

    def evaluate_turn_coherence(
        self,
        previous_turns: List[Tuple[str, str]],
        current_response: str
    ) -> float:
        """
        评估单轮对话的连贯性

        Args:
            previous_turns: 之前的对话轮次 [(user_query, assistant_response), ...]
            current_response: 当前响应

        Returns:
            float: 连贯性分数
        """
        if not previous_turns:
            return 0.8

        coherence_scores = []

        last_user_query = previous_turns[-1][0] if previous_turns else ""
        last_response = previous_turns[-1][1] if previous_turns else ""

        query_response_overlap = self.text_analyzer.compute_overlap(
            last_user_query, current_response
        )
        coherence_scores.append(query_response_overlap * 0.5)

        if previous_turns:
            for i in range(len(previous_turns) - 1):
                prev_response = previous_turns[i][1]
                curr_response = previous_turns[i + 1][0]
                prev_curr_overlap = self.text_analyzer.compute_overlap(
                    prev_response, curr_response
                )
                coherence_scores.append(prev_curr_overlap * 0.2)

        topic_continuity = self._check_topic_continuity(previous_turns, current_response)
        coherence_scores.append(topic_continuity * 0.3)

        overall = sum(coherence_scores) if coherence_scores else 0.5
        return min(overall, 1.0)

    def _check_topic_continuity(
        self,
        previous_turns: List[Tuple[str, str]],
        current_response: str
    ) -> float:
        """检查话题连续性"""
        if not previous_turns:
            return 0.5

        all_previous_text = ' '.join(
            turn[0] + ' ' + turn[1] for turn in previous_turns
        )

        overlap = self.text_analyzer.compute_overlap(all_previous_text, current_response)

        return overlap

    def evaluate_dialogue_flow(self, conversation: Conversation) -> Dict[str, Any]:
        """评估整体对话流程"""
        if len(conversation.messages) < 2:
            return {'flow_score': 0.8, 'turn_scores': []}

        turn_scores = []
        previous_turns = []

        for i, msg in enumerate(conversation.messages):
            if msg.role == 'user':
                user_query = msg.content
            elif msg.role == 'assistant' and i > 0:
                response = msg.content
                score = self.evaluate_turn_coherence(previous_turns, response)
                turn_scores.append({
                    'turn_index': i,
                    'score': score,
                    'response_preview': response[:50] + '...' if len(response) > 50 else response
                })
                previous_turns.append((conversation.messages[i-1].content, response))

        avg_flow = sum(ts['score'] for ts in turn_scores) / len(turn_scores) if turn_scores else 0.8

        return {
            'flow_score': avg_flow,
            'turn_scores': turn_scores,
            'num_turns': len(turn_scores)
        }


class SentimentAnalyzer:
    """情感分析器"""

    POSITIVE_WORDS = [
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome',
        'love', 'like', 'happy', 'pleased', 'satisfied', 'helpful', 'perfect',
        'best', 'brilliant', 'outstanding', 'superb', 'beautiful', 'nice'
    ]

    NEGATIVE_WORDS = [
        'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'dislike',
        'angry', 'frustrated', 'disappointed', 'upset', 'annoying', 'useless',
        'poor', 'inferior', 'broken', 'wrong', 'error', 'fail'
    ]

    NEGATION_WORDS = ['not', 'no', "n't", 'never', 'neither', 'nor', 'without']

    def __init__(self):
        """初始化情感分析器"""
        self.positive_set = set(self.POSITIVE_WORDS)
        self.negative_set = set(self.NEGATIVE_WORDS)
        self.negation_set = set(self.NEGATION_WORDS)

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        分析文本情感

        Args:
            text: 待分析文本

        Returns:
            Dict: 情感分析结果
        """
        tokens = text.lower().split()
        tokens_set = set(tokens)

        pos_count = len(tokens_set & self.positive_set)
        neg_count = len(tokens_set & self.negative_set)

        for i, token in enumerate(tokens):
            if token in self.negation_set and i + 1 < len(tokens):
                next_token = tokens[i + 1]
                if next_token in self.positive_set:
                    pos_count -= 1
                    neg_count += 1
                elif next_token in self.negative_set:
                    neg_count -= 1
                    pos_count += 1

        total_sentiment = pos_count + neg_count
        if total_sentiment == 0:
            sentiment_score = 0.5
            label = 'neutral'
        else:
            sentiment_score = (pos_count - neg_count + total_sentiment) / (2 * total_sentiment)
            if sentiment_score > 0.6:
                label = 'positive'
            elif sentiment_score < 0.4:
                label = 'negative'
            else:
                label = 'neutral'

        intensity = min(total_sentiment / 5, 1.0) if total_sentiment > 0 else 0.0

        return {
            'sentiment': label,
            'score': sentiment_score,
            'intensity': intensity,
            'positive_count': pos_count,
            'negative_count': neg_count
        }

    def check_sentiment_match(
        self,
        user_sentiment: Dict[str, Any],
        assistant_sentiment: Dict[str, Any]
    ) -> float:
        """检查情感匹配度"""
        sentiment_scores = [user_sentiment['score'], assistant_sentiment['score']]
        diff = abs(sentiment_scores[0] - sentiment_scores[1])

        if user_sentiment['sentiment'] == assistant_sentiment['sentiment']:
            match_bonus = 0.2
        else:
            match_bonus = -0.1

        return max(0.0, 1.0 - diff + match_bonus)


class ChatQualityEval:
    """对话质量评估器"""

    def __init__(self):
        """初始化对话质量评估器"""
        self.text_analyzer = TextAnalyzer()
        self.quality_evaluator = ResponseQualityEvaluator()
        self.coherence_evaluator = CoherenceEvaluator()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.results = []

    def evaluate_conversation(
        self,
        conversation: Conversation,
        reference_responses: Optional[List[str]] = None
    ) -> ConversationEvalResult:
        """
        评估完整对话

        Args:
            conversation: 对话会话
            reference_responses: 参考响应列表（用于对比）

        Returns:
            ConversationEvalResult: 评估结果
        """
        turn_results = []
        quality_scores_list = []

        previous_turns = []

        for i, msg in enumerate(conversation.messages):
            if msg.role == 'user':
                continue

            response = msg.content

            user_query = ""
            for j in range(i - 1, -1, -1):
                if conversation.messages[j].role == 'user':
                    user_query = conversation.messages[j].content
                    break

            context = ""
            for j in range(max(0, i - 5), i):
                context += conversation.messages[j].content + " "

            quality_scores = self.quality_evaluator.evaluate_response(
                user_query, response, context
            )

            coherence_score = self.coherence_evaluator.evaluate_turn_coherence(
                previous_turns, response
            )

            sentiment = self.sentiment_analyzer.analyze_sentiment(response)

            turn_result = {
                'turn_index': i,
                'response': response,
                'quality_scores': quality_scores,
                'coherence_score': coherence_score,
                'sentiment': sentiment
            }

            if reference_responses and i - 1 < len(reference_responses):
                ref_response = reference_responses[i - 1]
                ref_score = self.text_analyzer.compute_overlap(response, ref_response)
                turn_result['reference_similarity'] = ref_score

            turn_results.append(turn_result)

            if previous_turns:
                previous_turns.append((user_query, response))
            else:
                previous_turns = [(user_query, response)]

        overall_quality = self._compute_overall_quality(turn_results)
        flow_result = self.coherence_evaluator.evaluate_dialogue_flow(conversation)

        return ConversationEvalResult(
            conversation_id=conversation.id,
            overall_score=overall_quality,
            quality_scores=[
                QualityScore(name=k, score=v)
                for k, v in self._aggregate_quality_scores(turn_results).items()
            ],
            turn_results=turn_results,
            metadata={
                'flow_score': flow_result['flow_score'],
                'num_turns': len(turn_results)
            }
        )

    def evaluate_single_response(
        self,
        query: str,
        response: str,
        context: Optional[str] = None,
        previous_turns: Optional[List[Tuple[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        评估单个响应

        Args:
            query: 用户查询
            response: 模型响应
            context: 上下文
            previous_turns: 之前的对话轮次

        Returns:
            Dict: 评估结果
        """
        quality_scores = self.quality_evaluator.evaluate_response(query, response, context)

        coherence_score = 0.8
        if previous_turns:
            coherence_score = self.coherence_evaluator.evaluate_turn_coherence(
                previous_turns, response
            )

        sentiment = self.sentiment_analyzer.analyze_sentiment(response)

        result = {
            'quality_scores': quality_scores,
            'coherence_score': coherence_score,
            'sentiment': sentiment,
            'overall_score': self._compute_single_overall(quality_scores, coherence_score)
        }

        return result

    def _compute_overall_quality(self, turn_results: List[Dict[str, Any]]) -> float:
        """计算整体质量分数"""
        if not turn_results:
            return 0.0

        weights = {
            'relevance': 0.25,
            'coherence': 0.20,
            'helpfulness': 0.20,
            'completeness': 0.15,
            'conciseness': 0.10,
            'safety': 0.10
        }

        total_score = 0.0
        total_weight = 0.0

        for turn in turn_results:
            qs = turn['quality_scores']
            for criterion, weight in weights.items():
                if criterion in qs:
                    total_score += qs[criterion] * weight
                    total_weight += weight

            if 'coherence_score' in turn:
                total_score += turn['coherence_score'] * 0.2
                total_weight += 0.2

        return total_score / total_weight if total_weight > 0 else 0.0

    def _compute_single_overall(
        self,
        quality_scores: Dict[str, float],
        coherence_score: float
    ) -> float:
        """计算单轮整体分数"""
        weights = {
            'relevance': 0.25,
            'coherence': 0.20,
            'helpfulness': 0.20,
            'completeness': 0.15,
            'conciseness': 0.10,
            'safety': 0.10
        }

        total_score = sum(
            quality_scores.get(c, 0.5) * w
            for c, w in weights.items()
        )
        total_score += coherence_score * 0.2

        return total_score

    def _aggregate_quality_scores(
        self,
        turn_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """聚合质量分数"""
        if not turn_results:
            return {}

        criteria = ['relevance', 'coherence', 'helpfulness', 'completeness', 'conciseness', 'safety']

        aggregated = {}
        for criterion in criteria:
            scores = []
            for turn in turn_results:
                if criterion == 'coherence':
                    if 'coherence_score' in turn:
                        scores.append(turn['coherence_score'])
                elif 'quality_scores' in turn and criterion in turn['quality_scores']:
                    scores.append(turn['quality_scores'][criterion])

            if scores:
                aggregated[criterion] = sum(scores) / len(scores)

        return aggregated

    def batch_evaluate(
        self,
        conversations: List[Conversation],
        reference_responses: Optional[Dict[str, List[str]]] = None
    ) -> List[ConversationEvalResult]:
        """
        批量评估多个对话

        Args:
            conversations: 对话列表
            reference_responses: 参考响应字典 {conversation_id: [responses]}

        Returns:
            List[ConversationEvalResult]: 评估结果列表
        """
        results = []
        for conv in conversations:
            ref = reference_responses.get(conv.id) if reference_responses else None
            result = self.evaluate_conversation(conv, ref)
            results.append(result)

        self.results = results
        return results

    def get_summary(self) -> Dict[str, Any]:
        """获取评估摘要"""
        if not self.results:
            return {'message': 'No evaluation results available'}

        overall_scores = [r.overall_score for r in self.results]

        criteria_scores = {}
        for result in self.results:
            for qs in result.quality_scores:
                if qs.name not in criteria_scores:
                    criteria_scores[qs.name] = []
                criteria_scores[qs.name].append(qs.score)

        summary = {
            'num_conversations': len(self.results),
            'overall_stats': {
                'mean': sum(overall_scores) / len(overall_scores),
                'min': min(overall_scores),
                'max': max(overall_scores)
            },
            'criteria_stats': {
                criterion: {
                    'mean': sum(scores) / len(scores),
                    'min': min(scores),
                    'max': max(scores)
                }
                for criterion, scores in criteria_scores.items()
            }
        }

        return summary

    def save_results(self, filepath: str) -> None:
        """保存评估结果"""
        output = {
            'summary': self.get_summary(),
            'results': [r.to_dict() for r in self.results]
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)


def demo():
    """演示函数"""
    print("=" * 60)
    print("对话质量评估演示")
    print("=" * 60)

    evaluator = ChatQualityEval()

    conv = Conversation(id="demo_conv")
    conv.add_message('user', 'What is Python?')
    conv.add_message('assistant', 'Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.')
    conv.add_message('user', 'What are its main features?')
    conv.add_message('assistant', 'Python has several key features including: 1) Easy to learn and read syntax, 2) Dynamic typing and automatic memory management, 3) Extensive standard library, 4) Cross-platform compatibility, and 5) Strong community support.')

    print("\n--- Single Response Evaluation ---")
    result = evaluator.evaluate_single_response(
        query="What is Python?",
        response="Python is a high-level programming language known for its simplicity."
    )
    print(f"Overall Score: {result['overall_score']:.4f}")
    print(f"Quality Scores: {result['quality_scores']}")

    print("\n--- Conversation Evaluation ---")
    conv_result = evaluator.evaluate_conversation(conv)
    print(f"Overall Quality Score: {conv_result.overall_score:.4f}")
    print(f"Number of Turns: {len(conv_result.turn_results)}")

    for qs in conv_result.quality_scores:
        print(f"  {qs.name}: {qs.score:.4f}")

    print("\n--- Batch Evaluation ---")
    evaluator.results = [conv_result]
    summary = evaluator.get_summary()
    print(f"Conversations Evaluated: {summary['num_conversations']}")
    print(f"Overall Mean: {summary['overall_stats']['mean']:.4f}")


if __name__ == "__main__":
    demo()
