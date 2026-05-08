"""
思维反思模块
负责对代理的行为和结果进行反思和评估，实现自我改进
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import uuid


class ReflectionLevel(Enum):
    """反思层级枚举"""
    SURFACE = "surface"                      # 表面反思
    CAUSAL = "causal"                        # 因果反思
    COUNTERFACTUAL = "counterfactual"       # 反事实反思
    META = "meta"                           # 元认知反思


class EvaluationDimension(Enum):
    """评估维度枚举"""
    CORRECTNESS = "correctness"             # 正确性
    EFFICIENCY = "efficiency"               # 效率
    COMPLETENESS = "completeness"           # 完整性
    COHERENCE = "coherence"                 # 一致性
    NOVELTY = "novelty"                     # 新颖性


class ImprovementType(Enum):
    """改进类型枚举"""
    STRATEGY_ADJUSTMENT = "strategy_adjustment"  # 策略调整
    TOOL_SELECTION = "tool_selection"        # 工具选择优化
    ERROR_CORRECTION = "error_correction"   # 错误纠正
    KNOWLEDGE_UPDATE = "knowledge_update"    # 知识更新
    PROMPT_REFINEMENT = "prompt_refinement"  # 提示词优化


@dataclass
class ReflectionResult:
    """反思结果数据类"""
    reflection_id: str
    timestamp: float
    level: ReflectionLevel
    evaluation: Dict[EvaluationDimension, float]
    insights: List[str]
    improvements: List[Dict[str, Any]]
    success_score: float
    reasoning_chain: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.reflection_id:
            self.reflection_id = str(uuid.uuid4())
    
    def get_overall_score(self) -> float:
        """获取总体评分"""
        if not self.evaluation:
            return 0.0
        return sum(self.evaluation.values()) / len(self.evaluation)
    
    def get_top_improvements(self, n: int = 3) -> List[Dict[str, Any]]:
        """获取最重要的改进建议"""
        sorted_improvements = sorted(
            self.improvements,
            key=lambda x: x.get("priority", 0),
            reverse=True
        )
        return sorted_improvements[:n]
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "reflection_id": self.reflection_id,
            "timestamp": self.timestamp,
            "level": self.level.value,
            "evaluation": {k.value: v for k, v in self.evaluation.items()},
            "insights": self.insights,
            "improvements": self.improvements,
            "success_score": self.success_score,
            "reasoning_chain": self.reasoning_chain,
            "overall_score": self.get_overall_score(),
            "metadata": self.metadata
        }


@dataclass
class ActionEvaluation:
    """行动评估数据类"""
    action_id: str
    action_type: str
    expected_outcome: Any
    actual_outcome: Any
    success: bool
    deviation: Optional[float] = None
    feedback: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def analyze_deviation(self) -> str:
        """分析偏差"""
        if not self.success:
            if self.deviation and self.deviation > 0.5:
                return "重大偏差：实际结果与预期差异很大"
            elif self.deviation and self.deviation > 0.2:
                return "中等偏差：存在一定差异"
            else:
                return "轻微偏差：基本符合预期"
        return "执行成功，符合预期"


class ThoughtReflection:
    """
    思维反思器
    
    负责对代理的行为和结果进行反思，包括：
    1. 表面层反思：评估行动结果
    2. 因果反思：分析行动与结果的关系
    3. 反事实反思：考虑替代方案
    4. 元认知反思：反思整个思考过程
    
    反思机制帮助代理：
    - 从错误中学习
    - 优化决策策略
    - 提高任务完成质量
    - 实现持续自我改进
    """
    
    def __init__(
        self,
        enable: bool = True,
        default_level: ReflectionLevel = ReflectionLevel.CAUSAL,
        max_history: int = 100
    ):
        """
        初始化思维反思器
        
        Args:
            enable: 是否启用反思功能
            default_level: 默认反思层级
            max_history: 最大历史记录数
        """
        self.enable = enable
        self.default_level = default_level
        self.max_history = max_history
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.reflection_history: List[ReflectionResult] = []
        self.action_evaluations: Dict[str, ActionEvaluation] = {}
        self.insights_cache: Dict[str, List[str]] = defaultdict(list)
        self.improvement_history: List[Dict[str, Any]] = []
        
        self._init_evaluation_thresholds()
        self._init_reflection_prompts()
    
    def _init_evaluation_thresholds(self):
        """初始化评估阈值"""
        self.evaluation_thresholds = {
            EvaluationDimension.CORRECTNESS: 0.7,
            EvaluationDimension.EFFICIENCY: 0.6,
            EvaluationDimension.COMPLETENESS: 0.75,
            EvaluationDimension.COHERENCE: 0.8,
            EvaluationDimension.NOVELTY: 0.3
        }
    
    def _init_reflection_prompts(self):
        """初始化反思提示词"""
        self.reflection_prompts = {
            ReflectionLevel.SURFACE: """评估以下行动结果：
行动：{action}
预期：{expected}
实际：{actual}
评估：这次行动是否成功？""",
            
            ReflectionLevel.CAUSAL: """分析以下行动与结果之间的因果关系：
行动：{action}
结果：{result}

请思考：
1. 什么因素导致了这一结果？
2. 行动中的哪些决策是正确的？
3. 哪些决策可能存在问题？
4. 如何改进以获得更好的结果？""",
            
            ReflectionLevel.COUNTERFACTUAL: """进行反事实推理：
实际行动：{actual_action}
实际结果：{actual_result}

请思考：
1. 如果采取不同的行动会怎样？
2. 是否有更好的替代方案？
3. 从这些替代方案中可以学到什么？""",
            
            ReflectionLevel.META: """进行元认知反思：
任务：{task}
执行过程：{process}
结果：{result}

请思考：
1. 整个思考过程是否有效？
2. 是否使用了正确的策略？
3. 如何改进思考方法以提高未来表现？"""
        }
    
    def reflect(
        self,
        action_result: Any,
        tool_calls: Optional[List[Dict]] = None,
        iterations: int = 0
    ) -> Dict[str, Any]:
        """
        执行反思（简化版本）
        
        Args:
            action_result: 行动结果
            tool_calls: 工具调用列表
            iterations: 迭代次数
            
        Returns:
            反思结果字典
        """
        reflection = self.reflect_full(action_result, tool_calls, iterations)
        return {
            "success_score": reflection.success_score,
            "insights": reflection.insights,
            "improvements": reflection.improvements,
            "evaluation": {k.value: v for k, v in reflection.evaluation.items()}
        }
    
    def reflect_full(
        self,
        action_result: Any,
        tool_calls: Optional[List[Dict]] = None,
        iterations: int = 0,
        level: Optional[ReflectionLevel] = None,
        context: Optional[Dict] = None
    ) -> ReflectionResult:
        """
        完整反思过程
        
        Args:
            action_result: 行动结果
            tool_calls: 工具调用列表
            iterations: 迭代次数
            level: 反思层级
            context: 额外上下文
            
        Returns:
            反思结果对象
        """
        if not self.enable:
            return self._create_empty_reflection()
        
        reflection_level = level or self.default_level
        
        self.logger.info(f"Starting {reflection_level.value} level reflection")
        
        evaluation = self._evaluate_result(action_result, tool_calls, iterations)
        
        insights = self._generate_insights(action_result, evaluation, tool_calls)
        
        improvements = self._suggest_improvements(
            action_result,
            evaluation,
            insights,
            tool_calls
        )
        
        success_score = self._calculate_success_score(evaluation, iterations)
        
        reasoning_chain = self._build_reasoning_chain(
            action_result,
            evaluation,
            improvements
        )
        
        reflection = ReflectionResult(
            reflection_id=str(uuid.uuid4()),
            timestamp=time.time(),
            level=reflection_level,
            evaluation=evaluation,
            insights=insights,
            improvements=improvements,
            success_score=success_score,
            reasoning_chain=reasoning_chain,
            metadata={
                "iterations": iterations,
                "tool_calls_count": len(tool_calls) if tool_calls else 0,
                "context": context or {}
            }
        )
        
        self._store_reflection(reflection)
        
        return reflection
    
    def _evaluate_result(
        self,
        action_result: Any,
        tool_calls: Optional[List[Dict]],
        iterations: int
    ) -> Dict[EvaluationDimension, float]:
        """
        评估结果
        
        Args:
            action_result: 行动结果
            tool_calls: 工具调用列表
            iterations: 迭代次数
            
        Returns:
            各维度评估得分
        """
        evaluation = {}
        
        evaluation[EvaluationDimension.CORRECTNESS] = self._evaluate_correctness(
            action_result, tool_calls
        )
        
        evaluation[EvaluationDimension.EFFICIENCY] = self._evaluate_efficiency(
            iterations, tool_calls
        )
        
        evaluation[EvaluationDimension.COMPLETENESS] = self._evaluate_completeness(
            action_result, tool_calls
        )
        
        evaluation[EvaluationDimension.COHERENCE] = self._evaluate_coherence(
            tool_calls
        )
        
        evaluation[EvaluationDimension.NOVELTY] = self._evaluate_novelty(
            action_result, tool_calls
        )
        
        return evaluation
    
    def _evaluate_correctness(
        self,
        action_result: Any,
        tool_calls: Optional[List[Dict]]
    ) -> float:
        """评估正确性"""
        if action_result is None:
            return 0.0
        
        if isinstance(action_result, dict):
            if action_result.get("error"):
                return 0.0
            if "success" in action_result:
                return 1.0 if action_result["success"] else 0.0
        
        if isinstance(action_result, str):
            if "error" in action_result.lower() or "failed" in action_result.lower():
                return 0.3
        
        if tool_calls:
            failed_calls = sum(
                1 for call in tool_calls
                if call.get("result", {}).get("error")
            )
            return 1.0 - (failed_calls / len(tool_calls))
        
        return 0.7
    
    def _evaluate_efficiency(self, iterations: int, tool_calls: Optional[List[Dict]]) -> float:
        """评估效率"""
        if iterations == 0:
            return 1.0
        
        iteration_score = max(0.0, 1.0 - (iterations / 20))
        
        tool_score = 1.0
        if tool_calls:
            tool_count = len(tool_calls)
            if tool_count > 10:
                tool_score = 0.5
            elif tool_count > 5:
                tool_score = 0.75
        
        return (iteration_score + tool_score) / 2
    
    def _evaluate_completeness(
        self,
        action_result: Any,
        tool_calls: Optional[List[Dict]]
    ) -> float:
        """评估完整性"""
        if action_result is None:
            return 0.0
        
        if isinstance(action_result, dict):
            required_keys = ["result", "summary"]
            has_keys = sum(1 for key in required_keys if key in action_result)
            return 0.5 + (has_keys * 0.25)
        
        if isinstance(action_result, str):
            if len(action_result) < 10:
                return 0.3
            elif len(action_result) < 100:
                return 0.6
            else:
                return 0.8
        
        return 0.7
    
    def _evaluate_coherence(self, tool_calls: Optional[List[Dict]]) -> float:
        """评估一致性"""
        if not tool_calls or len(tool_calls) <= 1:
            return 1.0
        
        coherence_scores = []
        for i in range(1, len(tool_calls)):
            prev_call = tool_calls[i - 1]
            curr_call = tool_calls[i]
            
            prev_name = prev_call.get("tool", "")
            curr_name = curr_call.get("tool", "")
            
            if prev_name == curr_name:
                coherence_scores.append(0.7)
            elif self._calls_are_related(prev_name, curr_name):
                coherence_scores.append(0.9)
            else:
                coherence_scores.append(0.5)
        
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.7
    
    def _calls_are_related(self, call1: str, call2: str) -> bool:
        """检查两次调用是否相关"""
        related_groups = [
            {"search", "filter", "sort"},
            {"read", "write", "update"},
            {"analyze", "report", "visualize"}
        ]
        
        for group in related_groups:
            if call1 in group and call2 in group:
                return True
        
        return False
    
    def _evaluate_novelty(
        self,
        action_result: Any,
        tool_calls: Optional[List[Dict]]
    ) -> float:
        """评估新颖性"""
        novelty_score = 0.5
        
        if tool_calls:
            unique_tools = len(set(call.get("tool") for call in tool_calls))
            novelty_score += min(0.3, unique_tools * 0.1)
        
        return min(1.0, novelty_score)
    
    def _generate_insights(
        self,
        action_result: Any,
        evaluation: Dict[EvaluationDimension, float],
        tool_calls: Optional[List[Dict]]
    ) -> List[str]:
        """
        生成洞察
        
        Args:
            action_result: 行动结果
            evaluation: 评估结果
            tool_calls: 工具调用列表
            
        Returns:
            洞察列表
        """
        insights = []
        
        low_dimensions = [
            dim for dim, score in evaluation.items()
            if score < self.evaluation_thresholds.get(dim, 0.7)
        ]
        
        for dim in low_dimensions:
            if dim == EvaluationDimension.CORRECTNESS:
                insights.append("执行过程中可能存在逻辑错误或使用了不当的工具")
            elif dim == EvaluationDimension.EFFICIENCY:
                insights.append("任务执行效率有提升空间，可能需要更直接的解决路径")
            elif dim == EvaluationDimension.COMPLETENESS:
                insights.append("结果可能不够完整，缺少某些关键信息")
            elif dim == EvaluationDimension.COHERENCE:
                insights.append("行动步骤之间的一致性可以改进")
        
        if evaluation.get(EvaluationDimension.NOVELTY, 0) > 0.7:
            insights.append("这次执行展现了创新性的解决方案")
        
        if not insights:
            insights.append("整体执行质量良好，保持当前策略")
        
        return insights
    
    def _suggest_improvements(
        self,
        action_result: Any,
        evaluation: Dict[EvaluationDimension, float],
        insights: List[str],
        tool_calls: Optional[List[Dict]]
    ) -> List[Dict[str, Any]]:
        """
        建议改进
        
        Args:
            action_result: 行动结果
            evaluation: 评估结果
            insights: 洞察列表
            tool_calls: 工具调用列表
            
        Returns:
            改进建议列表
        """
        improvements = []
        
        for dimension, score in evaluation.items():
            threshold = self.evaluation_thresholds.get(dimension, 0.7)
            
            if score < threshold:
                improvement = self._create_improvement(
                    dimension,
                    score,
                    threshold,
                    action_result,
                    tool_calls
                )
                improvements.append(improvement)
        
        improvements.sort(key=lambda x: x.get("priority", 0), reverse=True)
        
        return improvements[:5]
    
    def _create_improvement(
        self,
        dimension: EvaluationDimension,
        current_score: float,
        target_score: float,
        action_result: Any,
        tool_calls: Optional[List[Dict]]
    ) -> Dict[str, Any]:
        """创建改进建议"""
        improvement_types = {
            EvaluationDimension.CORRECTNESS: {
                "type": ImprovementType.ERROR_CORRECTION,
                "description": "需要纠正执行中的错误",
                "suggestions": [
                    "重新检查逻辑流程",
                    "验证输入参数的正确性",
                    "考虑使用更合适的工具"
                ],
                "priority": 0.9
            },
            EvaluationDimension.EFFICIENCY: {
                "type": ImprovementType.STRATEGY_ADJUSTMENT,
                "description": "需要优化执行效率",
                "suggestions": [
                    "减少不必要的工具调用",
                    "合并相似操作",
                    "使用更直接的解决路径"
                ],
                "priority": 0.7
            },
            EvaluationDimension.COMPLETENESS: {
                "type": ImprovementType.KNOWLEDGE_UPDATE,
                "description": "需要补充更多信息",
                "suggestions": [
                    "添加更多背景信息收集步骤",
                    "扩展分析维度",
                    "提供更详细的输出"
                ],
                "priority": 0.6
            },
            EvaluationDimension.COHERENCE: {
                "type": ImprovementType.STRATEGY_ADJUSTMENT,
                "description": "需要提高行动一致性",
                "suggestions": [
                    "确保步骤之间有清晰的逻辑关联",
                    "使用相关联的工具组合",
                    "保持目标的一致性"
                ],
                "priority": 0.5
            },
            EvaluationDimension.NOVELTY: {
                "type": ImprovementType.STRATEGY_ADJUSTMENT,
                "description": "可以考虑更创新的方法",
                "suggestions": [
                    "尝试新的工具组合",
                    "探索不同的解决思路",
                    "突破常规思维模式"
                ],
                "priority": 0.3
            }
        }
        
        base_improvement = improvement_types.get(dimension, {
            "type": ImprovementType.STRATEGY_ADJUSTMENT,
            "description": "一般性改进建议",
            "suggestions": [],
            "priority": 0.5
        })
        
        return {
            "dimension": dimension.value,
            "current_score": current_score,
            "target_score": target_score,
            **base_improvement
        }
    
    def _calculate_success_score(
        self,
        evaluation: Dict[EvaluationDimension, float],
        iterations: int
    ) -> float:
        """计算成功分数"""
        if not evaluation:
            return 0.0
        
        dimension_weights = {
            EvaluationDimension.CORRECTNESS: 0.35,
            EvaluationDimension.EFFICIENCY: 0.20,
            EvaluationDimension.COMPLETENESS: 0.25,
            EvaluationDimension.COHERENCE: 0.10,
            EvaluationDimension.NOVELTY: 0.10
        }
        
        weighted_score = sum(
            evaluation.get(dim, 0.0) * weight
            for dim, weight in dimension_weights.items()
        )
        
        if iterations > 15:
            weighted_score *= 0.8
        elif iterations > 10:
            weighted_score *= 0.9
        
        return round(weighted_score, 3)
    
    def _build_reasoning_chain(
        self,
        action_result: Any,
        evaluation: Dict[EvaluationDimension, float],
        improvements: List[Dict[str, Any]]
    ) -> List[str]:
        """构建推理链"""
        chain = []
        
        chain.append("执行结果分析：")
        if isinstance(action_result, str):
            chain.append(f"- 结果类型: 文本响应")
            chain.append(f"- 结果长度: {len(action_result)} 字符")
        elif isinstance(action_result, dict):
            chain.append(f"- 结果类型: 结构化数据")
            chain.append(f"- 包含键: {list(action_result.keys())}")
        else:
            chain.append(f"- 结果类型: {type(action_result).__name__}")
        
        chain.append("\n维度评估：")
        for dim, score in evaluation.items():
            status = "✓ 达标" if score >= self.evaluation_thresholds.get(dim, 0.7) else "✗ 未达标"
            chain.append(f"- {dim.value}: {score:.2f} {status}")
        
        if improvements:
            chain.append("\n优先改进项：")
            for imp in improvements[:3]:
                chain.append(f"- {imp.get('description', '未知')}")
        
        return chain
    
    def _store_reflection(self, reflection: ReflectionResult) -> None:
        """存储反思结果"""
        self.reflection_history.append(reflection)
        
        if len(self.reflection_history) > self.max_history:
            self.reflection_history = self.reflection_history[-self.max_history:]
        
        for insight in reflection.insights:
            if insight not in self.insights_cache["general"]:
                self.insights_cache["general"].append(insight)
        
        self.improvement_history.extend(reflection.improvements)
    
    def _create_empty_reflection(self) -> ReflectionResult:
        """创建空反思结果"""
        return ReflectionResult(
            reflection_id=str(uuid.uuid4()),
            timestamp=time.time(),
            level=ReflectionLevel.SURFACE,
            evaluation={},
            insights=["反思功能已禁用"],
            improvements=[],
            success_score=0.0
        )
    
    def get_reflection_history(
        self,
        level: Optional[ReflectionLevel] = None,
        limit: int = 10
    ) -> List[ReflectionResult]:
        """
        获取反思历史
        
        Args:
            level: 筛选特定层级的反思
            limit: 返回数量限制
            
        Returns:
            反思结果列表
        """
        history = self.reflection_history
        
        if level:
            history = [r for r in history if r.level == level]
        
        return history[-limit:]
    
    def get_average_scores(self) -> Dict[EvaluationDimension, float]:
        """获取各维度平均分"""
        if not self.reflection_history:
            return {dim: 0.0 for dim in EvaluationDimension}
        
        dimension_scores = defaultdict(list)
        
        for reflection in self.reflection_history:
            for dim, score in reflection.evaluation.items():
                dimension_scores[dim].append(score)
        
        return {
            dim: sum(scores) / len(scores) if scores else 0.0
            for dim, scores in dimension_scores.items()
        }
    
    def get_common_improvements(self, limit: int = 5) -> List[Dict[str, Any]]:
        """获取常见改进建议"""
        improvement_counts = defaultdict(int)
        improvement_examples = defaultdict(list)
        
        for imp in self.improvement_history:
            key = imp.get("description", "unknown")
            improvement_counts[key] += 1
            if len(improvement_examples[key]) < 2:
                improvement_examples[key].append(imp)
        
        sorted_improvements = sorted(
            improvement_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {
                "description": desc,
                "count": count,
                "examples": improvement_examples[desc]
            }
            for desc, count in sorted_improvements[:limit]
        ]
    
    def compare_reflections(
        self,
        reflection_id1: str,
        reflection_id2: str
    ) -> Dict[str, Any]:
        """比较两次反思"""
        r1 = next((r for r in self.reflection_history if r.reflection_id == reflection_id1), None)
        r2 = next((r for r in self.reflection_history if r.reflection_id == reflection_id2), None)
        
        if not r1 or not r2:
            return {"error": "Reflection not found"}
        
        comparison = {
            "reflection1": r1.to_dict(),
            "reflection2": r2.to_dict(),
            "score_difference": r2.success_score - r1.success_score,
            "dimension_differences": {}
        }
        
        all_dimensions = set(r1.evaluation.keys()) | set(r2.evaluation.keys())
        for dim in all_dimensions:
            score1 = r1.evaluation.get(dim, 0.0)
            score2 = r2.evaluation.get(dim, 0.0)
            comparison["dimension_differences"][dim.value] = score2 - score1
        
        return comparison
    
    def reset_history(self) -> None:
        """重置历史记录"""
        self.reflection_history = []
        self.action_evaluations = {}
        self.insights_cache.clear()
        self.improvement_history = []
        self.logger.info("Reflection history reset")


class AdvancedThoughtReflection(ThoughtReflection):
    """
    高级思维反思器
    
    提供更深入的反思能力，包括：
    - 跨任务模式识别
    - 学习曲线分析
    - 个性化建议生成
    """
    
    def __init__(self, *args, enable_pattern_recognition: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_pattern_recognition = enable_pattern_recognition
        self.pattern_history: Dict[str, List[Dict]] = defaultdict(list)
        self.learning_curves: Dict[str, List[float]] = defaultdict(list)
    
    def reflect_full(
        self,
        action_result: Any,
        tool_calls: Optional[List[Dict]] = None,
        iterations: int = 0,
        level: Optional[ReflectionLevel] = None,
        context: Optional[Dict] = None
    ) -> ReflectionResult:
        """完整反思（增强版）"""
        reflection = super().reflect_full(
            action_result,
            tool_calls,
            iterations,
            level,
            context
        )
        
        if self.enable_pattern_recognition:
            self._update_patterns(reflection, tool_calls)
        
        return reflection
    
    def _update_patterns(
        self,
        reflection: ReflectionResult,
        tool_calls: Optional[List[Dict]]
    ) -> None:
        """更新模式识别"""
        if tool_calls:
            pattern_key = self._extract_pattern(tool_calls)
            self.pattern_history[pattern_key].append({
                "timestamp": reflection.timestamp,
                "score": reflection.success_score,
                "insights": reflection.insights
            })
            
            if len(self.pattern_history[pattern_key]) > 10:
                self.pattern_history[pattern_key] = self.pattern_history[pattern_key][-10:]
    
    def _extract_pattern(self, tool_calls: List[Dict]) -> str:
        """提取调用模式"""
        tool_sequence = [call.get("tool", "") for call in tool_calls]
        return " -> ".join(tool_sequence[:5])
    
    def get_successful_patterns(self, min_score: float = 0.8) -> List[str]:
        """获取成功模式"""
        successful_patterns = []
        
        for pattern, history in self.pattern_history.items():
            avg_score = sum(h["score"] for h in history) / len(history) if history else 0
            if avg_score >= min_score:
                successful_patterns.append(pattern)
        
        return successful_patterns
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """获取学习洞察"""
        insights = {
            "total_reflections": len(self.reflection_history),
            "average_score": sum(r.success_score for r in self.reflection_history) / len(self.reflection_history) if self.reflection_history else 0,
            "successful_patterns": self.get_successful_patterns(),
            "dimension_trends": self._calculate_trends()
        }
        
        return insights
    
    def _calculate_trends(self) -> Dict[str, List[float]]:
        """计算趋势"""
        trends = defaultdict(list)
        
        sorted_reflections = sorted(
            self.reflection_history,
            key=lambda r: r.timestamp
        )
        
        for reflection in sorted_reflections[-20:]:
            for dim, score in reflection.evaluation.items():
                trends[dim.value].append(score)
        
        return dict(trends)


class SelfCorrectionReflection(ThoughtReflection):
    """
    自我纠正反思器
    
    专注于错误检测和纠正的反思机制
    """
    
    def __init__(self, *args, auto_correct: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.auto_correct = auto_correct
        self.error_patterns: Dict[str, int] = defaultdict(int)
        self.corrections_made: List[Dict] = []
    
    def _suggest_improvements(
        self,
        action_result: Any,
        evaluation: Dict[EvaluationDimension, float],
        insights: List[str],
        tool_calls: Optional[List[Dict]]
    ) -> List[Dict[str, Any]]:
        """建议改进（包含自我纠正）"""
        improvements = super()._suggest_improvements(
            action_result,
            evaluation,
            insights,
            tool_calls
        )
        
        if isinstance(action_result, dict) and action_result.get("error"):
            error_type = self._classify_error(action_result["error"])
            self.error_patterns[error_type] += 1
            
            corrections = self._generate_corrections(error_type, tool_calls)
            improvements.extend(corrections)
        
        return improvements
    
    def _classify_error(self, error_message: str) -> str:
        """错误分类"""
        error_lower = error_message.lower()
        
        if "timeout" in error_lower:
            return "timeout"
        elif "not found" in error_lower or "不存在" in error_lower:
            return "not_found"
        elif "permission" in error_lower or "权限" in error_lower:
            return "permission"
        elif "invalid" in error_lower or "无效" in error_lower:
            return "invalid_input"
        else:
            return "unknown"
    
    def _generate_corrections(
        self,
        error_type: str,
        tool_calls: Optional[List[Dict]]
    ) -> List[Dict[str, Any]]:
        """生成纠正方案"""
        corrections = {
            "timeout": {
                "type": ImprovementType.STRATEGY_ADJUSTMENT,
                "description": "超时错误纠正",
                "suggestions": [
                    "增加超时时间",
                    "优化执行步骤",
                    "分解复杂任务"
                ],
                "priority": 0.9
            },
            "not_found": {
                "type": ImprovementType.ERROR_CORRECTION,
                "description": "资源未找到",
                "suggestions": [
                    "验证资源标识符",
                    "检查路径是否正确",
                    "确认资源存在性"
                ],
                "priority": 0.8
            },
            "permission": {
                "type": ImprovementType.ERROR_CORRECTION,
                "description": "权限问题",
                "suggestions": [
                    "检查访问权限",
                    "使用替代方法",
                    "请求必要权限"
                ],
                "priority": 0.85
            },
            "invalid_input": {
                "type": ImprovementType.ERROR_CORRECTION,
                "description": "输入参数错误",
                "suggestions": [
                    "验证输入格式",
                    "检查参数类型",
                    "参考API文档"
                ],
                "priority": 0.8
            }
        }
        
        return [corrections.get(error_type, corrections["unknown"])]
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计"""
        total_errors = sum(self.error_patterns.values())
        
        return {
            "total_errors": total_errors,
            "error_types": dict(self.error_patterns),
            "error_distribution": {
                error: count / total_errors if total_errors > 0 else 0
                for error, count in self.error_patterns.items()
            },
            "corrections_made": len(self.corrections_made)
        }


__all__ = [
    "ThoughtReflection",
    "AdvancedThoughtReflection",
    "SelfCorrectionReflection",
    "ReflectionLevel",
    "EvaluationDimension",
    "ImprovementType",
    "ReflectionResult",
    "ActionEvaluation"
]
