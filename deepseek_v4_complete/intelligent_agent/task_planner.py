"""
任务规划器模块
负责将复杂任务分解为可执行的步骤序列
"""

import re
import logging
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import uuid


class StepType(Enum):
    """步骤类型枚举"""
    REASONING = "reasoning"                  # 推理步骤
    TOOL_CALL = "tool_call"                  # 工具调用
    CODE_EXECUTION = "code_execution"         # 代码执行
    MODEL_GENERATION = "model_generation"     # 模型生成
    INFORMATION_GATHERING = "information_gathering"  # 信息收集
    VERIFICATION = "verification"            # 验证步骤
    SYNTHESIS = "synthesis"                  # 综合步骤
    OUTPUT = "output"                        # 输出步骤


class PlanStrategy(Enum):
    """规划策略枚举"""
    LINEAR = "linear"                        # 线性规划
    HIERARCHICAL = "hierarchical"           # 层级规划
    REACTIVE = "reactive"                    # 反应式规划
    GOAL_DECOMPOSITION = "goal_decomposition"  # 目标分解
    BACKWARD_CHAINING = "backward_chaining"  # 逆向链
    FORWARD_CHAINING = "forward_chaining"    # 正向链


class StepStatus(Enum):
    """步骤状态枚举"""
    PENDING = "pending"                      # 待执行
    IN_PROGRESS = "in_progress"             # 执行中
    COMPLETED = "completed"                 # 已完成
    FAILED = "failed"                       # 失败
    SKIPPED = "skipped"                      # 跳过


@dataclass
class PlanStep:
    """
    计划步骤数据类
    
    表示一个独立的执行步骤，包含步骤的所有相关信息
    """
    step_id: str
    step_type: StepType
    content: Any
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    expected_output: Optional[str] = None
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.step_id:
            self.step_id = str(uuid.uuid4())[:8]
    
    def can_execute(self, completed_steps: Set[str]) -> bool:
        """
        检查步骤是否可以执行
        
        Args:
            completed_steps: 已完成的步骤ID集合
            
        Returns:
            是否可以执行
        """
        return all(dep in completed_steps for dep in self.dependencies)
    
    def mark_completed(self, result: Any) -> None:
        """标记步骤完成"""
        self.status = StepStatus.COMPLETED
        self.result = result
    
    def mark_failed(self, error: str) -> None:
        """标记步骤失败"""
        self.status = StepStatus.FAILED
        self.error = error
    
    def should_retry(self) -> bool:
        """检查是否应该重试"""
        return self.retry_count < self.max_retries
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "content": self.content,
            "description": self.description,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "retry_count": self.retry_count
        }


@dataclass
class ExecutionPlan:
    """
    执行计划数据类
    
    包含完整的任务执行计划
    """
    plan_id: str
    goal: str
    intent: str
    steps: List[Dict[str, Any]]
    strategy: PlanStrategy = PlanStrategy.LINEAR
    estimated_steps: int = 0
    created_at: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.plan_id:
            self.plan_id = str(uuid.uuid4())
    
    def get_step(self, step_id: str) -> Optional[Dict]:
        """获取指定步骤"""
        for step in self.steps:
            if step.get("step_id") == step_id:
                return step
        return None
    
    def get_executable_steps(self, completed: Set[str]) -> List[Dict]:
        """获取可执行的步骤"""
        executable = []
        for step in self.steps:
            deps = set(step.get("dependencies", []))
            if deps.issubset(completed):
                executable.append(step)
        return executable
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        验证计划的有效性
        
        Returns:
            (是否有效, 错误信息列表)
        """
        errors = []
        
        if not self.steps:
            errors.append("计划中没有步骤")
            return False, errors
        
        all_deps = set()
        step_ids = {step["step_id"] for step in self.steps}
        
        for step in self.steps:
            step_id = step.get("step_id")
            deps = set(step.get("dependencies", []))
            
            if not step_id:
                errors.append(f"步骤缺少ID: {step}")
            
            invalid_deps = deps - step_ids
            if invalid_deps:
                errors.append(f"步骤 {step_id} 依赖不存在的步骤: {invalid_deps}")
            
            all_deps.update(deps)
        
        dangling_deps = all_deps - step_ids
        if dangling_deps:
            errors.append(f"存在未定义的依赖: {dangling_deps}")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "intent": self.intent,
            "steps": self.steps,
            "strategy": self.strategy.value,
            "estimated_steps": self.estimated_steps,
            "metadata": self.metadata
        }


class TaskPlanner:
    """
    任务规划器
    
    负责将用户输入的复杂任务分解为可执行的步骤序列
    支持多种规划策略：
    1. 线性规划：简单顺序执行
    2. 层级规划：分层的任务分解
    3. 目标分解：将目标分解为子目标
    4. 逆向链：从目标反向推导所需步骤
    5. 正向链：从已知条件正向推导
    """
    
    def __init__(
        self,
        enable_planning: bool = True,
        default_strategy: PlanStrategy = PlanStrategy.GOAL_DECOMPOSITION,
        max_steps: int = 20
    ):
        """
        初始化任务规划器
        
        Args:
            enable_planning: 是否启用规划功能
            default_strategy: 默认规划策略
            max_steps: 最大步骤数
        """
        self.enable_planning = enable_planning
        self.default_strategy = default_strategy
        self.max_steps = max_steps
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self._init_decomposition_rules()
        self._init_step_templates()
    
    def _init_decomposition_rules(self):
        """初始化任务分解规则"""
        self.decomposition_rules = {
            "search": self._decompose_search_task,
            "execute_code": self._decompose_code_task,
            "analysis": self._decompose_analysis_task,
            "question": self._decompose_question_task,
            "task": self._decompose_general_task,
            "chat": self._decompose_chat_task
        }
    
    def _init_step_templates(self):
        """初始化步骤模板"""
        self.step_templates = {
            StepType.INFORMATION_GATHERING: {
                "description": "收集必要信息",
                "timeout": 30.0
            },
            StepType.REASONING: {
                "description": "分析推理",
                "timeout": 60.0
            },
            StepType.TOOL_CALL: {
                "description": "调用工具",
                "timeout": 120.0
            },
            StepType.CODE_EXECUTION: {
                "description": "执行代码",
                "timeout": 60.0
            },
            StepType.VERIFICATION: {
                "description": "验证结果",
                "timeout": 30.0
            },
            StepType.SYNTHESIS: {
                "description": "综合结果",
                "timeout": 60.0
            }
        }
    
    def create_plan(
        self,
        goal: str,
        intent: str,
        context: Optional[Dict] = None
    ) -> ExecutionPlan:
        """
        创建执行计划
        
        Args:
            goal: 用户目标/任务描述
            intent: 识别的意图类型
            context: 额外的上下文信息
            
        Returns:
            ExecutionPlan: 执行计划对象
        """
        if not self.enable_planning:
            return self._create_simple_plan(goal, intent)
        
        self.logger.info(f"Creating plan for goal: {goal[:50]}... with intent: {intent}")
        
        intent_key = intent.lower()
        decompose_func = self.decomposition_rules.get(
            intent_key,
            self._decompose_general_task
        )
        
        steps = decompose_func(goal, context)
        
        if len(steps) > self.max_steps:
            steps = self._prune_plan(steps)
        
        steps = self._optimize_plan(steps)
        
        plan = ExecutionPlan(
            plan_id=str(uuid.uuid4()),
            goal=goal,
            intent=intent,
            steps=steps,
            strategy=self.default_strategy,
            estimated_steps=len(steps),
            context=context or {}
        )
        
        is_valid, errors = plan.validate()
        if not is_valid:
            self.logger.warning(f"Plan validation errors: {errors}")
        
        return plan
    
    def _create_simple_plan(self, goal: str, intent: str) -> ExecutionPlan:
        """
        创建简单计划（规划禁用时）
        
        Args:
            goal: 目标
            intent: 意图
            
        Returns:
            简单执行计划
        """
        return ExecutionPlan(
            plan_id=str(uuid.uuid4()),
            goal=goal,
            intent=intent,
            steps=[{
                "type": "model_generation",
                "content": goal,
                "description": "直接处理任务",
                "step_id": str(uuid.uuid4())[:8]
            }]
        )
    
    def _decompose_search_task(self, goal: str, context: Optional[Dict]) -> List[Dict]:
        """
        分解搜索任务
        
        Args:
            goal: 搜索目标
            context: 上下文
            
        Returns:
            步骤列表
        """
        steps = [
            {
                "type": "reasoning",
                "content": f"分析搜索需求：{goal}",
                "description": "理解搜索目标",
                "step_id": str(uuid.uuid4())[:8]
            },
            {
                "type": "tool_call",
                "content": {
                    "name": "search",
                    "args": {"query": goal}
                },
                "description": "执行搜索",
                "step_id": str(uuid.uuid4())[:8]
            },
            {
                "type": "reasoning",
                "content": "分析搜索结果，提取关键信息",
                "description": "分析结果",
                "step_id": str(uuid.uuid4())[:8],
                "dependencies": []  # 将在优化时更新
            }
        ]
        
        steps[2]["dependencies"] = [steps[1]["step_id"]]
        
        return steps
    
    def _decompose_code_task(self, goal: str, context: Optional[Dict]) -> List[Dict]:
        """
        分解代码任务
        
        Args:
            goal: 代码任务描述
            context: 上下文
            
        Returns:
            步骤列表
        """
        steps = [
            {
                "type": "reasoning",
                "content": f"分析代码需求：{goal}",
                "description": "理解代码需求",
                "step_id": str(uuid.uuid4())[:8]
            },
            {
                "type": "model_generation",
                "content": f"编写满足需求的代码：{goal}",
                "description": "生成代码",
                "step_id": str(uuid.uuid4())[:8],
                "dependencies": []
            },
            {
                "type": "code_execution",
                "content": "运行生成的代码",
                "description": "执行代码验证",
                "step_id": str(uuid.uuid4())[:8],
                "dependencies": []
            },
            {
                "type": "verification",
                "content": "验证代码执行结果",
                "description": "验证结果正确性",
                "step_id": str(uuid.uuid4())[:8],
                "dependencies": []
            }
        ]
        
        steps[2]["dependencies"] = [steps[1]["step_id"]]
        steps[3]["dependencies"] = [steps[2]["step_id"]]
        
        return steps
    
    def _decompose_analysis_task(self, goal: str, context: Optional[Dict]) -> List[Dict]:
        """
        分解分析任务
        
        Args:
            goal: 分析目标
            context: 上下文
            
        Returns:
            步骤列表
        """
        steps = [
            {
                "type": "information_gathering",
                "content": "收集分析所需的数据和背景信息",
                "description": "收集数据",
                "step_id": str(uuid.uuid4())[:8]
            },
            {
                "type": "reasoning",
                "content": f"制定分析框架：{goal}",
                "description": "制定分析计划",
                "step_id": str(uuid.uuid4())[:8],
                "dependencies": []
            },
            {
                "type": "reasoning",
                "content": "执行详细分析",
                "description": "执行分析",
                "step_id": str(uuid.uuid4())[:8],
                "dependencies": []
            },
            {
                "type": "synthesis",
                "content": "综合分析结论",
                "description": "综合结果",
                "step_id": str(uuid.uuid4())[:8],
                "dependencies": []
            }
        ]
        
        steps[2]["dependencies"] = [steps[1]["step_id"]]
        steps[3]["dependencies"] = [steps[2]["step_id"]]
        
        return steps
    
    def _decompose_question_task(self, goal: str, context: Optional[Dict]) -> List[Dict]:
        """
        分解问答任务
        
        Args:
            goal: 问题描述
            context: 上下文
            
        Returns:
            步骤列表
        """
        steps = [
            {
                "type": "reasoning",
                "content": f"理解问题：{goal}",
                "description": "理解问题",
                "step_id": str(uuid.uuid4())[:8]
            },
            {
                "type": "model_generation",
                "content": f"回答问题：{goal}",
                "description": "生成答案",
                "step_id": str(uuid.uuid4())[:8],
                "dependencies": []
            },
            {
                "type": "verification",
                "content": "验证答案的准确性",
                "description": "验证答案",
                "step_id": str(uuid.uuid4())[:8],
                "dependencies": []
            }
        ]
        
        steps[2]["dependencies"] = [steps[1]["step_id"]]
        
        return steps
    
    def _decompose_chat_task(self, goal: str, context: Optional[Dict]) -> List[Dict]:
        """
        分解聊天任务
        
        Args:
            goal: 聊天内容
            context: 上下文
            
        Returns:
            步骤列表
        """
        return [
            {
                "type": "model_generation",
                "content": goal,
                "description": "生成回复",
                "step_id": str(uuid.uuid4())[:8]
            }
        ]
    
    def _decompose_general_task(self, goal: str, context: Optional[Dict]) -> List[Dict]:
        """
        分解通用任务
        
        Args:
            goal: 任务描述
            context: 上下文
            
        Returns:
            步骤列表
        """
        steps = [
            {
                "type": "reasoning",
                "content": f"分析任务：{goal}",
                "description": "理解任务目标",
                "step_id": str(uuid.uuid4())[:8]
            },
            {
                "type": "reasoning",
                "content": "确定执行步骤",
                "description": "规划执行方案",
                "step_id": str(uuid.uuid4())[:8],
                "dependencies": []
            },
            {
                "type": "model_generation",
                "content": goal,
                "description": "执行任务",
                "step_id": str(uuid.uuid4())[:8],
                "dependencies": []
            }
        ]
        
        steps[2]["dependencies"] = [steps[1]["step_id"]]
        
        return steps
    
    def _prune_plan(self, steps: List[Dict]) -> List[Dict]:
        """
        精简计划，限制步骤数量
        
        Args:
            steps: 原始步骤列表
            
        Returns:
            精简后的步骤列表
        """
        if len(steps) <= self.max_steps:
            return steps
        
        self.logger.warning(f"Plan has {len(steps)} steps, pruning to {self.max_steps}")
        
        critical_steps = [s for s in steps if s.get("type") in ["tool_call", "code_execution"]]
        other_steps = [s for s in steps if s not in critical_steps]
        
        remaining_slots = self.max_steps - len(critical_steps)
        
        if remaining_slots > 0 and other_steps:
            result = critical_steps + other_steps[:remaining_slots]
        else:
            result = critical_steps[:self.max_steps]
        
        return self._rebuild_dependencies(result)
    
    def _rebuild_dependencies(self, steps: List[Dict]) -> List[Dict]:
        """
        重建步骤依赖关系
        
        Args:
            steps: 步骤列表
            
        Returns:
            更新依赖后的步骤列表
        """
        step_map = {step["step_id"]: i for i, step in enumerate(steps)}
        
        for step in steps:
            old_deps = step.get("dependencies", [])
            new_deps = []
            
            for dep in old_deps:
                if dep in step_map:
                    new_deps.append(dep)
            
            step["dependencies"] = new_deps
        
        return steps
    
    def _optimize_plan(self, steps: List[Dict]) -> List[Dict]:
        """
        优化计划执行顺序
        
        Args:
            steps: 原始步骤列表
            
        Returns:
            优化后的步骤列表
        """
        if self.default_strategy == PlanStrategy.LINEAR:
            return steps
        
        return self._topological_sort(steps)
    
    def _topological_sort(self, steps: List[Dict]) -> List[Dict]:
        """
        拓扑排序优化步骤顺序
        
        Args:
            steps: 步骤列表
            
        Returns:
            排序后的步骤列表
        """
        in_degree = defaultdict(int)
        step_map = {step["step_id"]: step for step in steps}
        adj_list = defaultdict(list)
        
        for step in steps:
            step_id = step["step_id"]
            if step_id not in in_degree:
                in_degree[step_id] = 0
            
            for dep in step.get("dependencies", []):
                if dep in step_map:
                    adj_list[dep].append(step_id)
                    in_degree[step_id] += 1
        
        queue = [sid for sid, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(step_map[current])
            
            for neighbor in adj_list[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(steps):
            self.logger.warning("Cycle detected in plan dependencies, using original order")
            return steps
        
        return result
    
    def create_subplan(
        self,
        parent_plan: ExecutionPlan,
        sub_goal: str,
        dependencies: Optional[List[str]] = None
    ) -> ExecutionPlan:
        """
        创建子计划
        
        Args:
            parent_plan: 父计划
            sub_goal: 子目标
            dependencies: 依赖的父计划步骤
            
        Returns:
            子执行计划
        """
        sub_steps = self._decompose_general_task(sub_goal, parent_plan.context)
        
        if dependencies:
            for step in sub_steps:
                step["dependencies"] = list(dependencies) + step.get("dependencies", [])
        
        return ExecutionPlan(
            plan_id=str(uuid.uuid4()),
            goal=sub_goal,
            intent="task",
            steps=sub_steps,
            context=parent_plan.context
        )
    
    def merge_plans(
        self,
        plans: List[ExecutionPlan],
        merge_strategy: str = "sequential"
    ) -> ExecutionPlan:
        """
        合并多个计划
        
        Args:
            plans: 计划列表
            merge_strategy: 合并策略 ("sequential", "parallel", "conditional")
            
        Returns:
            合并后的计划
        """
        if not plans:
            raise ValueError("No plans to merge")
        
        if len(plans) == 1:
            return plans[0]
        
        merged_steps = []
        
        for i, plan in enumerate(plans):
            for step in plan.steps:
                new_step = step.copy()
                new_step["step_id"] = f"{new_step['step_id']}_plan{i}"
                merged_steps.append(new_step)
        
        if merge_strategy == "sequential":
            merged_steps = self._optimize_plan(merged_steps)
        
        return ExecutionPlan(
            plan_id=str(uuid.uuid4()),
            goal="; ".join(p.goal for p in plans),
            intent="complex",
            steps=merged_steps,
            strategy=self.default_strategy
        )
    
    def update_plan(
        self,
        plan: ExecutionPlan,
        step_results: Dict[str, Any],
        failed_step_id: Optional[str] = None
    ) -> ExecutionPlan:
        """
        更新计划状态
        
        Args:
            plan: 当前计划
            step_results: 步骤结果字典
            failed_step_id: 失败的步骤ID
            
        Returns:
            更新后的计划
        """
        for step in plan.steps:
            step_id = step["step_id"]
            
            if step_id in step_results:
                step["result"] = step_results[step_id]
                step["status"] = StepStatus.COMPLETED.value
            elif step_id == failed_step_id:
                step["status"] = StepStatus.FAILED.value
        
        return plan
    
    def estimate_execution_time(self, plan: ExecutionPlan) -> float:
        """
        估算计划执行时间
        
        Args:
            plan: 执行计划
            
        Returns:
            估算时间（秒）
        """
        total_time = 0.0
        
        for step in plan.steps:
            step_type = step.get("type")
            template = self.step_templates.get(StepType(step_type))
            
            if template and "timeout" in template:
                total_time += template["timeout"]
            else:
                total_time += 30.0
        
        return total_time
    
    def get_critical_path(self, plan: ExecutionPlan) -> List[str]:
        """
        获取关键路径
        
        Args:
            plan: 执行计划
            
        Returns:
            关键路径上的步骤ID列表
        """
        step_map = {step["step_id"]: step for step in plan.steps}
        earliest_start = {}
        earliest_finish = {}
        
        sorted_steps = self._topological_sort(plan.steps.copy())
        
        for step in sorted_steps:
            step_id = step["step_id"]
            deps = step.get("dependencies", [])
            
            if not deps:
                earliest_start[step_id] = 0
            else:
                earliest_start[step_id] = max(
                    earliest_finish.get(dep, 0) for dep in deps
                )
            
            template = self.step_templates.get(
                StepType(step.get("type", "reasoning")),
                self.step_templates[StepType.REASONING]
            )
            duration = template.get("timeout", 30.0)
            earliest_finish[step_id] = earliest_start[step_id] + duration
        
        if not earliest_finish:
            return []
        
        max_finish = max(earliest_finish.values())
        critical_path = []
        
        for step_id, finish_time in earliest_finish.items():
            if finish_time == max_finish:
                critical_path.append(step_id)
                break
        
        return critical_path


class HierarchicalPlanner(TaskPlanner):
    """
    层级规划器
    
    支持多层级的任务分解，适用于复杂任务
    """
    
    def __init__(self, *args, max_depth: int = 5, **kwargs):
        """
        初始化层级规划器
        
        Args:
            max_depth: 最大分解深度
        """
        super().__init__(*args, **kwargs)
        self.max_depth = max_depth
    
    def create_hierarchical_plan(
        self,
        goal: str,
        intent: str,
        context: Optional[Dict] = None,
        depth: int = 0
    ) -> Dict:
        """
        创建层级计划
        
        Args:
            goal: 目标
            intent: 意图
            context: 上下文
            depth: 当前深度
            
        Returns:
            层级计划字典
        """
        if depth >= self.max_depth:
            return self._create_leaf_node(goal, intent)
        
        sub_goals = self._decompose_goal(goal, intent)
        
        if not sub_goals:
            return self._create_leaf_node(goal, intent)
        
        children = []
        for sub_goal in sub_goals:
            child_plan = self.create_hierarchical_plan(
                sub_goal["goal"],
                sub_goal.get("intent", intent),
                context,
                depth + 1
            )
            children.append(child_plan)
        
        return {
            "type": "hierarchical",
            "goal": goal,
            "intent": intent,
            "depth": depth,
            "children": children,
            "step_id": str(uuid.uuid4())[:8]
        }
    
    def _decompose_goal(self, goal: str, intent: str) -> List[Dict]:
        """
        分解目标为子目标
        
        Args:
            goal: 目标
            intent: 意图
            
        Returns:
            子目标列表
        """
        return [
            {"goal": f"子目标1: {goal}", "intent": intent},
            {"goal": f"子目标2: {goal}", "intent": intent}
        ]
    
    def _create_leaf_node(self, goal: str, intent: str) -> Dict:
        """
        创建叶子节点
        
        Args:
            goal: 目标
            intent: 意图
            
        Returns:
            叶子节点字典
        """
        return {
            "type": "leaf",
            "goal": goal,
            "intent": intent,
            "step_id": str(uuid.uuid4())[:8],
            "depth": self.max_depth
        }
    
    def flatten_plan(self, hierarchical_plan: Dict) -> List[Dict]:
        """
        扁平化层级计划
        
        Args:
            hierarchical_plan: 层级计划
            
        Returns:
            扁平化的步骤列表
        """
        flat_steps = []
        
        def traverse(node: Dict):
            if node["type"] == "leaf":
                flat_steps.append({
                    "step_id": node["step_id"],
                    "content": node["goal"],
                    "description": f"执行: {node['goal'][:30]}...",
                    "type": "model_generation",
                    "depth": node.get("depth", 0)
                })
            elif "children" in node:
                for child in node["children"]:
                    traverse(child)
        
        traverse(hierarchical_plan)
        return flat_steps


class ReactivePlanner(TaskPlanner):
    """
    反应式规划器
    
    能够根据执行结果动态调整计划
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptation_rules: Dict[str, Callable] = {}
        self._init_adaptation_rules()
    
    def _init_adaptation_rules(self):
        """初始化自适应规则"""
        self.adaptation_rules = {
            "tool_failed": self._adapt_on_tool_failure,
            "unexpected_result": self._adapt_on_unexpected_result,
            "step_timeout": self._adapt_on_timeout,
            "resource_constraint": self._adapt_on_resource_constraint
        }
    
    def adapt_plan(
        self,
        plan: ExecutionPlan,
        event: str,
        event_data: Dict
    ) -> ExecutionPlan:
        """
        根据事件调整计划
        
        Args:
            plan: 当前计划
            event: 事件类型
            event_data: 事件数据
            
        Returns:
            调整后的计划
        """
        if event not in self.adaptation_rules:
            return plan
        
        adaptation_func = self.adaptation_rules[event]
        return adaptation_func(plan, event_data)
    
    def _adapt_on_tool_failure(
        self,
        plan: ExecutionPlan,
        event_data: Dict
    ) -> ExecutionPlan:
        """工具失败时的自适应"""
        failed_step_id = event_data.get("step_id")
        
        for step in plan.steps:
            if step["step_id"] == failed_step_id:
                step["retry_count"] = step.get("retry_count", 0) + 1
                
                if step["retry_count"] >= 3:
                    alternative = event_data.get("alternative")
                    if alternative:
                        step["content"] = alternative
                        step["status"] = StepStatus.PENDING.value
        
        return plan
    
    def _adapt_on_unexpected_result(
        self,
        plan: ExecutionPlan,
        event_data: Dict
    ) -> ExecutionPlan:
        """意外结果时的自适应"""
        recovery_steps = event_data.get("recovery_steps", [])
        
        if recovery_steps:
            plan.steps.extend(recovery_steps)
        
        return plan
    
    def _adapt_on_timeout(
        self,
        plan: ExecutionPlan,
        event_data: Dict
    ) -> ExecutionPlan:
        """超时的自适应"""
        timed_out_step_id = event_data.get("step_id")
        
        for step in plan.steps:
            if step["step_id"] == timed_out_step_id:
                step["timeout"] = step.get("timeout", 30) * 2
        
        return plan
    
    def _adapt_on_resource_constraint(
        self,
        plan: ExecutionPlan,
        event_data: Dict
    ) -> ExecutionPlan:
        """资源约束时的自适应"""
        plan = self._prune_plan(plan.steps.copy())
        
        return plan


__all__ = [
    "TaskPlanner",
    "HierarchicalPlanner",
    "ReactivePlanner",
    "StepType",
    "PlanStrategy",
    "StepStatus",
    "PlanStep",
    "ExecutionPlan"
]
