"""
核心智能代理模块
负责协调和管理整个智能代理系统的核心运行逻辑
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid

from .intent_recognize import IntentRecognizer
from .task_planner import TaskPlanner
from .thought_reflect import ThoughtReflection
from .tool_manager import ToolManager
from .function_call_parse import FunctionCallParser
from .code_interpreter import CodeInterpreter
from .plugin_loader import PluginLoader
from .memory_think_pool import MemoryThinkPool


class AgentState(Enum):
    """代理状态枚举"""
    IDLE = "idle"
    THINKING = "thinking"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    WAITING_TOOL = "waiting_tool"
    FINISHED = "finished"
    ERROR = "error"


class ErrorType(Enum):
    """错误类型枚举"""
    TOOL_NOT_FOUND = "tool_not_found"
    TOOL_EXECUTION_FAILED = "tool_execution_failed"
    PARSING_FAILED = "parsing_failed"
    PLANNING_FAILED = "planning_failed"
    MAX_ITERATIONS = "max_iterations"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ExecutionResult:
    """执行结果数据类"""
    success: bool
    content: Any
    error: Optional[str] = None
    error_type: Optional[ErrorType] = None
    tool_calls: List[Dict] = field(default_factory=list)
    execution_time: float = 0.0
    iterations: int = 0
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "success": self.success,
            "content": self.content,
            "error": self.error,
            "error_type": self.error_type.value if self.error_type else None,
            "tool_calls": self.tool_calls,
            "execution_time": self.execution_time,
            "iterations": self.iterations
        }


@dataclass
class AgentConfig:
    """代理配置类"""
    max_iterations: int = 10
    max_execution_time: float = 60.0
    enable_reflection: bool = True
    enable_memory: bool = True
    enable_planning: bool = True
    enable_code_interpreter: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    model_name: str = "deepseek-v4"
    stream_output: bool = False
    
    
class CoreAgent:
    """
    核心智能代理类
    
    这是整个智能代理系统的核心 orchestrator，负责：
    1. 协调各个子模块的运作
    2. 管理代理状态
    3. 处理用户输入并生成响应
    4. 执行工具调用和代码解释
    5. 维护对话历史和思想记忆
    """
    
    def __init__(
        self,
        model: Any = None,
        config: Optional[AgentConfig] = None,
        tools: Optional[Dict[str, callable]] = None,
        plugins_dir: Optional[str] = None
    ):
        """
        初始化核心智能代理
        
        Args:
            model: 语言模型实例（用于生成文本）
            config: 代理配置
            tools: 工具字典 {tool_name: tool_function}
            plugins_dir: 插件目录路径
        """
        self.config = config or AgentConfig()
        self.model = model
        
        # 初始化各子模块
        self.intent_recognizer = IntentRecognizer()
        self.task_planner = TaskPlanner(enable_planning=self.config.enable_planning)
        self.thought_reflector = ThoughtReflection(enable=self.config.enable_reflection)
        self.tool_manager = ToolManager()
        self.function_parser = FunctionCallParser()
        self.code_interpreter = CodeInterpreter(enable=self.config.enable_code_interpreter)
        self.plugin_loader = PluginLoader(plugins_dir=plugins_dir)
        self.memory_pool = MemoryThinkPool(capacity=1000)
        
        # 注册工具
        if tools:
            for name, func in tools.items():
                self.register_tool(name, func)
        
        # 代理状态
        self.state = AgentState.IDLE
        self.conversation_history: List[Dict[str, str]] = []
        self.current_task_id: Optional[str] = None
        
        # 日志配置
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 执行统计
        self.stats = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "total_tool_calls": 0,
            "total_reflections": 0
        }
    
    def register_tool(self, name: str, func: callable, description: str = "") -> None:
        """
        注册工具到工具管理器
        
        Args:
            name: 工具名称
            func: 工具函数
            description: 工具描述
        """
        self.tool_manager.register(name, func, description)
        self.logger.info(f"Registered tool: {name}")
    
    def register_tools(self, tools: Dict[str, callable]) -> None:
        """
        批量注册工具
        
        Args:
            tools: 工具字典
        """
        for name, func in tools.items():
            self.register_tool(name, func)
    
    async def arun(self, task: str, context: Optional[Dict] = None) -> ExecutionResult:
        """
        异步运行代理处理任务
        
        Args:
            task: 用户输入任务
            context: 额外的上下文信息
            
        Returns:
            ExecutionResult: 执行结果
        """
        start_time = time.time()
        task_id = str(uuid.uuid4())
        self.current_task_id = task_id
        self.stats["total_tasks"] += 1
        
        try:
            # 状态更新：思考中
            self.state = AgentState.THINKING
            
            # 添加到对话历史
            self.conversation_history.append({
                "role": "user",
                "content": task,
                "task_id": task_id
            })
            
            # 意图识别
            intent = self.intent_recognizer.recognize(task)
            self.logger.info(f"Recognized intent: {intent}")
            
            # 任务规划
            if self.config.enable_planning:
                self.state = AgentState.PLANNING
                plan = self.task_planner.create_plan(task, intent, context)
                self.logger.info(f"Created plan with {len(plan.steps)} steps")
            else:
                plan = self.task_planner.create_plan(task, intent, context)
            
            # 执行计划
            self.state = AgentState.EXECUTING
            result = await self._execute_plan(plan, context)
            result.execution_time = time.time() - start_time
            
            # 反思
            if self.config.enable_reflection and result.success:
                self.state = AgentState.REFLECTING
                reflection = await self._reflect(result)
                result.content = self._incorporate_reflection(result.content, reflection)
                self.stats["total_reflections"] += 1
            
            # 更新状态
            self.state = AgentState.FINISHED
            self.stats["successful_tasks"] += 1
            
            # 存储到记忆池
            if self.config.enable_memory:
                self._add_to_memory(task, result)
            
            return result
            
        except Exception as e:
            self.state = AgentState.ERROR
            self.stats["failed_tasks"] += 1
            self.logger.error(f"Error in agent execution: {e}")
            return ExecutionResult(
                success=False,
                content=None,
                error=str(e),
                error_type=ErrorType.UNKNOWN_ERROR,
                execution_time=time.time() - start_time
            )
    
    def run(self, task: str, context: Optional[Dict] = None) -> ExecutionResult:
        """
        同步运行代理处理任务
        
        Args:
            task: 用户输入任务
            context: 额外的上下文信息
            
        Returns:
            ExecutionResult: 执行结果
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.arun(task, context))
    
    async def _execute_plan(self, plan, context: Optional[Dict] = None) -> ExecutionResult:
        """
        执行计划
        
        Args:
            plan: 任务计划
            context: 上下文信息
            
        Returns:
            ExecutionResult: 执行结果
        """
        iterations = 0
        all_results = []
        tool_calls = []
        
        for step in plan.steps:
            if iterations >= self.config.max_iterations:
                return ExecutionResult(
                    success=False,
                    content=all_results,
                    error="Maximum iterations reached",
                    error_type=ErrorType.MAX_ITERATIONS,
                    tool_calls=tool_calls,
                    iterations=iterations
                )
            
            step_type = step.get("type")
            step_content = step.get("content")
            
            if step_type == "tool_call":
                self.state = AgentState.WAITING_TOOL
                result = await self._execute_tool_call(step_content, context)
                tool_calls.append({
                    "tool": step_content.get("name"),
                    "args": step_content.get("args"),
                    "result": result
                })
                self.stats["total_tool_calls"] += 1
                all_results.append(result)
                
            elif step_type == "code_execution":
                result = self.code_interpreter.execute(step_content, context)
                all_results.append(result)
                
            elif step_type == "model_generation":
                result = await self._generate_with_model(step_content, context)
                all_results.append(result)
                
            elif step_type == "reasoning":
                reasoning_result = await self._reason(step_content, context)
                all_results.append(reasoning_result)
            
            iterations += 1
        
        # 聚合结果
        final_content = self._aggregate_results(all_results)
        
        return ExecutionResult(
            success=True,
            content=final_content,
            tool_calls=tool_calls,
            iterations=iterations
        )
    
    async def _execute_tool_call(
        self,
        tool_call: Dict[str, Any],
        context: Optional[Dict]
    ) -> Any:
        """
        执行工具调用
        
        Args:
            tool_call: 工具调用信息
            context: 上下文
            
        Returns:
            工具执行结果
        """
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        
        # 检查工具是否存在
        if not self.tool_manager.has_tool(tool_name):
            raise ValueError(f"Tool not found: {tool_name}")
        
        # 合并上下文
        if context:
            tool_args = {**context, **tool_args}
        
        # 执行工具
        return await self.tool_manager.acall(tool_name, **tool_args)
    
    async def _generate_with_model(
        self,
        prompt: str,
        context: Optional[Dict]
    ) -> str:
        """
        使用模型生成文本
        
        Args:
            prompt: 提示词
            context: 上下文
            
        Returns:
            生成的文本
        """
        if self.model is None:
            return f"Model not configured. Received prompt: {prompt}"
        
        # 合并对话历史和上下文
        full_prompt = self._build_prompt(prompt, context)
        
        if asyncio.iscoroutinefunction(self.model.generate):
            return await self.model.generate(full_prompt)
        else:
            return self.model.generate(full_prompt)
    
    def _build_prompt(self, prompt: str, context: Optional[Dict]) -> str:
        """
        构建完整的提示词
        
        Args:
            prompt: 用户提示
            context: 上下文
            
        Returns:
            完整的提示词
        """
        parts = []
        
        # 添加对话历史
        if self.conversation_history:
            history_prompt = "## Conversation History:\n"
            for msg in self.conversation_history[-5:]:
                history_prompt += f"- {msg['role']}: {msg['content']}\n"
            parts.append(history_prompt)
        
        # 添加上下文
        if context:
            context_prompt = "## Context:\n"
            for key, value in context.items():
                context_prompt += f"- {key}: {value}\n"
            parts.append(context_prompt)
        
        # 添加当前任务
        parts.append(f"## Current Task:\n{prompt}")
        
        return "\n\n".join(parts)
    
    async def _reason(self, reasoning_task: str, context: Optional[Dict]) -> str:
        """
        执行推理任务
        
        Args:
            reasoning_task: 推理任务描述
            context: 上下文
            
        Returns:
            推理结果
        """
        reasoning_prompt = f"""Please reason through the following task step by step:

Task: {reasoning_task}

Think about:
1. What is the goal?
2. What information do we have?
3. What are the possible approaches?
4. What is the best path forward?

Provide your reasoning and conclusion.
"""
        
        return await self._generate_with_model(reasoning_prompt, context)
    
    async def _reflect(self, result: ExecutionResult) -> Dict[str, Any]:
        """
        执行反思
        
        Args:
            result: 执行结果
            
        Returns:
            反思结果
        """
        reflection = self.thought_reflector.reflect(
            action_result=result.content,
            tool_calls=result.tool_calls,
            iterations=result.iterations
        )
        
        # 添加到记忆池
        self.memory_pool.add({
            "type": "reflection",
            "content": reflection,
            "task_id": self.current_task_id,
            "timestamp": time.time()
        })
        
        return reflection
    
    def _incorporate_reflection(self, content: Any, reflection: Dict[str, Any]) -> Any:
        """
        将反思结果融入内容
        
        Args:
            content: 原始内容
            reflection: 反思结果
            
        Returns:
            融入反思后的内容
        """
        if isinstance(content, str):
            improvements = reflection.get("improvements", [])
            if improvements:
                content += "\n\n## Suggestions for Improvement:\n"
                content += "\n".join(f"- {imp}" for imp in improvements)
        
        return content
    
    def _aggregate_results(self, results: List[Any]) -> Any:
        """
        聚合多个结果
        
        Args:
            results: 结果列表
            
        Returns:
            聚合后的结果
        """
        if not results:
            return None
        
        if len(results) == 1:
            return results[0]
        
        # 尝试合并为结构化响应
        aggregated = {
            "steps": len(results),
            "results": results,
            "summary": f"Completed {len(results)} steps successfully"
        }
        
        return aggregated
    
    def _add_to_memory(self, task: str, result: ExecutionResult) -> None:
        """
        添加任务到记忆池
        
        Args:
            task: 任务描述
            result: 执行结果
        """
        memory_entry = {
            "type": "task",
            "task": task,
            "result": result.content,
            "success": result.success,
            "task_id": self.current_task_id,
            "timestamp": time.time(),
            "stats": {
                "iterations": result.iterations,
                "execution_time": result.execution_time,
                "tool_calls_count": len(result.tool_calls)
            }
        }
        
        self.memory_pool.add(memory_entry)
    
    def get_memory(self, query: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """
        获取记忆
        
        Args:
            query: 查询关键词
            limit: 返回数量限制
            
        Returns:
            记忆列表
        """
        return self.memory_pool.retrieve(query, limit)
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        获取对话历史
        
        Args:
            limit: 限制返回条数
            
        Returns:
            对话历史列表
        """
        if limit:
            return self.conversation_history[-limit:]
        return self.conversation_history
    
    def clear_history(self) -> None:
        """清空对话历史"""
        self.conversation_history = []
        self.logger.info("Conversation history cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        stats = self.stats.copy()
        stats["state"] = self.state.value
        stats["tool_count"] = len(self.tool_manager.list_tools())
        stats["memory_size"] = len(self.memory_pool.pool)
        return stats
    
    def reset(self) -> None:
        """重置代理状态"""
        self.state = AgentState.IDLE
        self.current_task_id = None
        self.logger.info("Agent state reset")
    
    async def chat(self, message: str) -> str:
        """
        简单的对话接口
        
        Args:
            message: 用户消息
            
        Returns:
            代理回复
        """
        result = await self.arun(message)
        
        if result.success:
            if isinstance(result.content, str):
                return result.content
            elif isinstance(result.content, dict):
                return result.content.get("summary", str(result.content))
            else:
                return str(result.content)
        else:
            return f"Error: {result.error}"
    
    def register_plugin(self, plugin_path: str) -> bool:
        """
        注册插件
        
        Args:
            plugin_path: 插件路径
            
        Returns:
            是否成功
        """
        plugin = self.plugin_loader.load_plugin(plugin_path)
        if plugin:
            # 注册插件提供的工具
            tools = plugin.get("tools", {})
            self.register_tools(tools)
            return True
        return False
    
    def get_available_tools(self) -> List[Dict[str, str]]:
        """
        获取可用工具列表
        
        Returns:
            工具信息列表
        """
        return self.tool_manager.list_tools()
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"CoreAgent(state={self.state.value}, "
            f"tools={len(self.tool_manager.list_tools())}, "
            f"memory_size={len(self.memory_pool.pool)})"
        )


class MultiTurnAgent(CoreAgent):
    """
    多轮对话代理
    
    继承自CoreAgent，支持多轮对话和上下文保持
    """
    
    def __init__(self, *args, max_turns: int = 20, **kwargs):
        """
        初始化多轮对话代理
        
        Args:
            max_turns: 最大对话轮数
        """
        super().__init__(*args, **kwargs)
        self.max_turns = max_turns
        self.turn_count = 0
        self.session_context: Dict[str, Any] = {}
    
    async def arun(self, task: str, context: Optional[Dict] = None) -> ExecutionResult:
        """
        异步运行代理处理任务（带多轮对话支持）
        
        Args:
            task: 用户输入任务
            context: 额外的上下文信息
            
        Returns:
            ExecutionResult: 执行结果
        """
        self.turn_count += 1
        
        # 检查是否超过最大轮数
        if self.turn_count > self.max_turns:
            return ExecutionResult(
                success=False,
                content=None,
                error="Maximum conversation turns reached",
                error_type=ErrorType.MAX_ITERATIONS
            )
        
        # 合并会话上下文
        merged_context = {**self.session_context, **(context or {})}
        
        # 调用父类方法
        result = await super().arun(task, merged_context)
        
        # 更新会话上下文
        if result.success and isinstance(result.content, dict):
            self.session_context.update(result.content.get("context_updates", {}))
        
        return result
    
    def reset_session(self) -> None:
        """重置会话"""
        self.turn_count = 0
        self.session_context = {}
        self.clear_history()


class ReActAgent(CoreAgent):
    """
    ReAct (Reason + Act) 代理
    
    实现ReAct推理模式，交替进行推理和行动
    """
    
    def __init__(self, *args, max_steps: int = 10, **kwargs):
        """
        初始化ReAct代理
        
        Args:
            max_steps: 最大推理步骤数
        """
        super().__init__(*args, **kwargs)
        self.max_steps = max_steps
    
    async def arun(self, task: str, context: Optional[Dict] = None) -> ExecutionResult:
        """
        使用ReAct模式运行代理
        
        Args:
            task: 用户输入任务
            context: 额外的上下文信息
            
        Returns:
            ExecutionResult: 执行结果
        """
        start_time = time.time()
        self.current_task_id = str(uuid.uuid4())
        self.stats["total_tasks"] += 1
        
        observation = None
        thought_history = []
        action_history = []
        
        for step in range(self.max_steps):
            # 推理阶段
            self.state = AgentState.THINKING
            thought = await self._reason_with_observation(task, observation, thought_history)
            thought_history.append(thought)
            
            # 检查是否完成
            if self._is_finished(thought):
                return ExecutionResult(
                    success=True,
                    content={
                        "final_answer": thought,
                        "thought_history": thought_history,
                        "action_history": action_history
                    },
                    execution_time=time.time() - start_time,
                    iterations=step + 1
                )
            
            # 行动阶段
            self.state = AgentState.EXECUTING
            action = self._extract_action(thought)
            
            if action:
                action_result = await self._execute_action(action, context)
                observation = action_result
                action_history.append({
                    "action": action,
                    "result": action_result
                })
            else:
                observation = thought
        
        # 达到最大步骤
        return ExecutionResult(
            success=False,
            content={
                "thought_history": thought_history,
                "action_history": action_history,
                "last_observation": observation
            },
            error="Maximum steps reached",
            error_type=ErrorType.MAX_ITERATIONS,
            execution_time=time.time() - start_time
        )
    
    async def _reason_with_observation(
        self,
        task: str,
        observation: Any,
        history: List[str]
    ) -> str:
        """
        基于观察进行推理
        
        Args:
            task: 原始任务
            observation: 上一步的观察结果
            history: 思考历史
            
        Returns:
            推理结果
        """
        prompt_parts = [f"Task: {task}\n"]
        
        if history:
            prompt_parts.append("\nThought History:")
            for i, thought in enumerate(history):
                prompt_parts.append(f"{i+1}. {thought}")
        
        if observation:
            prompt_parts.append(f"\nLast Observation: {observation}")
        
        prompt_parts.append("""
Think about what to do next. Consider:
1. What information do we have now?
2. What is the next logical step?
3. Should we use a tool or provide the final answer?

Provide your next thought.
""")
        
        return await self._generate_with_model("\n".join(prompt_parts), None)
    
    def _extract_action(self, thought: str) -> Optional[Dict[str, Any]]:
        """
        从思考中提取行动
        
        Args:
            thought: 思考内容
            
        Returns:
            行动字典，如果无需行动则返回None
        """
        # 简单的行动提取逻辑
        if "use tool:" in thought.lower():
            for tool_name in self.tool_manager.list_tools():
                if tool_name in thought.lower():
                    # 提取参数（简化版）
                    return {"name": tool_name, "args": {}}
        
        return None
    
    async def _execute_action(self, action: Dict, context: Optional[Dict]) -> Any:
        """
        执行行动
        
        Args:
            action: 行动字典
            context: 上下文
            
        Returns:
            执行结果
        """
        return await self._execute_tool_call(action, context)
    
    def _is_finished(self, thought: str) -> bool:
        """
        检查是否完成
        
        Args:
            thought: 当前思考
            
        Returns:
            是否完成
        """
        finish_keywords = [
            "final answer",
            "the answer is",
            "i have completed",
            "task finished",
            "done"
        ]
        
        thought_lower = thought.lower()
        return any(keyword in thought_lower for keyword in finish_keywords)


__all__ = [
    "CoreAgent",
    "MultiTurnAgent", 
    "ReActAgent",
    "AgentState",
    "ErrorType",
    "ExecutionResult",
    "AgentConfig"
]
