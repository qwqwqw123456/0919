"""
多线程/多进程池工具模块

提供完整的并行计算功能，包括：
- 线程池执行器
- 进程池执行器
- 任务队列管理
- 异步任务处理
- 结果收集和错误处理
- 进度跟踪
- 资源限制
"""

import os
import sys
import time
import queue
import threading
import multiprocessing as mp
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    Future,
    as_completed,
    wait,
    FIRST_COMPLETED
)
from typing import Callable, Any, List, Dict, Optional, Union, Tuple, Iterator, Generic, TypeVar
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
from abc import ABC, abstractmethod
import logging


T = TypeVar('T')
R = TypeVar('R')


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class TaskResult(Generic[T]):
    """任务结果数据类"""
    task_id: Any
    status: TaskStatus
    result: Optional[T] = None
    error: Optional[Exception] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_time(self) -> Optional[float]:
        """任务执行耗时"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def is_success(self) -> bool:
        """是否成功完成"""
        return self.status == TaskStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        """是否失败"""
        return self.status == TaskStatus.FAILED


@dataclass
class PoolConfig:
    """线程池/进程池配置"""
    max_workers: Optional[int] = None
    thread_name_prefix: str = "PoolWorker"
    queue_size: Optional[int] = None
    timeout: Optional[float] = None
    return_exceptions: bool = False


class BaseExecutor(ABC):
    """执行器抽象基类"""

    def __init__(self, config: Optional[PoolConfig] = None):
        self.config = config or PoolConfig()
        self._executor = None
        self._tasks: Dict[str, Future] = {}
        self._lock = threading.Lock()

    @abstractmethod
    def submit(self, func: Callable, *args, **kwargs) -> str:
        """提交任务"""
        pass

    @abstractmethod
    def map(self, func: Callable, *iterables, timeout=None, chunksize=1) -> List[Any]:
        """批量映射"""
        pass

    def shutdown(self, wait: bool = True) -> None:
        """
        关闭执行器

        Args:
            wait: 是否等待所有任务完成
        """
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)
        return False


class ThreadPoolExecutorHelper(BaseExecutor):
    """
    线程池执行器助手类

    提供增强的线程池功能
    """

    def __init__(self, config: Optional[PoolConfig] = None):
        super().__init__(config)
        max_workers = self.config.max_workers or (os.cpu_count() or 1) * 2
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=self.config.thread_name_prefix
        )
        self._task_counter = 0
        self._task_results: Dict[str, TaskResult] = {}

    def submit(self, func: Callable, *args, task_id: Optional[str] = None, **kwargs) -> str:
        """
        提交任务到线程池

        Args:
            func: 要执行的函数
            *args: 位置参数
            task_id: 任务 ID，None 表示自动生成
            **kwargs: 关键字参数

        Returns:
            任务 ID
        """
        if task_id is None:
            with self._lock:
                self._task_counter += 1
                task_id = f"task_{self._task_counter}"

        def wrapped_func():
            task_result = self._task_results.get(task_id)
            if task_result:
                task_result.status = TaskStatus.RUNNING
                task_result.start_time = time.time()

            try:
                result = func(*args, **kwargs)
                status = TaskStatus.COMPLETED
                error = None
            except Exception as e:
                result = None
                status = TaskStatus.FAILED
                error = e

            end_time = time.time()

            task_result = TaskResult(
                task_id=task_id,
                status=status,
                result=result,
                error=error,
                start_time=task_result.start_time if task_result else None,
                end_time=end_time
            )
            self._task_results[task_id] = task_result

            if error and not self.config.return_exceptions:
                raise error

            return result

        future = self._executor.submit(wrapped_func)
        self._tasks[task_id] = future

        self._task_results[task_id] = TaskResult(
            task_id=task_id,
            status=TaskStatus.PENDING
        )

        return task_id

    def submit_with_callback(
        self,
        func: Callable,
        callback: Callable,
        error_callback: Optional[Callable] = None,
        *args,
        **kwargs
    ) -> str:
        """
        提交带回调的任务

        Args:
            func: 要执行的函数
            callback: 成功回调
            error_callback: 错误回调
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            任务 ID
        """
        task_id = self.submit(func, *args, **kwargs)

        def handle_result(future: Future):
            try:
                result = future.result()
                if callback:
                    callback(result)
            except Exception as e:
                if error_callback:
                    error_callback(e)

        future = self._tasks[task_id]
        future.add_done_callback(handle_result)

        return task_id

    def map(self, func: Callable, *iterables, timeout=None, chunksize=1) -> List[Any]:
        """
        批量映射执行

        Args:
            func: 要执行的函数
            *iterables: 可迭代对象
            timeout: 超时时间
            chunksize: 分块大小

        Returns:
            结果列表
        """
        return list(self._executor.map(func, *iterables, timeout=timeout, chunksize=chunksize))

    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        获取任务结果

        Args:
            task_id: 任务 ID
            timeout: 超时时间

        Returns:
            任务结果

        Raises:
            KeyError: 任务不存在
            TimeoutError: 获取超时
        """
        if task_id not in self._tasks:
            raise KeyError(f"任务不存在: {task_id}")

        future = self._tasks[task_id]
        return future.result(timeout=timeout)

    def get_task_status(self, task_id: str) -> TaskStatus:
        """
        获取任务状态

        Args:
            task_id: 任务 ID

        Returns:
            任务状态
        """
        if task_id not in self._task_results:
            return TaskStatus.PENDING

        return self._task_results[task_id].status

    def get_task_result_info(self, task_id: str) -> Optional[TaskResult]:
        """
        获取任务结果信息

        Args:
            task_id: 任务 ID

        Returns:
            TaskResult 对象
        """
        return self._task_results.get(task_id)

    def cancel(self, task_id: str) -> bool:
        """
        取消任务

        Args:
            task_id: 任务 ID

        Returns:
            是否成功取消
        """
        if task_id not in self._tasks:
            return False

        future = self._tasks[task_id]
        cancelled = future.cancel()

        if cancelled:
            self._task_results[task_id].status = TaskStatus.CANCELLED

        return cancelled

    def wait_all(self, timeout: Optional[float] = None) -> List[TaskResult]:
        """
        等待所有任务完成

        Args:
            timeout: 超时时间

        Returns:
            任务结果列表
        """
        futures = list(self._tasks.values())

        if timeout:
            done, _ = wait(futures, timeout=timeout)
        else:
            done = futures

        results = []
        for task_id, future in self._tasks.items():
            if future in done or future.done():
                result = self._task_results[task_id]
                try:
                    future.result()
                    result.status = TaskStatus.COMPLETED
                except Exception as e:
                    result.status = TaskStatus.FAILED
                    result.error = e
                results.append(result)

        return results


class ProcessPoolExecutorHelper(BaseExecutor):
    """
    进程池执行器助手类

    提供增强的进程池功能，适用于 CPU 密集型任务
    """

    def __init__(self, config: Optional[PoolConfig] = None):
        super().__init__(config)
        max_workers = self.config.max_workers or os.cpu_count() or 1
        self._executor = ProcessPoolExecutor(max_workers=max_workers)
        self._task_counter = 0
        self._task_results: Dict[str, TaskResult] = {}
        self._manager = mp.Manager()
        self._result_queue = self._manager.Queue()

    def submit(self, func: Callable, *args, task_id: Optional[str] = None, **kwargs) -> str:
        """
        提交任务到进程池

        Args:
            func: 要执行的函数
            *args: 位置参数
            task_id: 任务 ID
            **kwargs: 关键字参数

        Returns:
            任务 ID
        """
        if task_id is None:
            self._task_counter += 1
            task_id = f"process_task_{self._task_counter}"

        self._task_results[task_id] = TaskResult(
            task_id=task_id,
            status=TaskStatus.PENDING
        )

        future = self._executor.submit(func, *args, **kwargs)
        self._tasks[task_id] = future

        def handle_result(future: Future):
            result_info = self._task_results[task_id]
            result_info.start_time = time.time()
            result_info.status = TaskStatus.RUNNING

            try:
                result = future.result()
                result_info.status = TaskStatus.COMPLETED
                result_info.result = result
            except Exception as e:
                result_info.status = TaskStatus.FAILED
                result_info.error = e

            result_info.end_time = time.time()

        future.add_done_callback(handle_result)

        return task_id

    def map(self, func: Callable, *iterables, timeout=None, chunksize=1) -> List[Any]:
        """
        批量映射执行

        Args:
            func: 要执行的函数
            *iterables: 可迭代对象
            timeout: 超时时间
            chunksize: 分块大小

        Returns:
            结果列表
        """
        return list(self._executor.map(func, *iterables, timeout=timeout, chunksize=chunksize))

    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """获取任务结果"""
        if task_id not in self._tasks:
            raise KeyError(f"任务不存在: {task_id}")
        return self._tasks[task_id].result(timeout=timeout)

    def get_task_status(self, task_id: str) -> TaskStatus:
        """获取任务状态"""
        return self._task_results.get(task_id, TaskResult(task_id=task_id, status=TaskStatus.PENDING)).status

    def shutdown(self, wait: bool = True) -> None:
        """关闭进程池"""
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None


class TaskQueue:
    """
    任务队列类

    提供线程安全的任务队列管理
    """

    def __init__(self, maxsize: int = 0):
        """
        初始化任务队列

        Args:
            maxsize: 队列最大容量，0 表示无限制
        """
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._results: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    def put(self, task_id: str, task_data: Any, block: bool = True, timeout: Optional[float] = None) -> None:
        """
        添加任务到队列

        Args:
            task_id: 任务 ID
            task_data: 任务数据
            block: 是否阻塞
            timeout: 超时时间
        """
        self._queue.put((task_id, task_data), block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Tuple[str, Any]:
        """
        从队列获取任务

        Args:
            block: 是否阻塞
            timeout: 超时时间

        Returns:
            (任务 ID, 任务数据)
        """
        task_id, task_data = self._queue.get(block=block, timeout=timeout)
        return task_id, task_data

    def put_result(self, task_id: str, result: Any) -> None:
        """
        存储任务结果

        Args:
            task_id: 任务 ID
            result: 任务结果
        """
        with self._lock:
            self._results[task_id] = result

    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        获取任务结果

        Args:
            task_id: 任务 ID
            timeout: 超时时间

        Returns:
            任务结果
        """
        start_time = time.time()

        while True:
            with self._lock:
                if task_id in self._results:
                    return self._results.pop(task_id)

            if timeout:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(f"获取结果超时: {task_id}")
                remaining = timeout - elapsed
            else:
                remaining = 1.0

            time.sleep(0.1)

    def task_done(self) -> None:
        """标记任务完成"""
        self._queue.task_done()

    def join(self) -> None:
        """等待队列清空"""
        self._queue.join()

    def empty(self) -> bool:
        """队列是否为空"""
        return self._queue.empty()

    def size(self) -> int:
        """队列大小"""
        return self._queue.qsize()

    def clear(self) -> None:
        """清空队列"""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

        with self._lock:
            self._results.clear()

    def stop(self) -> None:
        """停止队列"""
        self._stop_event.set()

    @property
    def is_stopped(self) -> bool:
        """是否已停止"""
        return self._stop_event.is_set()


class WorkerPool:
    """
    工作线程池类

    管理多个工作线程处理队列中的任务
    """

    def __init__(
        self,
        num_workers: int = 4,
        queue: Optional[TaskQueue] = None,
        worker_func: Optional[Callable] = None
    ):
        """
        初始化工作线程池

        Args:
            num_workers: 工作线程数
            queue: 任务队列，None 表示创建新队列
            worker_func: 工作函数
        """
        self.num_workers = num_workers
        self.queue = queue or TaskQueue()
        self.worker_func = worker_func
        self._workers: List[threading.Thread] = []
        self._stop_event = threading.Event()
        self._started = False

    def start(self) -> None:
        """启动工作线程"""
        if self._started:
            return

        self._stop_event.clear()

        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"Worker-{i}",
                daemon=True
            )
            worker.start()
            self._workers.append(worker)

        self._started = True

    def _worker_loop(self) -> None:
        """工作线程主循环"""
        while not self._stop_event.is_set():
            try:
                task_id, task_data = self.queue.get(block=True, timeout=1.0)

                try:
                    if self.worker_func:
                        result = self.worker_func(task_data)
                    else:
                        result = task_data

                    self.queue.put_result(task_id, result)

                except Exception as e:
                    self.queue.put_result(task_id, e)

                finally:
                    self.queue.task_done()

            except queue.Empty:
                continue

    def stop(self, wait: bool = True) -> None:
        """
        停止工作线程

        Args:
            wait: 是否等待所有任务完成
        """
        self._stop_event.set()

        if wait:
            self.queue.join()

        for worker in self._workers:
            worker.join(timeout=5.0)

        self._workers.clear()
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop(wait=True)
        return False


def parallel_map(
    func: Callable,
    items: List[Any],
    max_workers: int = 4,
    use_process: bool = False,
    timeout: Optional[float] = None
) -> List[Any]:
    """
    并行映射执行

    Args:
        func: 要执行的函数
        items: 输入项列表
        max_workers: 最大工作数
        use_process: 是否使用进程池
        timeout: 超时时间

    Returns:
        结果列表

    Example:
        >>> results = parallel_map(lambda x: x * 2, [1, 2, 3, 4], max_workers=4)
        >>> print(results)
        [2, 4, 6, 8]
    """
    if use_process:
        executor_class = ProcessPoolExecutorHelper
    else:
        executor_class = ThreadPoolExecutorHelper

    with executor_class(PoolConfig(max_workers=max_workers)) as executor:
        task_ids = [executor.submit(func, item) for item in items]

        results = []
        for task_id in task_ids:
            try:
                result = executor.get_result(task_id, timeout=timeout)
                results.append(result)
            except Exception as e:
                results.append(e)

        return results


def parallel_execute(
    funcs: List[Callable],
    max_workers: int = 4,
    use_process: bool = False
) -> List[Any]:
    """
    并行执行多个函数

    Args:
        funcs: 函数列表
        max_workers: 最大工作数
        use_process: 是否使用进程池

    Returns:
        结果列表
    """
    return parallel_map(lambda f: f(), funcs, max_workers=max_workers, use_process=use_process)


def batch_process(
    items: List[Any],
    batch_size: int = 32,
    process_func: Optional[Callable] = None,
    reduce_func: Optional[Callable] = None
) -> List[Any]:
    """
    分批处理

    Args:
        items: 输入项列表
        batch_size: 批次大小
        process_func: 批处理函数，None 表示直接返回
        reduce_func: 合并函数，None 表示合并结果

    Returns:
        处理结果
    """
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    if process_func is None:
        return batches

    results = []
    for batch in batches:
        batch_result = process_func(batch)
        results.append(batch_result)

    if reduce_func:
        return reduce_func(results)

    return results


class ProgressTracker:
    """
    进度跟踪器

    跟踪任务执行进度
    """

    def __init__(self, total: int, desc: str = "Progress"):
        """
        初始化进度跟踪器

        Args:
            total: 总任务数
            desc: 描述信息
        """
        self.total = total
        self.desc = desc
        self.current = 0
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._callbacks: List[Callable] = []

    def update(self, n: int = 1) -> None:
        """
        更新进度

        Args:
            n: 增加的数量
        """
        with self._lock:
            self.current = min(self.current + n, self.total)
            self._notify()

    def set(self, value: int) -> None:
        """
        设置进度

        Args:
            value: 进度值
        """
        with self._lock:
            self.current = min(value, self.total)
            self._notify()

    def reset(self) -> None:
        """重置进度"""
        with self._lock:
            self.current = 0
            self._start_time = time.time()

    @property
    def progress(self) -> float:
        """进度百分比"""
        if self.total == 0:
            return 0.0
        return self.current / self.total

    @property
    def elapsed_time(self) -> float:
        """已用时间"""
        return time.time() - self._start_time

    @property
    def estimated_remaining(self) -> Optional[float]:
        """预计剩余时间"""
        if self.current == 0:
            return None
        elapsed = self.elapsed_time
        rate = self.current / elapsed
        remaining = self.total - self.current
        return remaining / rate

    def add_callback(self, callback: Callable) -> None:
        """
        添加回调函数

        Args:
            callback: 回调函数，签名为 callback(progress: float)
        """
        self._callbacks.append(callback)

    def _notify(self) -> None:
        """通知回调"""
        p = self.progress
        for callback in self._callbacks:
            try:
                callback(p)
            except Exception:
                pass

    def __str__(self) -> str:
        """字符串表示"""
        percentage = int(self.progress * 100)
        return f"{self.desc}: {percentage}%"


@contextmanager
def thread_pool_context(max_workers: int = 4):
    """
    线程池上下文管理器

    Args:
        max_workers: 最大工作数

    Example:
        with thread_pool_context(max_workers=8) as executor:
            future = executor.submit(heavy_function, arg)
            result = future.result()
    """
    executor = ThreadPoolExecutorHelper(PoolConfig(max_workers=max_workers))
    try:
        yield executor
    finally:
        executor.shutdown(wait=True)


@contextmanager
def process_pool_context(max_workers: Optional[int] = None):
    """
    进程池上下文管理器

    Args:
        max_workers: 最大工作数

    Example:
        with process_pool_context() as executor:
            future = executor.submit(cpu_intensive_function, arg)
            result = future.result()
    """
    executor = ProcessPoolExecutorHelper(PoolConfig(max_workers=max_workers))
    try:
        yield executor
    finally:
        executor.shutdown(wait=True)


class AsyncBatchExecutor:
    """
    异步批量执行器

    支持异步提交和批量获取结果
    """

    def __init__(
        self,
        max_workers: int = 4,
        use_process: bool = False,
        queue_size: int = 1000
    ):
        """
        初始化异步批量执行器

        Args:
            max_workers: 最大工作数
            use_process: 是否使用进程池
            queue_size: 队列大小
        """
        self.max_workers = max_workers
        self.use_process = use_process

        if use_process:
            self._executor = ProcessPoolExecutorHelper(PoolConfig(max_workers=max_workers))
        else:
            self._executor = ThreadPoolExecutorHelper(PoolConfig(max_workers=max_workers))

        self._pending: Dict[str, Any] = {}
        self._completed: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def submit(self, func: Callable, *args, **kwargs) -> str:
        """
        异步提交任务

        Args:
            func: 要执行的函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            任务 ID
        """
        task_id = self._executor.submit(func, *args, **kwargs)

        with self._lock:
            self._pending[task_id] = True

        return task_id

    def poll(self) -> Dict[str, Any]:
        """
        轮询已完成的的任务

        Returns:
            已完成的任务字典 {task_id: result}
        """
        completed = {}

        with self._lock:
            pending_ids = list(self._pending.keys())

        for task_id in pending_ids:
            status = self._executor.get_task_status(task_id)

            if status == TaskStatus.COMPLETED:
                try:
                    result = self._executor.get_result(task_id)
                    completed[task_id] = result

                    with self._lock:
                        self._pending.pop(task_id, None)
                        self._completed[task_id] = result

                except Exception as e:
                    completed[task_id] = e

                    with self._lock:
                        self._pending.pop(task_id, None)
                        self._completed[task_id] = e

            elif status in (TaskStatus.FAILED, TaskStatus.CANCELLED):
                with self._lock:
                    self._pending.pop(task_id, None)

        return completed

    def wait(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        等待所有任务完成

        Args:
            timeout: 超时时间

        Returns:
            所有任务结果
        """
        start_time = time.time()

        while True:
            results = self.poll()

            with self._lock:
                if not self._pending:
                    return self._completed

            if timeout:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    break

            time.sleep(0.1)

        return self._completed

    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        获取特定任务结果

        Args:
            task_id: 任务 ID
            timeout: 超时时间

        Returns:
            任务结果
        """
        with self._lock:
            if task_id in self._completed:
                return self._completed[task_id]

        status = self._executor.get_task_status(task_id)

        if status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            result = self._executor.get_result(task_id, timeout=timeout)

            with self._lock:
                self._completed[task_id] = result

            return result

        if timeout:
            start = time.time()
            while time.time() - start < timeout:
                time.sleep(0.1)
                status = self._executor.get_task_status(task_id)
                if status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    return self._executor.get_result(task_id)

        raise TimeoutError(f"获取结果超时: {task_id}")

    def shutdown(self, wait: bool = True) -> None:
        """关闭执行器"""
        self._executor.shutdown(wait=wait)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False


executor = ThreadPoolExecutor(max_workers=4)
