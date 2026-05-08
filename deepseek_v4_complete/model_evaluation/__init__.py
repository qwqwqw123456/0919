"""
模型评估模块

提供全面的语言模型评估功能，包括：
- 基础评估指标计算 (PPL、BLEU、ROUGE等)
- 通用基准测试 (MMLU、GSM8K、HumanEval等)
- 推理能力评估 (逻辑推理、数学推理等)
- 对话质量评估 (相关性、连贯性等)
- 安全性评估 (毒性检测、偏见检测等)
- 评估报告生成 (Markdown、JSON、HTML格式)
"""

from .basic_metric_calc import BasicMetricCalc
from .common_bench_test import (
    CommonBenchTest,
    MMLUBenchmark,
    GSM8KBenchmark,
    HumanEvalBenchmark,
    HellaSwagBenchmark,
    TruthfulQABenchmark,
    BenchmarkResult
)
from .reasoning_eval import (
    ReasoningEval,
    ReasoningDataset,
    ReasoningStepExtractor,
    ReasoningResult,
    ReasoningEvalResult
)
from .chat_quality_eval import (
    ChatQualityEval,
    Conversation,
    Message,
    ConversationEvalResult
)
from .safety_eval import (
    SafetyEval,
    SafetyTestCase,
    SafetyEvalResult,
    SafetyReport,
    ToxicityDetector,
    SensitiveContentDetector,
    MaliciousInstructionDetector,
    BiasDetector,
    PrivacyProtector
)
from .report_generator import (
    ReportGenerator,
    EvaluationReport,
    ReportMetadata,
    BenchmarkSection,
    MarkdownReportGenerator,
    JSONReportGenerator,
    HTMLReportGenerator
)

__all__ = [
    'BasicMetricCalc',
    'CommonBenchTest',
    'MMLUBenchmark',
    'GSM8KBenchmark',
    'HumanEvalBenchmark',
    'HellaSwagBenchmark',
    'TruthfulQABenchmark',
    'BenchmarkResult',
    'ReasoningEval',
    'ReasoningDataset',
    'ReasoningStepExtractor',
    'ReasoningResult',
    'ReasoningEvalResult',
    'ChatQualityEval',
    'Conversation',
    'Message',
    'ConversationEvalResult',
    'SafetyEval',
    'SafetyTestCase',
    'SafetyEvalResult',
    'SafetyReport',
    'ToxicityDetector',
    'SensitiveContentDetector',
    'MaliciousInstructionDetector',
    'BiasDetector',
    'PrivacyProtector',
    'ReportGenerator',
    'EvaluationReport',
    'ReportMetadata',
    'BenchmarkSection',
    'MarkdownReportGenerator',
    'JSONReportGenerator',
    'HTMLReportGenerator'
]

__version__ = '1.0.0'
