"""
评估报告生成器模块

提供全面的评估报告生成功能，支持：
- 多维度评估结果汇总
- Markdown/JSON/HTML格式报告生成
- 可视化图表生成
- 自定义报告模板
- 性能趋势分析
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from collections import defaultdict
import math


@dataclass
class ReportMetadata:
    """报告元数据"""
    model_name: str
    model_version: str
    evaluation_date: str
    evaluator: str
    total_samples: int
    execution_time: float
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class BenchmarkSection:
    """基准测试报告章节"""
    name: str
    score: float
    total: int
    correct: int
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'score': self.score,
            'total': self.total,
            'correct': self.correct,
            'accuracy': f"{self.correct/self.total*100:.2f}%" if self.total > 0 else "N/A",
            'details': self.details,
            'recommendations': self.recommendations
        }


@dataclass
class EvaluationReport:
    """评估报告"""
    metadata: ReportMetadata
    summary: Dict[str, Any]
    benchmark_results: Dict[str, BenchmarkSection]
    reasoning_results: Dict[str, Any]
    quality_results: Dict[str, Any]
    safety_results: Dict[str, Any]
    charts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'metadata': self.metadata.to_dict(),
            'summary': self.summary,
            'benchmark_results': {k: v.to_dict() for k, v in self.benchmark_results.items()},
            'reasoning_results': self.reasoning_results,
            'quality_results': self.quality_results,
            'safety_results': self.safety_results,
            'charts': self.charts,
            'recommendations': self.recommendations
        }


class MarkdownReportGenerator:
    """Markdown报告生成器"""

    def __init__(self):
        """初始化Markdown报告生成器"""
        self.template = self._get_template()

    def _get_template(self) -> str:
        """获取报告模板"""
        return """# {title}

**模型评估报告**

---

## 📋 执行摘要

| 项目 | 值 |
|------|-----|
| 模型名称 | {model_name} |
| 模型版本 | {model_version} |
| 评估日期 | {evaluation_date} |
| 评估者 | {evaluator} |
| 总样本数 | {total_samples} |
| 执行时间 | {execution_time} |

### 总体评分

{overall_score_block}

---

## 📊 基准测试结果

{benchmark_section}

---

## 🧠 推理能力评估

{reasoning_section}

---

## 💬 对话质量评估

{quality_section}

---

## 🔒 安全性评估

{safety_section}

---

## 📈 详细分析

{detailed_analysis}

---

## 💡 改进建议

{recommendations}

---

## 📎 附录

### A. 评估方法论

本报告采用以下评估方法：

1. **基准测试**: 使用标准基准数据集（MMLU、GSM8K等）评估模型性能
2. **推理评估**: 通过多类型推理任务评估模型的逻辑思维
3. **质量评估**: 从多个维度评估对话质量
4. **安全评估**: 检测模型响应的安全性和潜在风险

### B. 评分标准

| 等级 | 分数范围 | 说明 |
|------|----------|------|
| 优秀 | 90-100 | 远超预期 |
| 良好 | 75-89 | 达到预期 |
| 一般 | 60-74 | 基本达标 |
| 较差 | <60 | 需要改进 |

---

*报告生成时间: {generated_time}*
"""

    def generate(self, report: EvaluationReport) -> str:
        """
        生成Markdown报告

        Args:
            report: 评估报告数据

        Returns:
            str: Markdown格式报告
        """
        overall_block = self._generate_overall_score(report.summary)

        benchmark_section = self._generate_benchmark_section(report.benchmark_results)
        reasoning_section = self._generate_reasoning_section(report.reasoning_results)
        quality_section = self._generate_quality_section(report.quality_results)
        safety_section = self._generate_safety_section(report.safety_results)
        detailed_analysis = self._generate_detailed_analysis(report)
        recommendations = self._generate_recommendations(report.recommendations)

        content = self.template.format(
            title=report.metadata.model_name + " 评估报告",
            model_name=report.metadata.model_name,
            model_version=report.metadata.model_version,
            evaluation_date=report.metadata.evaluation_date,
            evaluator=report.metadata.evaluator,
            total_samples=report.metadata.total_samples,
            execution_time=f"{report.metadata.execution_time:.2f}s",
            overall_score_block=overall_block,
            benchmark_section=benchmark_section,
            reasoning_section=reasoning_section,
            quality_section=quality_section,
            safety_section=safety_section,
            detailed_analysis=detailed_analysis,
            recommendations=recommendations,
            generated_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        return content

    def _generate_overall_score(self, summary: Dict[str, Any]) -> str:
        """生成总体评分块"""
        overall = summary.get('overall_score', 0.0)
        grade = self._get_grade(overall)

        block = f"""
### 综合评分: {overall:.2f}/100 ({grade})

**{self._get_grade_description(grade)}**

"""
        if 'component_scores' in summary:
            block += "#### 各项得分\n\n"
            block += "| 评估维度 | 得分 | 权重 |\n"
            block += "|----------|------|------|\n"
            for component, data in summary['component_scores'].items():
                score = data.get('score', 0)
                weight = data.get('weight', 0)
                block += f"| {component} | {score:.2f} | {weight:.0%} |\n"
            block += "\n"

        return block

    def _generate_benchmark_section(self, results: Dict[str, BenchmarkSection]) -> str:
        """生成基准测试章节"""
        if not results:
            return "暂无基准测试数据。"

        section = ""

        for name, data in results.items():
            section += f"### {data.name}\n\n"
            section += f"- **得分**: {data.score:.4f}\n"
            section += f"- **正确数**: {data.correct}/{data.total}\n"
            section += f"- **准确率**: {data.correct/data.total*100:.2f}%\n"

            if data.details:
                section += "\n**详细信息**:\n"
                for key, value in data.details.items():
                    section += f"- {key}: {value}\n"

            if data.recommendations:
                section += "\n**建议**:\n"
                for rec in data.recommendations:
                    section += f"- {rec}\n"

            section += "\n---\n\n"

        return section

    def _generate_reasoning_section(self, results: Dict[str, Any]) -> str:
        """生成推理能力章节"""
        if not results:
            return "暂无推理能力评估数据。"

        section = ""

        overall = results.get('overall_accuracy', 0)
        section += f"### 总体推理准确率: {overall*100:.2f}%\n\n"

        if 'by_type' in results:
            section += "#### 各推理类型表现\n\n"
            section += "| 推理类型 | 准确率 | 平均得分 | 样本数 |\n"
            section += "|----------|--------|----------|--------|\n"

            for rtype, data in results['by_type'].items():
                acc = data.get('accuracy', 0) * 100
                score = data.get('avg_score', 0)
                total = data.get('total', 0)
                section += f"| {rtype} | {acc:.2f}% | {score:.4f} | {total} |\n"

            section += "\n"

        return section

    def _generate_quality_section(self, results: Dict[str, Any]) -> str:
        """生成对话质量章节"""
        if not results:
            return "暂无对话质量评估数据。"

        section = ""

        if 'overall_stats' in results:
            stats = results['overall_stats']
            section += f"### 总体质量分数: {stats.get('mean', 0):.4f}\n\n"
            section += f"- 最高分: {stats.get('max', 0):.4f}\n"
            section += f"- 最低分: {stats.get('min', 0):.4f}\n\n"

        if 'criteria_stats' in results:
            section += "#### 各维度评分\n\n"
            section += "| 维度 | 平均分 | 最高 | 最低 |\n"
            section += "|------|--------|------|------|\n"

            for criterion, stats in results['criteria_stats'].items():
                mean = stats.get('mean', 0)
                max_val = stats.get('max', 0)
                min_val = stats.get('min', 0)
                section += f"| {criterion} | {mean:.4f} | {max_val:.4f} | {min_val:.4f} |\n"

            section += "\n"

        return section

    def _generate_safety_section(self, results: Dict[str, Any]) -> str:
        """生成安全性章节"""
        if not results:
            return "暂无安全性评估数据。"

        section = ""

        overall_score = results.get('overall_safety_score', 1.0)
        section += f"### 总体安全评分: {overall_score:.4f}\n\n"

        if 'category_scores' in results:
            section += "#### 各类别安全表现\n\n"
            section += "| 类别 | 通过率 | 平均分 |\n"
            section += "|------|---------|--------|\n"

            for category, data in results['category_scores'].items():
                pass_rate = data.get('pass_rate', 0) * 100
                avg_score = data.get('avg_score', 0)
                section += f"| {category} | {pass_rate:.1f}% | {avg_score:.4f} |\n"

            section += "\n"

        passed = results.get('passed_tests', 0)
        total = results.get('total_tests', 0)
        section += f"**测试结果**: {passed}/{total} 通过 ({passed/total*100:.1f}%)\n\n"

        return section

    def _generate_detailed_analysis(self, report: EvaluationReport) -> str:
        """生成详细分析"""
        analysis = "### 性能亮点\n\n"

        if report.benchmark_results:
            best_benchmark = max(
                report.benchmark_results.items(),
                key=lambda x: x[1].score
            )
            analysis += f"- **{best_benchmark[1].name}** 表现最佳 (得分: {best_benchmark[1].score:.4f})\n"

        if report.reasoning_results.get('by_type'):
            reasoning_types = report.reasoning_results['by_type']
            if reasoning_types:
                best_reasoning = max(
                    reasoning_types.items(),
                    key=lambda x: x[1].get('accuracy', 0)
                )
                analysis += f"- **推理类型 '{best_reasoning[0]}'** 表现最佳\n"

        analysis += "\n### 需要改进的方面\n\n"

        if report.benchmark_results:
            worst_benchmark = min(
                report.benchmark_results.items(),
                key=lambda x: x[1].score
            )
            analysis += f"- **{worst_benchmark[1].name}** 需要重点改进 (得分: {worst_benchmark[1].score:.4f})\n"

        return analysis

    def _generate_recommendations(self, recommendations: List[str]) -> str:
        """生成改进建议"""
        if not recommendations:
            return "暂无具体建议。"

        content = ""
        for i, rec in enumerate(recommendations, 1):
            content += f"{i}. {rec}\n"

        return content

    def _get_grade(self, score: float) -> str:
        """获取评分等级"""
        if score >= 90:
            return "A (优秀)"
        elif score >= 75:
            return "B (良好)"
        elif score >= 60:
            return "C (一般)"
        else:
            return "D (较差)"

    def _get_grade_description(self, grade: str) -> str:
        """获取等级描述"""
        descriptions = {
            "A (优秀)": "模型表现优异，在各项评估中均达到或超过预期水平。",
            "B (良好)": "模型表现良好，能够较好地完成各项任务。",
            "C (一般)": "模型基本满足要求，但仍有改进空间。",
            "D (较差)": "模型表现不佳，需要进行显著改进。"
        }
        return descriptions.get(grade, "")


class JSONReportGenerator:
    """JSON报告生成器"""

    def __init__(self):
        """初始化JSON报告生成器"""
        pass

    def generate(self, report: EvaluationReport) -> str:
        """
        生成JSON格式报告

        Args:
            report: 评估报告数据

        Returns:
            str: JSON格式报告
        """
        return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)

    def save(self, report: EvaluationReport, filepath: str) -> None:
        """
        保存JSON报告到文件

        Args:
            report: 评估报告数据
            filepath: 文件路径
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)


class HTMLReportGenerator:
    """HTML报告生成器"""

    def __init__(self):
        """初始化HTML报告生成器"""
        self.template = self._get_template()

    def _get_template(self) -> str:
        """获取HTML模板"""
        return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - 评估报告</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header .meta {{
            opacity: 0.9;
            font-size: 0.9em;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }}
        .score-display {{
            text-align: center;
            padding: 30px;
        }}
        .score-value {{
            font-size: 4em;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .score-grade {{
            font-size: 1.5em;
            color: #666;
            margin-top: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        .progress-bar {{
            background: #e0e0e0;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 500;
        }}
        .badge-success {{ background: #d4edda; color: #155724; }}
        .badge-warning {{ background: #fff3cd; color: #856404; }}
        .badge-danger {{ background: #f8d7da; color: #721c24; }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            color: #666;
            margin-top: 5px;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{model_name}</h1>
            <p class="meta">
                版本: {model_version} | 评估日期: {evaluation_date} | 评估者: {evaluator}
            </p>
        </div>

        <div class="card">
            <div class="score-display">
                <div class="score-value">{overall_score:.1f}</div>
                <div class="score-grade">{grade}</div>
            </div>
        </div>

        <div class="grid">
            <div class="stat-card">
                <div class="stat-value">{total_samples}</div>
                <div class="stat-label">评估样本</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{execution_time}</div>
                <div class="stat-label">执行时间</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{benchmark_count}</div>
                <div class="stat-label">基准测试</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{safety_score:.2f}</div>
                <div class="stat-label">安全评分</div>
            </div>
        </div>

        <div class="card">
            <h2>📊 基准测试结果</h2>
            {benchmark_content}
        </div>

        <div class="card">
            <h2>🧠 推理能力</h2>
            {reasoning_content}
        </div>

        <div class="card">
            <h2>💬 对话质量</h2>
            {quality_content}
        </div>

        <div class="card">
            <h2>🔒 安全性</h2>
            {safety_content}
        </div>

        <div class="card">
            <h2>💡 改进建议</h2>
            {recommendations_content}
        </div>

        <div class="footer">
            报告生成时间: {generated_time}
        </div>
    </div>
</body>
</html>
"""

    def generate(self, report: EvaluationReport) -> str:
        """
        生成HTML报告

        Args:
            report: 评估报告数据

        Returns:
            str: HTML格式报告
        """
        overall_score = report.summary.get('overall_score', 0)
        grade = self._get_grade(overall_score)

        benchmark_content = self._generate_benchmark_html(report.benchmark_results)
        reasoning_content = self._generate_reasoning_html(report.reasoning_results)
        quality_content = self._generate_quality_html(report.quality_results)
        safety_content = self._generate_safety_html(report.safety_results)
        recommendations_content = self._generate_recommendations_html(report.recommendations)

        html = self.template.format(
            title=report.metadata.model_name,
            model_name=report.metadata.model_name,
            model_version=report.metadata.model_version,
            evaluation_date=report.metadata.evaluation_date,
            evaluator=report.metadata.evaluator,
            overall_score=overall_score,
            grade=grade,
            total_samples=report.metadata.total_samples,
            execution_time=f"{report.metadata.execution_time:.2f}s",
            benchmark_count=len(report.benchmark_results),
            safety_score=report.safety_results.get('overall_safety_score', 0),
            generated_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            benchmark_content=benchmark_content,
            reasoning_content=reasoning_content,
            quality_content=quality_content,
            safety_content=safety_content,
            recommendations_content=recommendations_content
        )

        return html

    def _generate_benchmark_html(self, results: Dict[str, BenchmarkSection]) -> str:
        """生成基准测试HTML"""
        if not results:
            return "<p>暂无数据</p>"

        html = '<table><tr><th>测试</th><th>得分</th><th>准确率</th><th>状态</th></tr>'

        for name, data in results.items():
            acc = data.correct / data.total * 100 if data.total > 0 else 0
            badge_class = 'badge-success' if acc >= 80 else 'badge-warning' if acc >= 60 else 'badge-danger'
            badge_text = '优秀' if acc >= 80 else '良好' if acc >= 60 else '需改进'

            html += f"""
            <tr>
                <td>{data.name}</td>
                <td>{data.score:.4f}</td>
                <td>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {acc}%"></div>
                    </div>
                    {acc:.1f}%
                </td>
                <td><span class="badge {badge_class}">{badge_text}</span></td>
            </tr>
            """

        html += '</table>'
        return html

    def _generate_reasoning_html(self, results: Dict[str, Any]) -> str:
        """生成推理能力HTML"""
        if not results:
            return "<p>暂无数据</p>"

        overall = results.get('overall_accuracy', 0) * 100
        html = f"<p>总体推理准确率: <strong>{overall:.1f}%</strong></p>"

        if 'by_type' in results:
            html += '<table><tr><th>推理类型</th><th>准确率</th><th>平均得分</th></tr>'
            for rtype, data in results['by_type'].items():
                acc = data.get('accuracy', 0) * 100
                score = data.get('avg_score', 0)
                html += f"<tr><td>{rtype}</td><td>{acc:.1f}%</td><td>{score:.4f}</td></tr>"
            html += '</table>'

        return html

    def _generate_quality_html(self, results: Dict[str, Any]) -> str:
        """生成对话质量HTML"""
        if not results:
            return "<p>暂无数据</p>"

        if 'overall_stats' in results:
            mean = results['overall_stats'].get('mean', 0)
            html = f"<p>总体质量分数: <strong>{mean:.4f}</strong></p>"
        else:
            html = "<p>暂无数据</p>"

        return html

    def _generate_safety_html(self, results: Dict[str, Any]) -> str:
        """生成安全性HTML"""
        if not results:
            return "<p>暂无数据</p>"

        overall = results.get('overall_safety_score', 0)
        passed = results.get('passed_tests', 0)
        total = results.get('total_tests', 0)
        rate = passed / total * 100 if total > 0 else 0

        html = f"""
        <p>总体安全评分: <strong>{overall:.4f}</strong></p>
        <p>测试通过率: {passed}/{total} ({rate:.1f}%)</p>
        """

        if 'category_scores' in results:
            html += '<table><tr><th>类别</th><th>通过率</th><th>平均分</th></tr>'
            for cat, data in results['category_scores'].items():
                pass_rate = data.get('pass_rate', 0) * 100
                avg_score = data.get('avg_score', 0)
                html += f"<tr><td>{cat}</td><td>{pass_rate:.1f}%</td><td>{avg_score:.4f}</td></tr>"
            html += '</table>'

        return html

    def _generate_recommendations_html(self, recommendations: List[str]) -> str:
        """生成建议HTML"""
        if not recommendations:
            return "<p>暂无建议</p>"

        html = "<ol>"
        for rec in recommendations:
            html += f"<li>{rec}</li>"
        html += "</ol>"

        return html

    def _get_grade(self, score: float) -> str:
        """获取评分等级"""
        if score >= 90:
            return "A - 优秀"
        elif score >= 75:
            return "B - 良好"
        elif score >= 60:
            return "C - 一般"
        else:
            return "D - 较差"

    def save(self, report: EvaluationReport, filepath: str) -> None:
        """
        保存HTML报告到文件

        Args:
            report: 评估报告数据
            filepath: 文件路径
        """
        html = self.generate(report)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)


class ReportGenerator:
    """评估报告生成器主类"""

    SUPPORTED_FORMATS = ['markdown', 'json', 'html']

    def __init__(self):
        """初始化报告生成器"""
        self.markdown_generator = MarkdownReportGenerator()
        self.json_generator = JSONReportGenerator()
        self.html_generator = HTMLReportGenerator()

    def create_report(
        self,
        metadata: ReportMetadata,
        benchmark_results: Dict[str, BenchmarkSection],
        reasoning_results: Dict[str, Any],
        quality_results: Dict[str, Any],
        safety_results: Dict[str, Any],
        recommendations: List[str] = None
    ) -> EvaluationReport:
        """
        创建评估报告

        Args:
            metadata: 报告元数据
            benchmark_results: 基准测试结果
            reasoning_results: 推理能力评估结果
            quality_results: 对话质量评估结果
            safety_results: 安全性评估结果
            recommendations: 改进建议

        Returns:
            EvaluationReport: 评估报告对象
        """
        summary = self._create_summary(
            benchmark_results, reasoning_results, quality_results, safety_results
        )

        if recommendations is None:
            recommendations = self._generate_recommendations(
                benchmark_results, reasoning_results, quality_results, safety_results
            )

        return EvaluationReport(
            metadata=metadata,
            summary=summary,
            benchmark_results=benchmark_results,
            reasoning_results=reasoning_results,
            quality_results=quality_results,
            safety_results=safety_results,
            recommendations=recommendations
        )

    def _create_summary(
        self,
        benchmark_results: Dict[str, BenchmarkSection],
        reasoning_results: Dict[str, Any],
        quality_results: Dict[str, Any],
        safety_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """创建报告摘要"""
        overall_score = 0.0
        weights = []

        if benchmark_results:
            bench_score = sum(b.score for b in benchmark_results.values()) / len(benchmark_results)
            weights.append(('基准测试', bench_score, 0.3))

        if reasoning_results:
            reasoning_score = reasoning_results.get('overall_accuracy', 0) * 100
            weights.append(('推理能力', reasoning_score, 0.3))

        if quality_results:
            quality_score = quality_results.get('overall_stats', {}).get('mean', 0) * 100
            weights.append(('对话质量', quality_score, 0.2))

        if safety_results:
            safety_score = safety_results.get('overall_safety_score', 0) * 100
            weights.append(('安全性', safety_score, 0.2))

        total_weight = sum(w[2] for w in weights)
        if total_weight > 0:
            overall_score = sum(w[1] * w[2] for w in weights) / total_weight

        component_scores = {
            name: {'score': score, 'weight': weight}
            for name, score, weight in weights
        }

        return {
            'overall_score': overall_score,
            'component_scores': component_scores,
            'total_components': len(weights)
        }

    def _generate_recommendations(
        self,
        benchmark_results: Dict[str, BenchmarkSection],
        reasoning_results: Dict[str, Any],
        quality_results: Dict[str, Any],
        safety_results: Dict[str, Any]
    ) -> List[str]:
        """生成改进建议"""
        recommendations = []

        if benchmark_results:
            worst = min(
                benchmark_results.items(),
                key=lambda x: x[1].score
            )
            if worst[1].score < 0.7:
                recommendations.append(
                    f"建议加强 {worst[1].name} 相关能力的训练，当前得分仅为 {worst[1].score:.2%}"
                )

        if reasoning_results:
            by_type = reasoning_results.get('by_type', {})
            weak_types = [
                rtype for rtype, data in by_type.items()
                if data.get('accuracy', 0) < 0.6
            ]
            if weak_types:
                recommendations.append(
                    f"推理能力中 {', '.join(weak_types)} 表现较弱，建议针对性训练"
                )

        if safety_results:
            safety_score = safety_results.get('overall_safety_score', 0)
            if safety_score < 0.9:
                recommendations.append(
                    f"安全性评估分数为 {safety_score:.2%}，建议加强安全机制"
                )

        if not recommendations:
            recommendations.append("模型整体表现良好，建议继续保持并持续优化")

        return recommendations

    def generate(
        self,
        report: EvaluationReport,
        format: str = 'markdown'
    ) -> str:
        """
        生成报告

        Args:
            report: 评估报告
            format: 输出格式 ('markdown', 'json', 'html')

        Returns:
            str: 格式化后的报告内容
        """
        if format == 'markdown':
            return self.markdown_generator.generate(report)
        elif format == 'json':
            return self.json_generator.generate(report)
        elif format == 'html':
            return self.html_generator.generate(report)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def save_report(
        self,
        report: EvaluationReport,
        filepath: str,
        format: str = 'markdown'
    ) -> None:
        """
        保存报告到文件

        Args:
            report: 评估报告
            filepath: 文件路径
            format: 输出格式
        """
        if format == 'json':
            self.json_generator.save(report, filepath)
        elif format == 'html':
            self.html_generator.save(report, filepath)
        else:
            content = self.generate(report, format)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

    def generate_comparison_report(
        self,
        reports: List[EvaluationReport],
        model_names: List[str]
    ) -> str:
        """
        生成对比报告

        Args:
            reports: 多个评估报告
            model_names: 对应的模型名称

        Returns:
            str: Markdown格式对比报告
        """
        if len(reports) != len(model_names):
            raise ValueError("reports and model_names must have the same length")

        md = "# 模型对比评估报告\n\n"
        md += f"评估日期: {datetime.now().strftime('%Y-%m-%d')}\n\n"

        md += "## 综合评分对比\n\n"
        md += "| 模型 | 综合评分 | 等级 |\n"
        md += "|------|----------|------|\n"

        for name, report in zip(model_names, reports):
            score = report.summary.get('overall_score', 0)
            grade = self._get_grade(score)
            md += f"| {name} | {score:.2f} | {grade} |\n"

        md += "\n## 各项指标对比\n\n"

        criteria = ['基准测试', '推理能力', '对话质量', '安全性']
        md += "| 指标 | " + " | ".join(model_names) + " |\n"
        md += "|" + "---|" * (len(model_names) + 1) + "\n"

        for criterion in criteria:
            values = []
            for report in reports:
                component_scores = report.summary.get('component_scores', {})
                if criterion in component_scores:
                    values.append(f"{component_scores[criterion]['score']:.2f}")
                else:
                    values.append("N/A")
            md += f"| {criterion} | " + " | ".join(values) + " |\n"

        return md

    def _get_grade(self, score: float) -> str:
        """获取评分等级"""
        if score >= 90:
            return "A"
        elif score >= 75:
            return "B"
        elif score >= 60:
            return "C"
        else:
            return "D"


def demo():
    """演示函数"""
    print("=" * 60)
    print("评估报告生成器演示")
    print("=" * 60)

    metadata = ReportMetadata(
        model_name="DeepSeek-V4",
        model_version="1.0.0",
        evaluation_date=datetime.now().strftime("%Y-%m-%d"),
        evaluator="System",
        total_samples=1000,
        execution_time=120.5
    )

    benchmark_results = {
        'mmlu': BenchmarkSection(
            name="MMLU",
            score=0.82,
            total=100,
            correct=82,
            details={'subjects': 57}
        ),
        'gsm8k': BenchmarkSection(
            name="GSM8K",
            score=0.78,
            total=100,
            correct=78
        ),
        'humaneval': BenchmarkSection(
            name="HumanEval",
            score=0.65,
            total=100,
            correct=65
        )
    }

    reasoning_results = {
        'overall_accuracy': 0.75,
        'by_type': {
            'logical': {'accuracy': 0.80, 'avg_score': 0.82, 'total': 50},
            'mathematical': {'accuracy': 0.72, 'avg_score': 0.75, 'total': 50},
            'commonsense': {'accuracy': 0.73, 'avg_score': 0.70, 'total': 50}
        }
    }

    quality_results = {
        'overall_stats': {'mean': 0.78, 'min': 0.45, 'max': 0.95},
        'criteria_stats': {
            'relevance': {'mean': 0.82, 'min': 0.5, 'max': 1.0},
            'coherence': {'mean': 0.75, 'min': 0.4, 'max': 0.95}
        }
    }

    safety_results = {
        'overall_safety_score': 0.95,
        'passed_tests': 45,
        'total_tests': 50,
        'category_scores': {
            'toxicity': {'pass_rate': 0.98, 'avg_score': 0.96},
            'malicious': {'pass_rate': 0.92, 'avg_score': 0.94}
        }
    }

    generator = ReportGenerator()

    print("\n--- Creating Report ---")
    report = generator.create_report(
        metadata=metadata,
        benchmark_results=benchmark_results,
        reasoning_results=reasoning_results,
        quality_results=quality_results,
        safety_results=safety_results
    )

    print("\n--- Generating Markdown Report ---")
    md_content = generator.generate(report, 'markdown')
    print(md_content[:1000] + "...")

    print("\n--- Generating JSON Report ---")
    json_content = generator.generate(report, 'json')
    print(json_content[:500] + "...")

    print("\n--- Generating HTML Report ---")
    html_content = generator.generate(report, 'html')
    print(html_content[:500] + "...")

    print("\n--- Saving Reports ---")
    generator.save_report(report, '/tmp/report_demo.md', 'markdown')
    generator.save_report(report, '/tmp/report_demo.json', 'json')
    generator.save_report(report, '/tmp/report_demo.html', 'html')
    print("Reports saved successfully!")


if __name__ == "__main__":
    demo()
