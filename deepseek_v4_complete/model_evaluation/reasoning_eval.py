"""
推理能力评估模块

提供全面的语言模型推理能力评估，包括：
- 逻辑推理 (Logical Reasoning)
- 数学推理 (Mathematical Reasoning)
- 常识推理 (Commonsense Reasoning)
- 演绎推理 (Deductive Reasoning)
- 归纳推理 (Inductive Reasoning)
- 溯因推理 (Abductive Reasoning)
- 多步推理 (Multi-step Reasoning)
- 链式思维 (Chain-of-Thought) 评估
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from enum import Enum
import random


class ReasoningType(Enum):
    """推理类型枚举"""
    LOGICAL = "logical"
    MATHEMATICAL = "mathematical"
    COMMONSENSE = "commonsense"
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    MULTI_STEP = "multi_step"
    CHAIN_OF_THOUGHT = "chain_of_thought"


@dataclass
class ReasoningResult:
    """推理测试结果"""
    reasoning_type: str
    question: str
    model_response: str
    ground_truth: str
    is_correct: bool
    reasoning_steps: List[str] = field(default_factory=list)
    step_scores: List[float] = field(default_factory=list)
    overall_score: float = 0.0
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'reasoning_type': self.reasoning_type,
            'question': self.question,
            'model_response': self.model_response,
            'ground_truth': self.ground_truth,
            'is_correct': self.is_correct,
            'reasoning_steps': self.reasoning_steps,
            'step_scores': self.step_scores,
            'overall_score': self.overall_score,
            'execution_time': self.execution_time,
            'metadata': self.metadata
        }


@dataclass
class ReasoningEvalResult:
    """推理能力评估汇总结果"""
    reasoning_type: str
    total_questions: int
    correct_count: int
    accuracy: float
    avg_score: float
    avg_execution_time: float
    results: List[ReasoningResult] = field(default_factory=list)
    detailed_stats: Dict[str, Any] = field(default_factory=dict)


class ReasoningDataset:
    """推理数据集管理"""

    LOGICAL_REASONING_DATA = [
        {
            'id': 'logic_001',
            'question': 'If all cats are animals, and some animals are pets, can we conclude that some cats are pets?',
            'options': ['Yes', 'No', 'Cannot be determined'],
            'answer': 'Cannot be determined',
            'explanation': 'This is a syllogism that does not allow a definite conclusion. We know all cats are animals and some animals are pets, but we cannot determine if cats specifically are among those pets.'
        },
        {
            'id': 'logic_002',
            'question': 'If it rains, the ground is wet. The ground is wet. What can we conclude?',
            'options': ['It rained', 'It did not rain', 'Cannot be determined'],
            'answer': 'Cannot be determined',
            'explanation': 'This is the fallacy of affirming the consequent. The ground could be wet for other reasons besides rain.'
        },
        {
            'id': 'logic_003',
            'question': 'If John studies, he passes the exam. John did not pass the exam. What can we conclude?',
            'options': ['John did not study', 'John studied', 'Cannot be determined'],
            'answer': 'John did not study',
            'explanation': 'This is valid modus tollens: If P then Q. Not Q. Therefore not P.'
        },
        {
            'id': 'logic_004',
            'question': 'All roses are flowers. Some flowers fade quickly. Therefore: ?',
            'options': ['All roses fade quickly', 'Some roses may fade quickly', 'No roses fade quickly'],
            'answer': 'Some roses may fade quickly',
            'explanation': 'We cannot definitively conclude about roses from this information, only that some roses might fade.'
        },
        {
            'id': 'logic_005',
            'question': 'If A > B and B > C, then: ?',
            'options': ['A > C', 'A < C', 'A = C'],
            'answer': 'A > C',
            'explanation': 'By transitivity of greater than, if A > B and B > C, then A > C.'
        }
    ]

    MATHEMATICAL_REASONING_DATA = [
        {
            'id': 'math_001',
            'question': 'A train travels 120 miles in 2 hours, then stops for 30 minutes, then travels 80 miles in 1 hour. What is the average speed for the entire journey?',
            'answer': '80 mph',
            'solution_steps': [
                'Total distance: 120 + 80 = 200 miles',
                'Total time: 2 + 0.5 + 1 = 3.5 hours',
                'Average speed = Total distance / Total time = 200 / 3.5 ≈ 57.14 mph'
            ]
        },
        {
            'id': 'math_002',
            'question': 'If x + y = 10 and x - y = 4, what is the value of x?',
            'answer': '7',
            'solution_steps': [
                'Add the two equations: (x + y) + (x - y) = 10 + 4',
                '2x = 14',
                'x = 7'
            ]
        },
        {
            'id': 'math_003',
            'question': 'A rectangle has a perimeter of 24 cm. If its length is 7 cm, what is its area?',
            'answer': '35 square cm',
            'solution_steps': [
                'Perimeter = 2(length + width) = 24',
                'length + width = 12',
                'width = 12 - 7 = 5 cm',
                'Area = length × width = 7 × 5 = 35 sq cm'
            ]
        },
        {
            'id': 'math_004',
            'question': 'What is the sum of all even numbers from 1 to 100?',
            'answer': '2550',
            'solution_steps': [
                'Even numbers from 1 to 100: 2, 4, 6, ..., 100',
                'This is an arithmetic series with n = 50 terms',
                'First term a1 = 2, last term an = 100',
                'Sum = n(a1 + an)/2 = 50(2 + 100)/2 = 50 × 51 = 2550'
            ]
        },
        {
            'id': 'math_005',
            'question': 'If 40% of a number is 60, what is 60% of the same number?',
            'answer': '90',
            'solution_steps': [
                'Let x be the number: 0.4x = 60',
                'x = 60 / 0.4 = 150',
                '60% of 150 = 0.6 × 150 = 90'
            ]
        }
    ]

    COMMONSENSE_REASONING_DATA = [
        {
            'id': 'common_001',
            'question': 'You are thirsty and walking in a desert. You find a sealed water bottle. What should you do?',
            'options': ['Drink immediately', 'Check expiration date', 'Share with others'],
            'answer': 'Drink immediately',
            'explanation': 'In a survival situation where you are thirsty in a desert, drinking water immediately is the most urgent priority.'
        },
        {
            'id': 'common_002',
            'question': 'Why do we add salt to pasta water when cooking?',
            'options': ['To make water boil faster', 'To season the pasta', 'To prevent pasta from sticking'],
            'answer': 'To season the pasta',
            'explanation': 'Salt is added to pasta water primarily to season the pasta as it cooks. While salt can slightly raise the boiling point, this effect is minimal.'
        },
        {
            'id': 'common_003',
            'question': 'You forgot your phone at home and are late for an important meeting. What is the most practical action?',
            'options': ['Go back home to get it', 'Continue to the meeting', 'Call someone to bring it'],
            'answer': 'Continue to the meeting',
            'explanation': 'Since you are already late, going back would make you even later. You can deal with the phone issue after the meeting.'
        },
        {
            'id': 'common_004',
            'question': 'A glass of hot coffee is left on a table in a cold room. What happens over time?',
            'options': ['Coffee stays hot', 'Coffee gets colder', 'Coffee evaporates completely'],
            'answer': 'Coffee gets colder',
            'explanation': 'Heat naturally flows from hotter objects to cooler surroundings until thermal equilibrium is reached.'
        },
        {
            'id': 'common_005',
            'question': 'You see dark clouds approaching and feel the temperature drop. What will most likely happen soon?',
            'options': ['Sunny weather', 'Rain or storm', 'Snow'],
            'answer': 'Rain or storm',
            'explanation': 'Dark clouds, dropping temperature, and pressure changes typically indicate approaching precipitation.'
        }
    ]

    MULTI_STEP_REASONING_DATA = [
        {
            'id': 'multi_001',
            'question': 'Tom has twice as many apples as Jerry. Jerry has 3 more oranges than Tom. Together they have 27 fruits. How many apples does Tom have?',
            'answer': '12',
            'solution_steps': [
                'Let T = Tom apples, J = Jerry apples',
                'T = 2J (Tom has twice as many apples)',
                'Jerry oranges = T + 3',
                'Total fruits: J + T + (T + 3) = 27',
                'Substitute T = 2J: J + 2J + (2J + 3) = 27',
                '5J + 3 = 27, 5J = 24, J = 4.8 (must be whole, let me reconsider)',
                'Let A = Tom apples, Jerry has A/2 apples',
                'Jerry oranges = A/2 + 3',
                'Total: A + A/2 + A/2 + 3 = 27',
                '2A + 3 = 27, 2A = 24, A = 12'
            ]
        },
        {
            'id': 'multi_002',
            'question': 'A store offers 20% off, then an additional 10% off the sale price. What is the final price of a $100 item?',
            'answer': '$72',
            'solution_steps': [
                'Original price: $100',
                'After 20% off: $100 × 0.8 = $80',
                'After additional 10% off: $80 × 0.9 = $72',
                'Final price: $72'
            ]
        },
        {
            'id': 'multi_003',
            'question': 'In a group of 50 students, 30 play soccer, 25 play basketball, and 10 play both. How many play neither?',
            'answer': '5',
            'solution_steps': [
                'Students who play at least one sport: |S ∪ B| = |S| + |B| - |S ∩ B|',
                '= 30 + 25 - 10 = 45',
                'Students who play neither = Total - At least one',
                '= 50 - 45 = 5'
            ]
        }
    ]

    CHAIN_OF_THOUGHT_DATA = [
        {
            'id': 'cot_001',
            'prompt_type': 'explicit',
            'question': 'If a store sells 3 apples for $2, how much would 15 apples cost?',
            'answer': '$10',
            'expected_chain': [
                'First, determine the cost per apple: $2 / 3 = $0.67 per apple',
                'Then, multiply by 15: 15 × $0.67 = $10',
                'Alternative: 15 / 3 = 5 groups of 3 apples',
                '5 × $2 = $10'
            ]
        },
        {
            'id': 'cot_002',
            'prompt_type': 'explicit',
            'question': 'John is twice as old as Mary was 5 years ago. Mary is currently 15. How old is John?',
            'answer': '20',
            'expected_chain': [
                'Mary is currently 15',
                '5 years ago, Mary was 15 - 5 = 10',
                'John is twice that age: 2 × 10 = 20',
                'John is 20 years old'
            ]
        }
    ]

    @classmethod
    def get_dataset(cls, reasoning_type: str) -> List[Dict[str, Any]]:
        """
        获取指定类型的推理数据集

        Args:
            reasoning_type: 推理类型

        Returns:
            List[Dict]: 数据集
        """
        datasets = {
            'logical': cls.LOGICAL_REASONING_DATA,
            'mathematical': cls.MATHEMATICAL_REASONING_DATA,
            'commonsense': cls.COMMONSENSE_REASONING_DATA,
            'multi_step': cls.MULTI_STEP_REASONING_DATA,
            'chain_of_thought': cls.CHAIN_OF_THOUGHT_DATA,
            'deductive': cls.LOGICAL_REASONING_DATA[:3],
            'inductive': cls.COMMONSENSE_REASONING_DATA[:3],
            'abductive': cls.COMMONSENSE_REASONING_DATA[2:5]
        }
        return datasets.get(reasoning_type, [])


class ReasoningStepExtractor:
    """推理步骤提取器"""

    STEP_PATTERNS = [
        r'\d+\.\s*(.+)',
        r'[-•]\s*(.+)',
        r'First[,\s](.+)',
        r'Then[,\s](.+)',
        r'Finally[,\s](.+)',
        r'Step\s*\d+[:\s](.+)',
        r'Therefore[,\s](.+)',
        r'So[,\s](.+)',
        r'Thus[,\s](.+)'
    ]

    def __init__(self):
        """初始化步骤提取器"""
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.STEP_PATTERNS]

    def extract_steps(self, text: str) -> List[str]:
        """
        从文本中提取推理步骤

        Args:
            text: 模型响应文本

        Returns:
            List[str]: 提取的推理步骤列表
        """
        steps = []

        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            for pattern in self.patterns:
                match = pattern.match(line)
                if match:
                    step = match.group(1).strip()
                    if step and len(step) > 5:
                        steps.append(step)
                        break
            else:
                if len(line) > 10:
                    steps.append(line)

        return steps

    def evaluate_step_quality(self, steps: List[str], expected_steps: List[str] = None) -> List[float]:
        """
        评估每个推理步骤的质量

        Args:
            steps: 模型生成的步骤
            expected_steps: 期望的步骤（如果有）

        Returns:
            List[float]: 每个步骤的评分
        """
        if not steps:
            return []

        scores = []
        for step in steps:
            score = self._evaluate_single_step(step)
            scores.append(score)

        return scores

    def _evaluate_single_step(self, step: str) -> float:
        """评估单个步骤的质量"""
        score = 0.5

        keywords = ['if', 'then', 'therefore', 'because', 'since', 'thus', 'so', 'means', 'equals']
        keyword_count = sum(1 for kw in keywords if kw in step.lower())
        score += min(keyword_count * 0.1, 0.3)

        if re.search(r'\d+', step):
            score += 0.1

        if re.search(r'[=<>+\-*/]', step):
            score += 0.1

        return min(score, 1.0)


class ReasoningEval:
    """推理能力评估器"""

    def __init__(self):
        """初始化推理评估器"""
        self.dataset = ReasoningDataset()
        self.step_extractor = ReasoningStepExtractor()
        self.results = []

    def evaluate(
        self,
        model: Any,
        reasoning_types: List[str] = None,
        num_samples: int = None,
        verbose: bool = True
    ) -> Dict[str, ReasoningEvalResult]:
        """
        运行完整的推理能力评估

        Args:
            model: 待评估的模型
            reasoning_types: 要评估的推理类型列表
            num_samples: 每个类型的样本数量
            verbose: 是否打印详细信息

        Returns:
            Dict[str, ReasoningEvalResult]: 各推理类型的评估结果
        """
        if reasoning_types is None:
            reasoning_types = ['logical', 'mathematical', 'commonsense', 'multi_step']

        all_results = {}

        for rtype in reasoning_types:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Evaluating {rtype.upper()} Reasoning...")
                print(f"{'='*60}")

            result = self._evaluate_type(model, rtype, num_samples, verbose)
            all_results[rtype] = result

        return all_results

    def _evaluate_type(
        self,
        model: Any,
        reasoning_type: str,
        num_samples: int = None,
        verbose: bool = True
    ) -> ReasoningEvalResult:
        """评估特定推理类型"""
        dataset = self.dataset.get_dataset(reasoning_type)

        if num_samples and num_samples < len(dataset):
            dataset = random.sample(dataset, num_samples)

        results = []
        correct = 0
        total_score = 0.0
        total_time = 0.0

        for i, sample in enumerate(dataset):
            start_time = time.time()

            result = self._evaluate_sample(model, sample, reasoning_type)
            result.execution_time = time.time() - start_time

            results.append(result)

            if result.is_correct:
                correct += 1
            total_score += result.overall_score
            total_time += result.execution_time

            if verbose:
                print(f"  [{i+1}/{len(dataset)}] {'✓' if result.is_correct else '✗'} "
                      f"(Score: {result.overall_score:.2f})")

        n = len(dataset)
        accuracy = correct / n if n > 0 else 0
        avg_score = total_score / n if n > 0 else 0
        avg_time = total_time / n if n > 0 else 0

        return ReasoningEvalResult(
            reasoning_type=reasoning_type,
            total_questions=n,
            correct_count=correct,
            accuracy=accuracy,
            avg_score=avg_score,
            avg_execution_time=avg_time,
            results=results,
            detailed_stats=self._compute_detailed_stats(results)
        )

    def _evaluate_sample(
        self,
        model: Any,
        sample: Dict[str, Any],
        reasoning_type: str
    ) -> ReasoningResult:
        """评估单个样本"""
        question = sample['question']
        ground_truth = sample.get('answer', '')

        prompt = self._build_prompt(question, reasoning_type, sample)

        try:
            if callable(getattr(model, 'generate', None)):
                response = model.generate(prompt)
            elif callable(getattr(model, '__call__', None)):
                response = model(prompt)
            else:
                response = ground_truth

            steps = self.step_extractor.extract_steps(response)
            step_scores = self.step_extractor.evaluate_step_quality(steps)

            is_correct = self._check_answer(response, ground_truth, sample)
            overall_score = self._calculate_overall_score(
                response, ground_truth, steps, step_scores, sample
            )

            return ReasoningResult(
                reasoning_type=reasoning_type,
                question=question,
                model_response=response,
                ground_truth=ground_truth,
                is_correct=is_correct,
                reasoning_steps=steps,
                step_scores=step_scores,
                overall_score=overall_score
            )

        except Exception as e:
            return ReasoningResult(
                reasoning_type=reasoning_type,
                question=question,
                model_response=str(e),
                ground_truth=ground_truth,
                is_correct=False,
                overall_score=0.0
            )

    def _build_prompt(
        self,
        question: str,
        reasoning_type: str,
        sample: Dict[str, Any]
    ) -> str:
        """构建评估提示"""
        if reasoning_type == 'chain_of_thought':
            return f"""Please think step by step and show your reasoning process.

Question: {question}

Your response should include clear reasoning steps."""

        elif reasoning_type == 'logical':
            if 'options' in sample:
                options = '\n'.join(sample['options'])
                return f"Question: {question}\nOptions:\n{options}\n\nExplain your reasoning and give the answer."
            return f"Analyze the following logical statement and provide your answer with reasoning:\n\n{question}"

        elif reasoning_type == 'mathematical':
            steps_hint = 'Show your work step by step.'
            if 'solution_steps' in sample:
                steps_hint = f"Expected solution approach: Think step by step, considering {len(sample['solution_steps'])} key steps."
            return f"{question}\n\n{steps_hint}\n\nProvide your final answer."

        else:
            return f"{question}\n\nPlease provide your answer with reasoning."

    def _check_answer(
        self,
        response: str,
        ground_truth: str,
        sample: Dict[str, Any]
    ) -> bool:
        """检查答案是否正确"""
        response_lower = response.lower().strip()
        truth_lower = ground_truth.lower().strip()

        if response_lower == truth_lower:
            return True

        response_numbers = re.findall(r'-?\d+\.?\d*', response_lower)
        truth_numbers = re.findall(r'-?\d+\.?\d*', truth_lower)

        if response_numbers and truth_numbers:
            if response_numbers == truth_numbers:
                return True
            try:
                resp_num = float(response_numbers[-1])
                true_num = float(truth_numbers[-1])
                if abs(resp_num - true_num) < 0.01:
                    return True
            except ValueError:
                pass

        if truth_lower in response_lower:
            return True

        if 'options' in sample:
            for option in sample['options']:
                if option.lower() in response_lower or response_lower in option.lower():
                    if option.lower() == truth_lower:
                        return True

        return False

    def _calculate_overall_score(
        self,
        response: str,
        ground_truth: str,
        steps: List[str],
        step_scores: List[float],
        sample: Dict[str, Any]
    ) -> float:
        """计算综合评分"""
        correctness_score = 1.0 if self._check_answer(response, ground_truth, sample) else 0.0

        step_score = 0.0
        if step_scores:
            step_score = sum(step_scores) / len(step_scores)

        step_count_score = min(len(steps) / 3, 1.0) if steps else 0.0

        completeness_score = 0.5
        if len(response) > 50:
            completeness_score += 0.25
        if len(steps) >= 2:
            completeness_score += 0.25

        overall = correctness_score * 0.5 + step_score * 0.3 + step_count_score * 0.1 + completeness_score * 0.1

        return min(overall, 1.0)

    def _compute_detailed_stats(self, results: List[ReasoningResult]) -> Dict[str, Any]:
        """计算详细统计信息"""
        if not results:
            return {}

        step_counts = [len(r.reasoning_steps) for r in results]
        score_lists = [r.step_scores for r in results if r.step_scores]

        avg_steps = sum(step_counts) / len(step_counts) if step_counts else 0
        max_steps = max(step_counts) if step_counts else 0
        min_steps = min(step_counts) if step_counts else 0

        all_step_scores = [s for scores in score_lists for s in scores]
        avg_step_score = sum(all_step_scores) / len(all_step_scores) if all_step_scores else 0

        return {
            'avg_reasoning_steps': avg_steps,
            'max_reasoning_steps': max_steps,
            'min_reasoning_steps': min_steps,
            'avg_step_score': avg_step_score,
            'correct_with_steps': sum(1 for r in results if r.reasoning_steps and r.is_correct),
            'correct_without_steps': sum(1 for r in results if not r.reasoning_steps and r.is_correct)
        }

    def evaluate_logical_reasoning(self, model: Any) -> ReasoningEvalResult:
        """评估逻辑推理能力"""
        return self._evaluate_type(model, 'logical')

    def evaluate_mathematical_reasoning(self, model: Any) -> ReasoningEvalResult:
        """评估数学推理能力"""
        return self._evaluate_type(model, 'mathematical')

    def evaluate_commonsense_reasoning(self, model: Any) -> ReasoningEvalResult:
        """评估常识推理能力"""
        return self._evaluate_type(model, 'commonsense')

    def evaluate_chain_of_thought(self, model: Any) -> ReasoningEvalResult:
        """评估链式思维能力"""
        return self._evaluate_type(model, 'chain_of_thought')

    def evaluate_multi_step(self, model: Any) -> ReasoningEvalResult:
        """评估多步推理能力"""
        return self._evaluate_type(model, 'multi_step')

    def get_summary(self, results: Dict[str, ReasoningEvalResult]) -> Dict[str, Any]:
        """
        获取评估结果摘要

        Args:
            results: 各推理类型的评估结果

        Returns:
            Dict: 摘要信息
        """
        summary = {
            'overall_accuracy': 0.0,
            'overall_avg_score': 0.0,
            'by_type': {},
            'total_questions': 0,
            'total_correct': 0
        }

        total_acc = 0.0
        total_score = 0.0
        count = 0

        for rtype, result in results.items():
            summary['by_type'][rtype] = {
                'accuracy': result.accuracy,
                'avg_score': result.avg_score,
                'correct': result.correct_count,
                'total': result.total_questions,
                'avg_time': result.avg_execution_time
            }
            total_acc += result.accuracy
            total_score += result.avg_score
            count += 1
            summary['total_questions'] += result.total_questions
            summary['total_correct'] += result.correct_count

        if count > 0:
            summary['overall_accuracy'] = total_acc / count
            summary['overall_avg_score'] = total_score / count

        return summary

    def save_results(self, results: Dict[str, ReasoningEvalResult], filepath: str) -> None:
        """保存评估结果"""
        output = {
            'summary': self.get_summary(results),
            'detailed_results': {}
        }

        for rtype, result in results.items():
            output['detailed_results'][rtype] = {
                'stats': {
                    'accuracy': result.accuracy,
                    'avg_score': result.avg_score,
                    'total_questions': result.total_questions,
                    'correct_count': result.correct_count,
                    'detailed_stats': result.detailed_stats
                },
                'samples': [r.to_dict() for r in result.results]
            }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    def load_results(self, filepath: str) -> Dict[str, Any]:
        """加载评估结果"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)


class MockModel:
    """用于测试的模拟模型"""

    def generate(self, prompt: str) -> str:
        """模拟生成响应"""
        if 'logical' in prompt.lower() or 'analyze' in prompt.lower():
            return "Based on the logical structure, the answer is: No, Cannot be determined. This follows from the rules of deductive logic."
        elif 'step' in prompt.lower() or 'calculate' in prompt.lower():
            return "Step 1: Understand the problem.\nStep 2: Apply the relevant formula.\nStep 3: Calculate the result.\nTherefore, the answer is 42."
        elif 'apple' in prompt.lower():
            return "Each group of 3 apples costs $2.\n15 apples = 5 groups of 3.\n5 × $2 = $10\nFinal answer: $10"
        else:
            return "Based on careful reasoning and analysis, the answer is yes."

    def __call__(self, prompt: str) -> str:
        """使模拟模型可调用"""
        return self.generate(prompt)


def demo():
    """演示函数"""
    print("=" * 60)
    print("推理能力评估演示")
    print("=" * 60)

    mock_model = MockModel()
    evaluator = ReasoningEval()

    print("\n--- Logical Reasoning ---")
    logical_result = evaluator.evaluate_logical_reasoning(mock_model)
    print(f"Accuracy: {logical_result.accuracy:.4f}")
    print(f"Average Score: {logical_result.avg_score:.4f}")

    print("\n--- Mathematical Reasoning ---")
    math_result = evaluator.evaluate_mathematical_reasoning(mock_model)
    print(f"Accuracy: {math_result.accuracy:.4f}")
    print(f"Average Score: {math_result.avg_score:.4f}")

    print("\n--- Multi-Step Reasoning ---")
    multi_result = evaluator.evaluate_multi_step(mock_model)
    print(f"Accuracy: {multi_result.accuracy:.4f}")
    print(f"Average Score: {multi_result.avg_score:.4f}")

    print("\n--- Full Evaluation ---")
    all_results = evaluator.evaluate(
        mock_model,
        reasoning_types=['logical', 'mathematical', 'commonsense']
    )
    summary = evaluator.get_summary(all_results)
    print(f"\nOverall Accuracy: {summary['overall_accuracy']:.4f}")
    print(f"Overall Avg Score: {summary['overall_avg_score']:.4f}")


if __name__ == "__main__":
    demo()
