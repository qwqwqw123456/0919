"""
在线样本收集模块

用于在线 RLHF 流程中收集偏好样本，包括:
1. 策略模型响应生成
2. 偏好标注
3. 样本存储与管理

支持:
- 在线/离线样本收集
- 批量样本收集
- 多模型采样
- 样本过滤与质量控制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any, Callable, Iterator
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
import json
import os
import logging
from pathlib import Path
from tqdm import tqdm
import random
from collections import deque
import time
from threading import Lock
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SampleCollectionConfig:
    """样本收集配置"""
    max_response_length: int = 512
    max_prompt_length: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    num_samples_per_prompt: int = 2
    min_response_length: int = 10
    max_response_length_sample: int = 2048
    batch_size: int = 1
    num_workers: int = 4
    save_interval: int = 100
    output_dir: str = "./collected_samples"
    sample_format: str = "jsonl"


class ResponseGenerator:
    """
    响应生成器

    使用策略模型生成响应，支持多种采样策略。
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: Optional[torch.device] = None,
        max_length: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        repetition_penalty: float = 1.0
    ) -> List[str]:
        """
        生成响应

        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 采样温度
            top_p: top-p 采样参数
            top_k: top-k 采样参数
            do_sample: 是否采样
            num_return_sequences: 返回序列数量
            repetition_penalty: 重复惩罚

        Returns:
            生成的响应列表
        """
        if max_length is None:
            max_length = self.max_length
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        if top_k is None:
            top_k = self.top_k

        prompt_tokens = self.tokenizer.encode(
            prompt,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        generated_ids = self.model.generate(
            prompt_tokens,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id or 0,
            eos_token_id=self.tokenizer.eos_token_id
        )

        responses = []
        prompt_len = prompt_tokens.size(1)

        for gen_ids in generated_ids:
            response_ids = gen_ids[prompt_len:]
            response = self.tokenizer.decode(
                response_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            responses.append(response.strip())

        return responses

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        num_return_sequences: int = 1
    ) -> List[List[str]]:
        """
        批量生成响应

        Args:
            prompts: prompt 列表
            max_length: 最大生成长度
            temperature: 采样温度
            num_return_sequences: 每个 prompt 返回的序列数量

        Returns:
            每个 prompt 对应的响应列表
        """
        if max_length is None:
            max_length = self.max_length
        if temperature is None:
            temperature = self.temperature

        all_responses = []

        for prompt in prompts:
            responses = self.generate(
                prompt,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=num_return_sequences
            )
            all_responses.append(responses)

        return all_responses


class PreferenceLabeler:
    """
    偏好标注器

    支持多种偏好标注方式:
    1. 基于奖励模型
    2. 基于规则
    3. 随机
    """

    def __init__(
        self,
        reward_model: Optional[nn.Module] = None,
        tokenizer = None,
        device: Optional[torch.device] = None
    ):
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.reward_model is not None:
            self.reward_model.to(self.device)
            self.reward_model.eval()

    def label_with_reward_model(
        self,
        prompt: str,
        responses: List[str]
    ) -> Tuple[str, str]:
        """
        使用奖励模型标注偏好

        Args:
            prompt: 输入提示
            responses: 响应列表

        Returns:
            (chosen_response, rejected_response)
        """
        if self.reward_model is None or self.tokenizer is None:
            raise ValueError("Reward model and tokenizer are required")

        scores = []
        sep_id = getattr(self.tokenizer, 'sep_token_id', None) or 151643
        bos_id = getattr(self.tokenizer, 'bos_token_id', None) or 151643

        for response in responses:
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            response_tokens = self.tokenizer.encode(response, add_special_tokens=False)

            full_tokens = [bos_id] + prompt_tokens + [sep_id] + response_tokens
            input_ids = torch.tensor([full_tokens], dtype=torch.long).to(self.device)
            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                score = self.reward_model(input_ids, attention_mask)
                if isinstance(score, tuple):
                    score = score[0]
                score = score.item() if hasattr(score, 'item') else score[0, 0].item()

            scores.append(score)

        sorted_responses = sorted(zip(scores, responses), key=lambda x: x[0], reverse=True)

        return sorted_responses[0][1], sorted_responses[-1][1]

    def label_with_rules(
        self,
        prompt: str,
        responses: List[str]
    ) -> Tuple[str, str]:
        """
        使用规则标注偏好

        规则:
        1. 长度适中优先
        2. 包含特定关键词优先
        3. 无重复内容优先
        """
        scored_responses = []

        for response in responses:
            score = 0.0

            word_count = len(response.split())
            if 50 <= word_count <= 200:
                score += 2.0
            elif 20 <= word_count < 50:
                score += 1.0
            elif word_count > 200:
                score -= 0.5 * (word_count - 200) / 100

            if any(marker in response for marker in ['1.', '2.', '3.', '- ', '* ']):
                score += 1.0

            words = response.lower().split()
            if len(words) > 0:
                unique_ratio = len(set(words)) / len(words)
                score += unique_ratio

            if '?' not in response:
                score += 0.5

            scored_responses.append((score, response))

        scored_responses.sort(key=lambda x: x[0], reverse=True)

        return scored_responses[0][1], scored_responses[-1][1]

    def label_random(
        self,
        prompt: str,
        responses: List[str]
    ) -> Tuple[str, str]:
        """
        随机标注偏好
        """
        if len(responses) < 2:
            return responses[0], responses[0]

        shuffled = responses.copy()
        random.shuffle(shuffled)

        chosen_idx, rejected_idx = 0, 1 if len(shuffled) > 1 else 0

        return shuffled[chosen_idx], shuffled[rejected_idx]


class SampleCollector:
    """
    样本收集器基类

    定义样本收集的通用接口。
    """

    def __init__(
        self,
        generator: ResponseGenerator,
        labeler: PreferenceLabeler,
        config: SampleCollectionConfig
    ):
        self.generator = generator
        self.labeler = labeler
        self.config = config

        self.collected_samples = []
        self.stats = {
            'total_prompts': 0,
            'total_responses': 0,
            'total_samples': 0,
            'failed_generations': 0
        }

        os.makedirs(config.output_dir, exist_ok=True)

    def collect_single(
        self,
        prompt: str,
        label_method: str = 'reward_model'
    ) -> Optional[Dict[str, Any]]:
        """
        收集单个 prompt 的偏好样本

        Args:
            prompt: 输入提示
            label_method: 标注方法 ('reward_model', 'rules', 'random')

        Returns:
            收集的样本字典
        """
        try:
            responses = self.generator.generate(
                prompt,
                num_return_sequences=self.config.num_samples_per_prompt,
                max_length=self.config.max_response_length_sample
            )

            responses = [r for r in responses if len(r) >= self.config.min_response_length]

            if len(responses) < 2:
                self.stats['failed_generations'] += 1
                return None

            self.stats['total_prompts'] += 1
            self.stats['total_responses'] += len(responses)

            if label_method == 'reward_model':
                chosen, rejected = self.labeler.label_with_reward_model(prompt, responses)
            elif label_method == 'rules':
                chosen, rejected = self.labeler.label_with_rules(prompt, responses)
            else:
                chosen, rejected = self.labeler.label_random(prompt, responses)

            sample = {
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected,
                'all_responses': responses,
                'label_method': label_method,
                'timestamp': time.time()
            }

            self.collected_samples.append(sample)
            self.stats['total_samples'] += 1

            return sample

        except Exception as e:
            logger.error(f"Error collecting sample: {e}")
            self.stats['failed_generations'] += 1
            return None

    def collect_batch(
        self,
        prompts: List[str],
        label_method: str = 'reward_model',
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        批量收集偏好样本

        Args:
            prompts: prompt 列表
            label_method: 标注方法
            show_progress: 是否显示进度条

        Returns:
            收集的样本列表
        """
        samples = []
        iterator = tqdm(prompts, desc="Collecting samples") if show_progress else prompts

        for prompt in iterator:
            sample = self.collect_single(prompt, label_method)
            if sample is not None:
                samples.append(sample)

            if len(samples) % self.config.save_interval == 0 and len(samples) > 0:
                self.save_samples()

        self.save_samples()

        return samples

    def save_samples(self, filename: Optional[str] = None):
        """
        保存收集的样本

        Args:
            filename: 保存文件名
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"samples_{timestamp}.jsonl"

        output_path = os.path.join(self.config.output_dir, filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in self.collected_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        logger.info(f"Saved {len(self.collected_samples)} samples to {output_path}")

    def get_stats(self) -> Dict[str, Any]:
        """获取收集统计信息"""
        return self.stats.copy()


class OnlineSampleCollector(SampleCollector):
    """
    在线样本收集器

    支持实时交互环境中的样本收集。
    """

    def __init__(
        self,
        generator: ResponseGenerator,
        labeler: PreferenceLabeler,
        config: SampleCollectionConfig,
        environment: Optional[Any] = None,
        max_buffer_size: int = 1000
    ):
        super().__init__(generator, labeler, config)

        self.environment = environment
        self.sample_buffer = deque(maxlen=max_buffer_size)
        self.lock = Lock()

        self.episode_history = []

    def collect_from_environment(
        self,
        num_episodes: int = 100,
        label_method: str = 'reward_model'
    ) -> List[Dict[str, Any]]:
        """
        从环境中收集样本

        Args:
            num_episodes: 收集的 episode 数量
            label_method: 标注方法

        Returns:
            收集的样本列表
        """
        if self.environment is None:
            raise ValueError("Environment is required for online collection")

        samples = []

        for episode_idx in tqdm(range(num_episodes), desc="Collecting episodes"):
            episode_samples = self._collect_episode(episode_idx, label_method)
            samples.extend(episode_samples)

            if len(samples) % self.config.save_interval == 0:
                self.save_samples()

        self.save_samples()

        return samples

    def _collect_episode(
        self,
        episode_idx: int,
        label_method: str
    ) -> List[Dict[str, Any]]:
        """
        收集单个 episode

        Args:
            episode_idx: episode 索引
            label_method: 标注方法

        Returns:
            episode 中的样本列表
        """
        episode_samples = []
        state = self.environment.reset()

        done = False
        step = 0

        while not done and step < 100:
            prompt = self._state_to_prompt(state)

            sample = self.collect_single(prompt, label_method)

            if sample is not None:
                episode_samples.append(sample)

                with self.lock:
                    self.sample_buffer.append(sample)

                action = self._select_action(sample['chosen'])
                next_state, reward, done, info = self.environment.step(action)
                state = next_state

            step += 1

        self.episode_history.append({
            'episode_idx': episode_idx,
            'num_samples': len(episode_samples),
            'timestamp': time.time()
        })

        return episode_samples

    def _state_to_prompt(self, state: Any) -> str:
        """将环境状态转换为 prompt"""
        if isinstance(state, str):
            return state
        elif isinstance(state, dict):
            return state.get('observation', str(state))
        else:
            return str(state)

    def _select_action(self, response: str) -> str:
        """选择动作"""
        return response


class SampleFilter:
    """
    样本过滤器

    对收集的样本进行质量过滤。
    """

    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 2048,
        max_repetition_ratio: float = 0.7,
        require_diversity: bool = True,
        min_length_diff: int = 10
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.max_repetition_ratio = max_repetition_ratio
        self.require_diversity = require_diversity
        self.min_length_diff = min_length_diff

    def filter(self, sample: Dict[str, Any]) -> bool:
        """
        过滤样本

        Args:
            sample: 样本字典

        Returns:
            是否保留样本
        """
        chosen = sample.get('chosen', '')
        rejected = sample.get('rejected', '')

        if not self._check_length(chosen, rejected):
            return False

        if not self._check_repetition(chosen, rejected):
            return False

        if self.require_diversity and not self._check_diversity(chosen, rejected):
            return False

        return True

    def _check_length(self, chosen: str, rejected: str) -> bool:
        """检查长度"""
        if len(chosen) < self.min_length or len(rejected) < self.min_length:
            return False

        if len(chosen) > self.max_length or len(rejected) > self.max_length:
            return False

        return True

    def _check_repetition(self, chosen: str, rejected: str) -> bool:
        """检查重复"""
        for response in [chosen, rejected]:
            if self._has_excessive_repetition(response):
                return False

        return True

    def _has_excessive_repetition(self, text: str, n: int = 10) -> bool:
        """检查过度重复"""
        if len(text) < n:
            return False

        words = text.lower().split()
        if len(words) < n:
            return False

        ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
        counter = {}
        for ng in ngrams:
            counter[ng] = counter.get(ng, 0) + 1

        if not counter:
            return False

        most_common_count = max(counter.values())
        repetition_ratio = most_common_count / len(ngrams)

        return repetition_ratio > self.max_repetition_ratio

    def _check_diversity(self, chosen: str, rejected: str) -> bool:
        """检查多样性"""
        chosen_len = len(chosen)
        rejected_len = len(rejected)

        length_diff = abs(chosen_len - rejected_len)

        if length_diff < self.min_length_diff:
            common_chars = set(chosen.lower()) & set(rejected.lower())
            if len(common_chars) / 26 > 0.8:
                return False

        return True


class CollectedSampleDataset(Dataset):
    """
    收集的样本数据集

    用于存储和加载收集的偏好样本。
    """

    def __init__(
        self,
        data_path: Union[str, List[Dict[str, Any]]],
        filter: Optional[SampleFilter] = None
    ):
        self.filter = filter or SampleFilter()
        self.samples = self._load_samples(data_path)

    def _load_samples(
        self,
        data_path: Union[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """加载样本"""
        samples = []

        if isinstance(data_path, list):
            samples = data_path
        elif os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    samples.append(json.loads(line.strip()))
        else:
            raise ValueError(f"Invalid data_path: {data_path}")

        filtered_samples = [s for s in samples if self.filter.filter(s)]

        logger.info(f"Loaded {len(samples)} samples, {len(filtered_samples)} after filtering")

        return filtered_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        if not self.samples:
            return {}

        chosen_lengths = [len(s['chosen']) for s in self.samples]
        rejected_lengths = [len(s['rejected']) for s in self.samples]

        return {
            'num_samples': len(self.samples),
            'avg_chosen_length': sum(chosen_lengths) / len(chosen_lengths),
            'avg_rejected_length': sum(rejected_lengths) / len(rejected_lengths),
            'min_chosen_length': min(chosen_lengths),
            'max_chosen_length': max(chosen_lengths),
            'min_rejected_length': min(rejected_lengths),
            'max_rejected_length': max(rejected_lengths)
        }


def create_sample_collector(
    policy_model: nn.Module,
    tokenizer,
    reward_model: Optional[nn.Module] = None,
    config: Optional[SampleCollectionConfig] = None,
    device: Optional[torch.device] = None
) -> SampleCollector:
    """
    创建样本收集器的工厂函数

    Args:
        policy_model: 策略模型
        tokenizer: 分词器
        reward_model: 奖励模型（可选）
        config: 收集配置
        device: 设备

    Returns:
        SampleCollector 实例
    """
    if config is None:
        config = SampleCollectionConfig()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = ResponseGenerator(
        model=policy_model,
        tokenizer=tokenizer,
        device=device,
        max_length=config.max_response_length,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k
    )

    labeler = PreferenceLabeler(
        reward_model=reward_model,
        tokenizer=tokenizer,
        device=device
    )

    return SampleCollector(
        generator=generator,
        labeler=labeler,
        config=config
    )


def merge_samples(
    sample_paths: List[str],
    output_path: str,
    filter: Optional[SampleFilter] = None
) -> int:
    """
    合并多个样本文件

    Args:
        sample_paths: 样本文件路径列表
        output_path: 输出文件路径
        filter: 样本过滤器

    Returns:
        合并后的样本数量
    """
    all_samples = []

    for path in sample_paths:
        if not os.path.exists(path):
            continue

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                all_samples.append(sample)

    if filter is not None:
        filtered_samples = [s for s in all_samples if filter.filter(s)]
        all_samples = filtered_samples

    random.shuffle(all_samples)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    logger.info(f"Merged {len(all_samples)} samples to {output_path}")

    return len(all_samples)


def analyze_samples(sample_path: str) -> Dict[str, Any]:
    """
    分析样本文件

    Args:
        sample_path: 样本文件路径

    Returns:
        分析结果字典
    """
    samples = []

    with open(sample_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    if not samples:
        return {}

    chosen_lengths = [len(s['chosen']) for s in samples]
    rejected_lengths = [len(s['rejected']) for s in samples]

    label_methods = {}
    for s in samples:
        method = s.get('label_method', 'unknown')
        label_methods[method] = label_methods.get(method, 0) + 1

    return {
        'total_samples': len(samples),
        'chosen_lengths': {
            'mean': sum(chosen_lengths) / len(chosen_lengths),
            'min': min(chosen_lengths),
            'max': max(chosen_lengths)
        },
        'rejected_lengths': {
            'mean': sum(rejected_lengths) / len(rejected_lengths),
            'min': min(rejected_lengths),
            'max': max(rejected_lengths)
        },
        'label_methods': label_methods
    }


if __name__ == '__main__':
    logger.info("Online Sample Collection Module")
    logger.info("Import this module to use the SampleCollector class")
