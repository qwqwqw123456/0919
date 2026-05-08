"""
奖励评分模块
使用训练好的奖励模型对文本进行评分

功能:
1. 单个文本的奖励评分
2. 偏好对的奖励评分与比较
3. 批量评分
4. 多种评分策略（序列评分、最后token评分、平均评分）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScoringConfig:
    """评分配置参数"""
    max_length: int = 2048
    batch_size: int = 8
    use_amp: bool = True
    device: Optional[str] = None
    scoring_strategy: str = 'last_token'
    temperature: float = 1.0
    normalize_rewards: bool = True
    output_dir: str = "./reward_scores"


class RewardScorer:
    """
    奖励模型评分器
    
    使用预训练的奖励模型对文本进行评分，支持多种评分策略。
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: Optional[ScoringConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        
        if config is None:
            config = ScoringConfig()
        self.config = config
        
        if device is None:
            if config.device is not None:
                device = torch.device(config.device)
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        self.model.to(self.device)
        self.model.eval()
    
    def _encode_sequence(
        self,
        prompt: str,
        response: str,
        max_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        编码 prompt + response 序列
        
        格式: <bos> prompt <sep> response <eos>
        """
        if max_length is None:
            max_length = self.config.max_length
        
        sep_id = self.tokenizer.sep_token_id if hasattr(self.tokenizer, 'sep_token_id') else 151643
        bos_id = self.tokenizer.bos_token_id if hasattr(self.tokenizer, 'bos_token_id') else 151643
        eos_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else 151643
        
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
        
        full_tokens = [bos_id] + prompt_tokens + [sep_id] + response_tokens + [eos_id]
        
        prompt_len = len(prompt_tokens) + 2
        
        if len(full_tokens) > max_length:
            full_tokens = full_tokens[:max_length]
            response_mask = torch.zeros(max_length, dtype=torch.bool)
            response_mask[min(prompt_len, max_length):] = True
        else:
            response_mask = torch.zeros(len(full_tokens), dtype=torch.bool)
            response_mask[prompt_len:] = True
        
        input_ids = torch.tensor(full_tokens, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'prompt_len': prompt_len,
            'response_mask': response_mask
        }
    
    def _score_sequence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int
    ) -> Tuple[float, Optional[np.ndarray]]:
        """
        对序列进行评分
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            prompt_len: prompt长度
            
        Returns:
            (scalar_reward, per_token_rewards)
        """
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.config.use_amp):
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                return_hidden=True
            )
            
            if isinstance(outputs, tuple):
                if len(outputs) == 2:
                    rewards, hidden_states = outputs
                else:
                    hidden_states = outputs[-1]
                    rewards = self.model.reward_head(
                        hidden_states[:, -1, :]
                    ).squeeze(-1)
            else:
                hidden_states = outputs
                rewards = self.model.reward_head(
                    hidden_states[:, -1, :]
                ).squeeze(-1)
            
            if hasattr(rewards, '__len__') and len(rewards.shape) > 0:
                scalar_reward = rewards[0].item()
            else:
                scalar_reward = rewards.item()
            
            if hidden_states is not None and self.config.scoring_strategy == 'per_token':
                per_token_rewards = self.model.reward_head(hidden_states).squeeze(-1)
                per_token_rewards = per_token_rewards[0].cpu().numpy()
                return scalar_reward, per_token_rewards
            
            return scalar_reward, None
    
    def score(
        self,
        prompt: str,
        response: str,
        return_per_token: bool = False
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        对单个 (prompt, response) 对进行评分
        
        Args:
            prompt: 输入提示
            response: 响应文本
            return_per_token: 是否返回每个token的奖励
            
        Returns:
            评分结果字典
        """
        encoded = self._encode_sequence(prompt, response)
        
        scalar_reward, per_token_rewards = self._score_sequence(
            encoded['input_ids'],
            encoded['attention_mask'],
            encoded['prompt_len']
        )
        
        result = {
            'scalar_reward': scalar_reward,
            'prompt': prompt,
            'response': response,
            'prompt_len': encoded['prompt_len'],
            'response_len': len(self.tokenizer.encode(response, add_special_tokens=False))
        }
        
        if return_per_token and per_token_rewards is not None:
            result['per_token_rewards'] = per_token_rewards
            result['response_rewards'] = per_token_rewards[encoded['prompt_len']:]
        
        return result
    
    def score_pair(
        self,
        prompt: str,
        chosen: str,
        rejected: str
    ) -> Dict[str, Any]:
        """
        对偏好对进行评分
        
        Args:
            prompt: 输入提示
            chosen: 被选中的响应
            rejected: 被拒绝的响应
            
        Returns:
            包含两个响应评分的字典
        """
        chosen_result = self.score(prompt, chosen)
        rejected_result = self.score(prompt, rejected)
        
        return {
            'prompt': prompt,
            'chosen': {
                'response': chosen,
                'reward': chosen_result['scalar_reward'],
                'length': chosen_result['response_len']
            },
            'rejected': {
                'response': rejected,
                'reward': rejected_result['scalar_reward'],
                'length': rejected_result['response_len']
            },
            'reward_diff': chosen_result['scalar_reward'] - rejected_result['scalar_reward'],
            'chosen_is_better': chosen_result['scalar_reward'] > rejected_result['scalar_reward']
        }
    
    def score_batch(
        self,
        items: List[Dict[str, str]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        批量评分
        
        Args:
            items: 包含 'prompt' 和 'response' 字段的字典列表
            show_progress: 是否显示进度条
            
        Returns:
            评分结果列表
        """
        results = []
        iterator = tqdm(items, desc="Scoring") if show_progress else items
        
        for item in iterator:
            result = self.score(item['prompt'], item['response'])
            result['metadata'] = item.get('metadata', {})
            results.append(result)
        
        return results
    
    def score_pairs_batch(
        self,
        pairs: List[Dict[str, str]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        批量偏好对评分
        
        Args:
            pairs: 包含 'prompt', 'chosen', 'rejected' 字段的字典列表
            show_progress: 是否显示进度条
            
        Returns:
            评分结果列表
        """
        results = []
        iterator = tqdm(pairs, desc="Scoring pairs") if show_progress else pairs
        
        for pair in iterator:
            result = self.score_pair(pair['prompt'], pair['chosen'], pair['rejected'])
            result['metadata'] = pair.get('metadata', {})
            results.append(result)
        
        return results
    
    def rank_responses(
        self,
        prompt: str,
        responses: List[str],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        对多个响应进行排序
        
        Args:
            prompt: 输入提示
            responses: 响应列表
            top_k: 返回前k个，None表示返回全部
            
        Returns:
            按奖励分数排序的响应列表
        """
        scored_responses = []
        
        for resp in responses:
            result = self.score(prompt, resp)
            scored_responses.append({
                'response': resp,
                'reward': result['scalar_reward'],
                'length': result['response_len']
            })
        
        scored_responses.sort(key=lambda x: x['reward'], reverse=True)
        
        if top_k is not None:
            scored_responses = scored_responses[:top_k]
        
        for i, item in enumerate(scored_responses):
            item['rank'] = i + 1
        
        return scored_responses


class EnsembleRewardScorer:
    """
    集成奖励评分器
    
    使用多个奖励模型进行评分，结果取平均或加权平均。
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        tokenizer,
        weights: Optional[List[float]] = None,
        config: Optional[ScoringConfig] = None
    ):
        self.models = models
        self.tokenizer = tokenizer
        self.config = config or ScoringConfig()
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        self.scorers = [
            RewardScorer(model, tokenizer, self.config)
            for model in models
        ]
    
    def score(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        使用集成模型进行评分
        """
        scores = []
        per_token_all = []
        
        for scorer in self.scorers:
            result = scorer.score(prompt, response, return_per_token=True)
            scores.append(result['scalar_reward'])
            if 'per_token_rewards' in result:
                per_token_all.append(result['per_token_rewards'])
        
        weighted_score = sum(s * w for s, w in zip(scores, self.weights))
        
        result = {
            'scalar_reward': weighted_score,
            'individual_scores': scores,
            'weights': self.weights,
            'prompt': prompt,
            'response': response
        }
        
        if per_token_all:
            per_token_avg = np.average(per_token_all, axis=0, weights=self.weights)
            result['per_token_rewards'] = per_token_avg
        
        return result
    
    def score_pair(self, prompt: str, chosen: str, rejected: str) -> Dict[str, Any]:
        """
        集成偏好对评分
        """
        chosen_result = self.score(prompt, chosen)
        rejected_result = self.score(prompt, rejected)
        
        return {
            'prompt': prompt,
            'chosen_reward': chosen_result['scalar_reward'],
            'rejected_reward': rejected_result['scalar_reward'],
            'reward_diff': chosen_result['scalar_reward'] - rejected_result['scalar_reward'],
            'chosen_is_better': chosen_result['scalar_reward'] > rejected_result['scalar_reward'],
            'individual_results': {
                'chosen': chosen_result['individual_scores'],
                'rejected': rejected_result['individual_scores']
            }
        }


class RewardCalibrator:
    """
    奖励分数校准器
    
    用于校准原始奖励分数，使其符合特定分布或具有可解释性。
    """
    
    def __init__(
        self,
        reference_scores: Optional[List[float]] = None,
        target_mean: float = 0.0,
        target_std: float = 1.0
    ):
        self.reference_scores = reference_scores
        self.target_mean = target_mean
        self.target_std = target_std
        self.reference_mean = None
        self.reference_std = None
        
        if reference_scores is not None and len(reference_scores) > 0:
            self._fit(reference_scores)
    
    def _fit(self, scores: List[float]):
        """根据参考分数拟合校准参数"""
        scores_arr = np.array(scores)
        self.reference_mean = np.mean(scores_arr)
        self.reference_std = np.std(scores_arr) + 1e-8
    
    def calibrate(self, score: float) -> float:
        """校准单个分数"""
        if self.reference_mean is None:
            return score
        
        z_score = (score - self.reference_mean) / self.reference_std
        calibrated = z_score * self.target_std + self.target_mean
        
        return calibrated
    
    def calibrate_batch(self, scores: List[float]) -> List[float]:
        """批量校准分数"""
        return [self.calibrate(s) for s in scores]


class RewardHistogramAnalyzer:
    """
    奖励分布分析器
    
    分析奖励分数的分布，检测异常值，生成统计报告。
    """
    
    def __init__(self):
        self.scores_history = []
        self.pairs_history = []
    
    def add_scores(self, scores: List[float], metadata: Optional[List[Dict]] = None):
        """添加分数记录"""
        for i, score in enumerate(scores):
            self.scores_history.append({
                'score': score,
                'metadata': metadata[i] if metadata else {}
            })
    
    def add_pair_result(self, result: Dict[str, Any]):
        """添加偏好对结果"""
        self.pairs_history.append(result)
    
    def analyze(self) -> Dict[str, Any]:
        """分析奖励分布"""
        if not self.scores_history:
            return {}
        
        scores = np.array([s['score'] for s in self.scores_history])
        
        analysis = {
            'count': len(scores),
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores)),
            'q25': float(np.percentile(scores, 25)),
            'q75': float(np.percentile(scores, 75)),
            'outliers': self._detect_outliers(scores).tolist()
        }
        
        if self.pairs_history:
            chosen_scores = [p['chosen']['reward'] for p in self.pairs_history]
            rejected_scores = [p['rejected']['reward'] for p in self.pairs_history]
            diffs = np.array([p['reward_diff'] for p in self.pairs_history])
            
            analysis['pair_stats'] = {
                'num_pairs': len(self.pairs_history),
                'chosen_mean': float(np.mean(chosen_scores)),
                'rejected_mean': float(np.mean(rejected_scores)),
                'diff_mean': float(np.mean(diffs)),
                'diff_std': float(np.std(diffs)),
                'agreement_rate': np.mean([p['chosen_is_better'] for p in self.pairs_history])
            }
        
        return analysis
    
    def _detect_outliers(self, scores: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """检测异常值（基于Z-score）"""
        z_scores = np.abs((scores - np.mean(scores)) / (np.std(scores) + 1e-8))
        return scores[z_scores > threshold]
    
    def save_report(self, output_path: str):
        """保存分析报告"""
        analysis = self.analyze()
        
        report = {
            'analysis': analysis,
            'total_records': len(self.scores_history),
            'pair_records': len(self.pairs_history)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved reward analysis report to {output_path}")


class BatchRewardScorer:
    """
    高效批量评分器
    
    使用批处理提高评分效率。
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: Optional[ScoringConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or ScoringConfig()
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model.to(self.device)
        self.model.eval()
    
    def _collate_batch(
        self,
        items: List[Dict[str, str]],
        pad_token_id: int = 0
    ) -> Dict[str, torch.Tensor]:
        """整理批次数据"""
        max_len = 0
        processed = []
        
        for item in items:
            prompt_tokens = self.tokenizer.encode(item['prompt'], add_special_tokens=False)
            response_tokens = self.tokenizer.encode(item['response'], add_special_tokens=False)
            
            sep_id = getattr(self.tokenizer, 'sep_token_id', 151643) or 151643
            bos_id = getattr(self.tokenizer, 'bos_token_id', 151643) or 151643
            
            full_tokens = [bos_id] + prompt_tokens + [sep_id] + response_tokens
            max_len = max(max_len, len(full_tokens))
            processed.append(full_tokens)
        
        batch_input_ids = []
        batch_attention_mask = []
        prompt_lens = []
        
        for tokens in processed:
            prompt_lens.append(tokens.index(sep_id) + 1 if sep_id in tokens else len(tokens))
            
            padded = tokens + [pad_token_id] * (max_len - len(tokens))
            batch_input_ids.append(padded)
            batch_attention_mask.append([1] * len(tokens) + [0] * (max_len - len(tokens)))
        
        return {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long),
            'prompt_lens': prompt_lens
        }
    
    def score_batch(
        self,
        items: List[Dict[str, str]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """批量评分"""
        results = []
        
        iterator = tqdm(
            range(0, len(items), self.config.batch_size),
            desc="Batch scoring"
        ) if show_progress else range(0, len(items), self.config.batch_size)
        
        for i in iterator:
            batch_items = items[i:i + self.config.batch_size]
            batch = self._collate_batch(batch_items)
            
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.config.use_amp):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_hidden=True
                )
                
                if isinstance(outputs, tuple):
                    hidden_states = outputs[-1]
                else:
                    hidden_states = outputs
                
                rewards = self.model.reward_head(hidden_states[:, -1, :]).squeeze(-1)
                
                for j, (item, reward) in enumerate(zip(batch_items, rewards)):
                    results.append({
                        'prompt': item['prompt'],
                        'response': item['response'],
                        'scalar_reward': reward.item(),
                        'metadata': item.get('metadata', {})
                    })
        
        return results
    
    def score_pairs_batch(
        self,
        pairs: List[Dict[str, str]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """批量偏好对评分"""
        chosen_items = [
            {'prompt': p['prompt'], 'response': p['chosen'], 'metadata': p.get('metadata', {})}
            for p in pairs
        ]
        rejected_items = [
            {'prompt': p['prompt'], 'response': p['rejected'], 'metadata': p.get('metadata', {})}
            for p in pairs
        ]
        
        chosen_results = self.score_batch(chosen_items, show_progress=False)
        rejected_results = self.score_batch(rejected_items, show_progress=False)
        
        results = []
        for pair, chosen, rejected in zip(pairs, chosen_results, rejected_results):
            results.append({
                'prompt': pair['prompt'],
                'chosen': {
                    'response': pair['chosen'],
                    'reward': chosen['scalar_reward']
                },
                'rejected': {
                    'response': pair['rejected'],
                    'reward': rejected['scalar_reward']
                },
                'reward_diff': chosen['scalar_reward'] - rejected['scalar_reward'],
                'chosen_is_better': chosen['scalar_reward'] > rejected['scalar_reward'],
                'metadata': pair.get('metadata', {})
            })
        
        return results


def save_scores_to_file(
    scores: List[Dict[str, Any]],
    output_path: str,
    format: str = 'jsonl'
):
    """
    保存评分结果到文件
    
    Args:
        scores: 评分结果列表
        output_path: 输出文件路径
        format: 输出格式 ('jsonl' 或 'json')
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    if format == 'jsonl':
        with open(output_path, 'w', encoding='utf-8') as f:
            for score in scores:
                f.write(json.dumps(score, ensure_ascii=False) + '\n')
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(scores)} scores to {output_path}")


def load_scores_from_file(input_path: str) -> List[Dict[str, Any]]:
    """从文件加载评分结果"""
    scores = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            scores.append(json.loads(line.strip()))
    
    logger.info(f"Loaded {len(scores)} scores from {input_path}")
    return scores


if __name__ == '__main__':
    logger.info("Reward Scoring Module")
    logger.info("Import this module to use the RewardScorer class")
