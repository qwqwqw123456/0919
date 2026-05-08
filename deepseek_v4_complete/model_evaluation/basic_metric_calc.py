"""
基础评估指标计算模块

提供语言模型评估所需的基础指标计算功能，包括：
- PPL (Perplexity) - 困惑度
- BLEU - BLEU评分
- ROUGE - ROUGE系列评分
- METEOR - METEOR评分
- chrF - 字符级F-score
- Exact Match - 精确匹配
- F1 Score - F1分数
"""

import math
import re
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple


class BasicMetricCalc:
    """基础评估指标计算类"""

    def __init__(self, n_workers: int = 4):
        """
        初始化基础指标计算器

        Args:
            n_workers: 并行计算的工作线程数
        """
        self.n_workers = n_workers
        self._bleu_cache = {}
        self._rouge_cache = {}

    def perplexity(self, loss: float) -> float:
        """
        计算困惑度 (Perplexity)

        困惑度是语言模型质量的常用指标，值越低表示模型越好

        Args:
            loss: 平均对数损失 (cross-entropy loss)

        Returns:
            float: 困惑度值

        Example:
            >>> calc = BasicMetricCalc()
            >>> ppl = calc.perplexity(1.5)
            >>> print(f"PPL: {ppl:.2f}")
        """
        if loss is None or loss < 0:
            return float('inf')
        return math.exp(loss)

    def sentence_perplexity(self, log_probs: List[float]) -> float:
        """
        计算句子级别的困惑度

        Args:
            log_probs: 每个token的对数概率列表

        Returns:
            float: 句子困惑度
        """
        if not log_probs:
            return float('inf')
        n = len(log_probs)
        total_log_prob = sum(log_probs)
        avg_log_prob = total_log_prob / n
        return math.exp(-avg_log_prob)

    def calculate_bleu(
        self,
        reference: str,
        candidate: str,
        n_gram: int = 4,
        smooth: bool = False
    ) -> float:
        """
        计算BLEU评分 (Bilingual Evaluation Understudy)

        BLEU是一种用于评估生成文本质量的指标，通过计算n-gram精确度来衡量
        候选文本与参考文本的相似程度

        Args:
            reference: 参考文本
            candidate: 候选文本（模型生成）
            n_gram: 最大n-gram阶数，默认4
            smooth: 是否使用平滑处理

        Returns:
            float: BLEU分数，范围0-1

        Example:
            >>> calc = BasicMetricCalc()
            >>> score = calc.calculate_bleu("The cat is on the mat", "The cat sits on the mat")
            >>> print(f"BLEU: {score:.4f}")
        """
        reference = reference.lower().split()
        candidate = candidate.lower().split()

        if not candidate:
            return 0.0

        scores = []
        precision_denominators = []
        clip_counts = []

        for i in range(1, n_gram + 1):
            ref_ngrams = self._get_ngrams(reference, i)
            cand_ngrams = self._get_ngrams(candidate, i)

            if not cand_ngrams:
                return 0.0

            overlap = sum((cand_ngrams & ref_ngrams).values())
            total = sum(cand_ngrams.values())

            if smooth:
                overlap += 1
                total += 1

            if total > 0:
                precision = overlap / total
                scores.append(precision)
                precision_denominators.append(total)
                clip_counts.append(overlap)

        if not scores:
            return 0.0

        if any(s == 0 for s in scores):
            return 0.0

        brevity_penalty = self._calculate_brevity_penalty(reference, candidate)

        log_precision = sum(math.log(s) for s in scores) / len(scores)
        bleu = brevity_penalty * math.exp(log_precision)

        return min(bleu, 1.0)

    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """生成n-gram计数"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return Counter(ngrams)

    def _calculate_brevity_penalty(self, reference: List[str], candidate: List[str]) -> float:
        """计算简短惩罚因子"""
        ref_len = len(reference)
        cand_len = len(candidate)

        if cand_len >= ref_len:
            return 1.0

        bp = 1.0 - (ref_len - cand_len) / ref_len
        return math.exp(bp)

    def calculate_bleu_multi_references(
        self,
        references: List[str],
        candidate: str,
        n_gram: int = 4
    ) -> float:
        """
        计算多参考BLEU评分

        Args:
            references: 参考文本列表
            candidate: 候选文本
            n_gram: 最大n-gram阶数

        Returns:
            float: BLEU分数
        """
        scores = []
        for ref in references:
            score = self.calculate_bleu(ref, candidate, n_gram)
            scores.append(score)
        return max(scores) if scores else 0.0

    def calculate_rouge(
        self,
        reference: str,
        candidate: str,
        rouge_type: str = 'rouge-l'
    ) -> Dict[str, float]:
        """
        计算ROUGE评分 (Recall-Oriented Understudy for Gisting Evaluation)

        ROUGE是一组用于评估文本摘要的指标集合

        Args:
            reference: 参考文本
            candidate: 候选文本
            rouge_type: ROUGE类型
                - 'rouge-1': unigram重叠
                - 'rouge-2': bigram重叠
                - 'rouge-l': 最长公共子序列
                - 'rouge-w': 加权最长公共子序列

        Returns:
            Dict: 包含precision, recall, f_score的字典

        Example:
            >>> calc = BasicMetricCalc()
            >>> result = calc.calculate_rouge("The cat is on the mat", "The cat sits on the mat")
            >>> print(f"ROUGE-L: {result['f_score']:.4f}")
        """
        reference = reference.lower().split()
        candidate = candidate.lower().split()

        if not reference or not candidate:
            return {'precision': 0.0, 'recall': 0.0, 'f_score': 0.0}

        if rouge_type == 'rouge-1':
            return self._rouge_n(reference, candidate, 1)
        elif rouge_type == 'rouge-2':
            return self._rouge_n(reference, candidate, 2)
        elif rouge_type == 'rouge-l':
            return self._rouge_l(reference, candidate)
        else:
            return self._rouge_l(reference, candidate)

    def _rouge_n(self, reference: List[str], candidate: List[str], n: int) -> Dict[str, float]:
        """计算ROUGE-N分数"""
        ref_ngrams = self._get_ngrams(reference, n)
        cand_ngrams = self._get_ngrams(candidate, n)

        overlap_ngrams = ref_ngrams & cand_ngrams
        overlap_count = sum(overlap_ngrams.values())
        ref_count = sum(ref_ngrams.values())
        cand_count = sum(cand_ngrams.values())

        if ref_count == 0:
            recall = 0.0
        else:
            recall = overlap_count / ref_count

        if cand_count == 0:
            precision = 0.0
        else:
            precision = overlap_count / cand_count

        f_score = self._f_beta_score(precision, recall, beta=1)

        return {'precision': precision, 'recall': recall, 'f_score': f_score}

    def _rouge_l(self, reference: List[str], candidate: List[str]) -> Dict[str, float]:
        """计算ROUGE-L分数（基于最长公共子序列）"""
        lcs_length = self._lcs_length(reference, candidate)

        ref_len = len(reference)
        cand_len = len(candidate)

        if ref_len == 0 or cand_len == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f_score': 0.0}

        recall = lcs_length / ref_len
        precision = lcs_length / cand_len

        f_score = self._f_beta_score(precision, recall, beta=1)

        return {'precision': precision, 'recall': recall, 'f_score': f_score}

    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """计算最长公共子序列长度"""
        m, n = len(seq1), len(seq2)
        if m == 0 or n == 0:
            return 0

        dp = [[0] * (n + 1) for _ in range(2)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i % 2][j] = dp[(i-1) % 2][j-1] + 1
                else:
                    dp[i % 2][j] = max(dp[(i-1) % 2][j], dp[i % 2][j-1])

        return dp[m % 2][n]

    def calculate_rouge_batch(
        self,
        references: List[str],
        candidates: List[str],
        rouge_types: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        批量计算ROUGE分数

        Args:
            references: 参考文本列表
            candidates: 候选文本列表
            rouge_types: ROUGE类型列表

        Returns:
            Dict: 各类型的平均分数
        """
        if rouge_types is None:
            rouge_types = ['rouge-1', 'rouge-2', 'rouge-l']

        all_scores = {rt: {'precision': [], 'recall': [], 'f_score': []} for rt in rouge_types}

        for ref, cand in zip(references, candidates):
            for rt in rouge_types:
                scores = self.calculate_rouge(ref, cand, rt)
                all_scores[rt]['precision'].append(scores['precision'])
                all_scores[rt]['recall'].append(scores['recall'])
                all_scores[rt]['f_score'].append(scores['f_score'])

        result = {}
        for rt in rouge_types:
            result[rt] = {
                'precision': sum(all_scores[rt]['precision']) / len(all_scores[rt]['precision']),
                'recall': sum(all_scores[rt]['recall']) / len(all_scores[rt]['recall']),
                'f_score': sum(all_scores[rt]['f_score']) / len(all_scores[rt]['f_score'])
            }

        return result

    def _f_beta_score(self, precision: float, recall: float, beta: float = 1.0) -> float:
        """计算F-beta分数"""
        if precision + recall == 0:
            return 0.0
        beta_squared = beta ** 2
        return (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)

    def calculate_meteor(
        self,
        reference: str,
        candidate: str,
        alpha: float = 0.9,
        beta: float = 3.0,
        gamma: float = 0.5
    ) -> float:
        """
        计算METEOR评分

        METEOR (Metric for Evaluation of Translation with Explicit ORdering)
        是一种考虑词干提取和同义词匹配的对齐评估指标

        Args:
            reference: 参考文本
            candidate: 候选文本
            alpha: 惩罚参数
            beta: 惩罚参数
            gamma: 惩罚参数

        Returns:
            float: METEOR分数
        """
        reference = reference.lower().split()
        candidate = candidate.lower().split()

        if not candidate or not reference:
            return 0.0

        unigram_overlap = self._meteor_unigram_overlap(reference, candidate)
        fragmentation_penalty = self._fragmentation_penalty(reference, candidate, beta, gamma)

        meteor = (1 - gamma * fragmentation_penalty) * unigram_overlap * (1 - alpha * fragmentation_penalty)

        return max(0.0, min(1.0, meteor))

    def _meteor_unigram_overlap(self, reference: List[str], candidate: List[str]) -> float:
        """计算METEOR unigram重叠度"""
        ref_counter = Counter(reference)
        cand_counter = Counter(candidate)

        matches = sum((ref_counter & cand_counter).values())
        total_ref = len(reference)

        return matches / total_ref if total_ref > 0 else 0.0

    def _fragmentation_penalty(self, reference: List[str], candidate: List[str], beta: float, gamma: float) -> float:
        """计算碎片化惩罚"""
        matches = self._count_chunk_matches(reference, candidate)
        if matches == 0:
            return 1.0

        cand_len = len(candidate)
        penalty = gamma * (matches / cand_len) ** beta if cand_len > 0 else 1.0
        return min(1.0, penalty)

    def _count_chunk_matches(self, reference: List[str], candidate: List[str]) -> int:
        """计算块匹配数"""
        aligned = self._meteor_align(reference, candidate)
        if not aligned:
            return 0

        chunks = 0
        current_chunk = []

        for ref_idx, cand_idx in aligned:
            if not current_chunk:
                current_chunk.append((ref_idx, cand_idx))
            else:
                if ref_idx == current_chunk[-1][0] + 1 and cand_idx == current_chunk[-1][1] + 1:
                    current_chunk.append((ref_idx, cand_idx))
                else:
                    if len(current_chunk) > 0:
                        chunks += 1
                    current_chunk = [(ref_idx, cand_idx)]

        if len(current_chunk) > 0:
            chunks += 1

        return chunks

    def _meteor_align(self, reference: List[str], candidate: List[str]) -> List[Tuple[int, int]]:
        """METEOR对齐"""
        aligned = []
        ref_used = [False] * len(reference)
        cand_used = [False] * len(candidate)

        ref_counter = Counter(reference)
        cand_counter = Counter(candidate)

        for i, ref_word in enumerate(reference):
            for j, cand_word in enumerate(candidate):
                if not ref_used[i] and not cand_used[j]:
                    if ref_word == cand_word:
                        aligned.append((i, j))
                        ref_used[i] = True
                        cand_used[j] = True
                        ref_counter[ref_word] -= 1
                        cand_counter[cand_word] -= 1
                        break

        return aligned

    def calculate_chrf(
        self,
        reference: str,
        candidate: str,
        n_gram_order: int = 6,
        beta: float = 2.0
    ) -> float:
        """
        计算chrF分数 (Character n-gram F-score)

        基于字符级n-gram的评估指标，对词序变化不敏感

        Args:
            reference: 参考文本
            candidate: 候选文本
            n_gram_order: 字符n-gram阶数
            beta: F-beta的beta参数

        Returns:
            float: chrF分数
        """
        ref_char_ngrams = self._get_char_ngrams(reference, n_gram_order)
        cand_char_ngrams = self._get_char_ngrams(candidate, n_gram_order)

        overlap = sum((ref_char_ngrams & cand_char_ngrams).values())
        ref_total = sum(ref_char_ngrams.values())
        cand_total = sum(cand_char_ngrams.values())

        if ref_total == 0 or cand_total == 0:
            return 0.0

        precision = overlap / cand_total
        recall = overlap / ref_total

        f_score = self._f_beta_score(precision, recall, beta)

        return f_score

    def _get_char_ngrams(self, text: str, n: int) -> Counter:
        """获取字符级n-gram"""
        text = text.lower()
        ngrams = []
        for i in range(len(text) - n + 1):
            ngrams.append(text[i:i+n])
        return Counter(ngrams)

    def exact_match(self, reference: str, candidate: str) -> float:
        """
        计算精确匹配分数

        Args:
            reference: 参考文本
            candidate: 候选文本

        Returns:
            float: 1.0如果完全匹配，否则0.0
        """
        ref_normalized = self._normalize_text(reference)
        cand_normalized = self._normalize_text(candidate)
        return 1.0 if ref_normalized == cand_normalized else 0.0

    def _normalize_text(self, text: str) -> str:
        """标准化文本"""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def f1_score(
        self,
        reference: str,
        candidate: str,
        average: str = 'word'
    ) -> float:
        """
        计算F1分数

        Args:
            reference: 参考文本
            candidate: 候选文本
            average: 计算方式 ('word' 或 'char')

        Returns:
            float: F1分数
        """
        if average == 'word':
            ref_tokens = set(reference.lower().split())
            cand_tokens = set(candidate.lower().split())
        else:
            ref_tokens = set(reference.lower())
            cand_tokens = set(candidate.lower())

        if not ref_tokens and not cand_tokens:
            return 1.0
        if not ref_tokens or not cand_tokens:
            return 0.0

        overlap = len(ref_tokens & cand_tokens)
        precision = overlap / len(cand_tokens) if cand_tokens else 0
        recall = overlap / len(ref_tokens) if ref_tokens else 0

        return self._f_beta_score(precision, recall, beta=1)

    def calculate_accuracy(
        self,
        predictions: List[Any],
        references: List[Any]
    ) -> float:
        """
        计算准确率

        Args:
            predictions: 预测结果列表
            references: 参考结果列表

        Returns:
            float: 准确率 (0-1)
        """
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length")

        if not predictions:
            return 0.0

        correct = sum(1 for p, r in zip(predictions, references) if p == r)
        return correct / len(predictions)

    def token_accuracy(
        self,
        pred_tokens: List[str],
        ref_tokens: List[str]
    ) -> float:
        """
        计算token级别的准确率

        Args:
            pred_tokens: 预测的token列表
            ref_tokens: 参考的token列表

        Returns:
            float: Token准确率
        """
        if not ref_tokens:
            return 1.0 if not pred_tokens else 0.0

        min_len = min(len(pred_tokens), len(ref_tokens))
        matches = sum(1 for p, r in zip(pred_tokens[:min_len], ref_tokens[:min_len]) if p == r)

        max_len = max(len(pred_tokens), len(ref_tokens))
        return matches / max_len if max_len > 0 else 1.0

    def calculate_all_metrics(
        self,
        reference: str,
        candidate: str,
        include_metrics: List[str] = None
    ) -> Dict[str, float]:
        """
        计算所有评估指标

        Args:
            reference: 参考文本
            candidate: 候选文本
            include_metrics: 要计算的指标列表，None表示计算所有

        Returns:
            Dict: 所有指标及其分数
        """
        if include_metrics is None:
            include_metrics = ['bleu', 'rouge-1', 'rouge-2', 'rouge-l', 'meteor', 'chrf', 'exact_match', 'f1']

        results = {}

        if 'bleu' in include_metrics:
            results['bleu'] = self.calculate_bleu(reference, candidate)

        if 'rouge-1' in include_metrics:
            results['rouge-1'] = self.calculate_rouge(reference, candidate, 'rouge-1')['f_score']

        if 'rouge-2' in include_metrics:
            results['rouge-2'] = self.calculate_rouge(reference, candidate, 'rouge-2')['f_score']

        if 'rouge-l' in include_metrics:
            results['rouge-l'] = self.calculate_rouge(reference, candidate, 'rouge-l')['f_score']

        if 'meteor' in include_metrics:
            results['meteor'] = self.calculate_meteor(reference, candidate)

        if 'chrf' in include_metrics:
            results['chrf'] = self.calculate_chrf(reference, candidate)

        if 'exact_match' in include_metrics:
            results['exact_match'] = self.exact_match(reference, candidate)

        if 'f1' in include_metrics:
            results['f1'] = self.f1_score(reference, candidate)

        return results

    def batch_evaluate(
        self,
        references: List[str],
        candidates: List[str],
        metrics: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        批量评估多个文本对

        Args:
            references: 参考文本列表
            candidates: 候选文本列表
            metrics: 要计算的指标列表

        Returns:
            Dict: 各指标的平均分数
        """
        if len(references) != len(candidates):
            raise ValueError("references and candidates must have the same length")

        if not references:
            return {}

        all_results = []

        for ref, cand in zip(references, candidates):
            result = self.calculate_all_metrics(ref, cand, metrics)
            all_results.append(result)

        aggregated = {}
        for metric in all_results[0].keys():
            values = [r[metric] for r in all_results]
            aggregated[metric] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'std': self._standard_deviation(values)
            }

        return aggregated

    def _standard_deviation(self, values: List[float]) -> float:
        """计算标准差"""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)


def demo():
    """演示函数"""
    calc = BasicMetricCalc()

    reference = "The cat is sitting on the mat"
    candidate = "The cat is sitting on the mat"

    print("=" * 60)
    print("基础评估指标计算演示")
    print("=" * 60)

    print(f"\n参考文本: {reference}")
    print(f"候选文本: {candidate}")

    metrics = calc.calculate_all_metrics(reference, candidate)
    print("\n评估结果:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\n" + "-" * 60)

    reference2 = "The quick brown fox jumps over the lazy dog"
    candidate2 = "A fast brown fox leaps over the sleepy dog"

    print(f"\n参考文本: {reference2}")
    print(f"候选文本: {candidate2}")

    metrics2 = calc.calculate_all_metrics(reference2, candidate2)
    print("\n评估结果:")
    for metric, value in metrics2.items():
        print(f"  {metric}: {value:.4f}")

    print("\n" + "-" * 60)

    batch_refs = ["Hello world", "Good morning"]
    batch_cands = ["Hello there", "Good afternoon"]
    batch_results = calc.batch_evaluate(batch_refs, batch_cands)
    print("\n批量评估结果:")
    for metric, stats in batch_results.items():
        print(f"  {metric}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")


if __name__ == "__main__":
    demo()
