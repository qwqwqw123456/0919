"""
Multi Modal Inference - 多模态推理模块
=====================================

该模块提供完整的多模态推理实现，包括推理配置、批处理推理、流式推理等功能。

主要组件：
- InferenceConfig: 推理配置
- MultiModalInference: 多模态推理器
- BatchInference: 批处理推理
- StreamingInference: 流式推理
- generate_multimodal: 多模态生成函数

特点：
- 支持多种推理模式
- 支持批处理和流式推理
- 支持动态批处理
- 支持GPU加速
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple, Union, Callable, Iterator
from dataclasses import dataclass, field
import threading
import queue
from collections import deque


@dataclass
class InferenceConfig:
    """
    推理配置

    参数:
        max_length: 最大生成长度
        min_length: 最小生成长度
        temperature: 采样温度
        top_p: Nucleus采样参数
        top_k: Top-k采样参数
        repetition_penalty: 重复惩罚
        length_penalty: 长度惩罚
        num_beams: Beam搜索数量
        early_stopping: 是否提前停止
        do_sample: 是否采样
        pad_token_id: 填充token ID
        eos_token_id: 结束token ID
        bos_token_id: 起始token ID
        use_cache: 是否使用KV缓存
        batch_size: 批处理大小
        device: 推理设备
        dtype: 数据类型
    """
    max_length: int = 2048
    min_length: int = 1
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    num_beams: int = 1
    early_stopping: bool = False
    do_sample: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 2
    bos_token_id: int = 1
    use_cache: bool = True
    batch_size: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float16"


class KVCache:
    """
    Key-Value缓存

    用于存储和复用注意力层的Key和Value
    """

    def __init__(self):
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor
    ):
        """更新指定层的KV缓存"""
        if layer_idx >= len(self.key_cache):
            self.key_cache.append(key)
            self.value_cache.append(value)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key], dim=1)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value], dim=1)

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定层的KV缓存"""
        if layer_idx < len(self.key_cache):
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        return None, None

    def clear(self):
        """清空缓存"""
        self.key_cache = []
        self.value_cache = []

    def get_all(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """获取所有层的KV缓存"""
        return self.key_cache, self.value_cache


class MultiModalInference:
    """
    多模态推理器

    负责多模态模型的推理过程
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[InferenceConfig] = None,
        tokenizer: Optional[Any] = None
    ):
        """
        初始化多模态推理器

        参数:
            model: 多模态模型
            config: 推理配置
            tokenizer: 分词器
        """
        self.model = model
        self.config = config or InferenceConfig()
        self.tokenizer = tokenizer

        self.model.eval()
        self.device = torch.device(self.config.device)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[List[torch.Tensor]] = None,
        text: Optional[str] = None,
        return_dict: bool = True,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        生成文本

        参数:
            input_ids: 输入token IDs
            images: 输入图像列表
            text: 输入文本（与input_ids二选一）
            return_dict: 是否返回字典格式
            **generation_kwargs: 其他生成参数

        返回:
            生成结果
        """
        if text is not None and self.tokenizer is not None:
            input_ids = self.tokenizer.encode(text, return_tensors='pt')

        if input_ids is not None:
            input_ids = input_ids.to(self.device)

        if images is not None:
            images = [img.to(self.device) if isinstance(img, torch.Tensor) else img for img in images]

        gen_config = {**vars(self.config), **generation_kwargs}

        with torch.no_grad():
            if self.config.num_beams > 1:
                output_ids = self._beam_search(
                    input_ids,
                    images,
                    **gen_config
                )
            else:
                output_ids = self._sample(
                    input_ids,
                    images,
                    **gen_config
                )

        if self.tokenizer is not None:
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        else:
            output_text = None

        if return_dict:
            return {
                'sequences': output_ids,
                'text': output_text,
                'num_generated_tokens': output_ids.shape[1] - (input_ids.shape[1] if input_ids is not None else 0)
            }
        return output_ids

    def _sample(
        self,
        input_ids: torch.Tensor,
        images: Optional[List[torch.Tensor]],
        **kwargs
    ) -> torch.Tensor:
        """采样生成"""
        max_length = kwargs.get('max_length', self.config.max_length)
        min_length = kwargs.get('min_length', self.config.min_length)
        temperature = kwargs.get('temperature', self.config.temperature)
        top_p = kwargs.get('top_p', self.config.top_p)
        top_k = kwargs.get('top_k', self.config.top_k)
        repetition_penalty = kwargs.get('repetition_penalty', self.config.repetition_penalty)
        use_cache = kwargs.get('use_cache', self.config.use_cache)

        kv_cache = KVCache() if use_cache else None
        past_key_values = None

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        max_new_tokens = max_length - seq_len

        generated = input_ids.clone()

        for step in range(max_new_tokens):
            if use_cache and past_key_values is not None:
                outputs = self.model(
                    input_ids=generated[:, -1:],
                    images=images,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
            else:
                outputs = self.model(
                    input_ids=generated,
                    images=images
                )
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values if hasattr(outputs, 'past_key_values') else None

            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for j, prev_token in enumerate(generated[i]):
                        if logits[i, prev_token] < 0:
                            logits[i, prev_token] *= repetition_penalty
                        else:
                            logits[i, prev_token] /= repetition_penalty

            if temperature != 1.0:
                logits /= temperature

            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
                logits[indices_to_remove] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=-1)

            if (next_tokens == self.config.eos_token_id).all():
                break

            if step >= min_length - 1 and (next_tokens == self.config.eos_token_id).all():
                break

        return generated

    def _beam_search(
        self,
        input_ids: torch.Tensor,
        images: Optional[List[torch.Tensor]],
        **kwargs
    ) -> torch.Tensor:
        """Beam搜索生成"""
        num_beams = kwargs.get('num_beams', self.config.num_beams)
        max_length = kwargs.get('max_length', self.config.max_length)
        length_penalty = kwargs.get('length_penalty', self.config.length_penalty)
        early_stopping = kwargs.get('early_stopping', self.config.early_stopping)

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        max_new_tokens = max_length - seq_len

        beam_scores = torch.zeros(batch_size, num_beams, device=self.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        generated = input_ids.unsqueeze(1).expand(batch_size, num_beams, -1)
        generated = generated.reshape(batch_size * num_beams, -1)

        is_first = True
        past_key_values = None

        for step in range(max_new_tokens):
            if isFirst:
                outputs = self.model(input_ids=input_ids, images=images)
                isFirst = False
            else:
                if past_key_values is not None:
                    outputs = self.model(
                        input_ids=next_tokens.unsqueeze(-1),
                        images=images,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                else:
                    outputs = self.model(
                        input_ids=next_tokens,
                        images=images
                    )

            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values if hasattr(outputs, 'past_key_values') else None

            log_probs = F.log_softmax(logits, dim=-1)

            beam_scores = beam_scores.unsqueeze(-1) + log_probs

            beam_scores = beam_scores.view(batch_size, num_beams * logits.shape[-1])

            next_tokens = torch.topk(beam_scores, k=num_beams * 2, dim=-1)[1]

            token_idx = next_tokens % logits.shape[-1]
            beam_idx = next_tokens // logits.shape[-1]

            beam_scores = beam_scores.view(batch_size, num_beams * logits.shape[-1])
            beam_scores = torch.gather(beam_scores, 1, next_tokens)

            generated = generated.view(batch_size, num_beams, -1)
            generated = generated.unsqueeze(-1).expand(-1, -1, -1, 1)
            generated = generated.reshape(batch_size * num_beams, -1)

            generated = torch.cat([generated[beam_idx], token_idx.unsqueeze(-1)], dim=-1)

            eos_mask = token_idx == self.config.eos_token_id
            if eos_mask.any():
                beam_scores[eos_mask] = -1e9

            if (beam_scores >= -1e9).all():
                break

        best_score = beam_scores.view(batch_size, num_beams).max(dim=-1)[1]
        best_sequence = generated.view(batch_size, num_beams, -1)[range(batch_size), best_score]

        return best_sequence

    def encode_inputs(
        self,
        text: Optional[str] = None,
        images: Optional[List] = None,
        audio: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        编码输入

        参数:
            text: 输入文本
            images: 输入图像
            audio: 输入音频

        返回:
            编码后的输入
        """
        inputs = {}

        if text is not None and self.tokenizer is not None:
            text_inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True
            )
            inputs['input_ids'] = text_inputs['input_ids'].to(self.device)
            inputs['attention_mask'] = text_inputs['attention_mask'].to(self.device)

        if images is not None:
            inputs['images'] = [img.to(self.device) if isinstance(img, torch.Tensor) else img for img in images]

        if audio is not None:
            inputs['audio'] = audio.to(self.device)

        return inputs

    def get_model(self) -> nn.Module:
        """获取模型"""
        return self.model

    def set_device(self, device: str):
        """设置设备"""
        self.device = torch.device(device)
        self.model = self.model.to(self.device)


class BatchInference:
    """
    批处理推理器

    支持动态批处理和最大吞吐量优化
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[InferenceConfig] = None,
        tokenizer: Optional[Any] = None,
        max_batch_size: int = 32
    ):
        self.model = model
        self.config = config or InferenceConfig()
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size

        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_thread = None
        self.running = False

        self.model.eval()

    def start(self):
        """启动批处理工作线程"""
        if self.running:
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def stop(self):
        """停止批处理工作线程"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)

    def _worker_loop(self):
        """批处理工作线程主循环"""
        while self.running:
            try:
                batch = self._collect_batch(timeout=0.1)

                if batch:
                    results = self._process_batch(batch)
                    for result, callback in zip(results, batch):
                        if callback:
                            callback(result)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Batch inference error: {e}")

    def _collect_batch(self, timeout: float) -> List[Tuple]:
        """收集一批请求"""
        batch = []
        collected = 0

        while collected < self.max_batch_size:
            try:
                item = self.request_queue.get(timeout=timeout)
                batch.append(item)
                collected += 1
            except queue.Empty:
                break

        return batch

    def _process_batch(self, batch: List[Tuple]) -> List[Dict]:
        """处理一批请求"""
        inputs = [item[0] for item in batch]
        callbacks = [item[1] if len(item) > 1 else None for item in batch]

        inference = MultiModalInference(self.model, self.config, self.tokenizer)

        results = []
        for input_data in inputs:
            result = inference.generate(**input_data)
            results.append(result)

        return results

    def add_request(
        self,
        input_data: Dict,
        callback: Optional[Callable] = None
    ):
        """
        添加推理请求

        参数:
            input_data: 输入数据
            callback: 回调函数
        """
        self.request_queue.put((input_data, callback))

    def get_result(self, timeout: float = None) -> Dict:
        """获取结果"""
        return self.result_queue.get(timeout=timeout)

    @torch.no_grad()
    def infer_batch(
        self,
        texts: List[str],
        images: Optional[List[List]] = None
    ) -> List[str]:
        """
        批量推理

        参数:
            texts: 文本列表
            images: 图像列表

        返回:
            生成文本列表
        """
        results = []

        for i in range(0, len(texts), self.max_batch_size):
            batch_texts = texts[i:i + self.max_batch_size]
            batch_images = images[i:i + self.max_batch_size] if images else None

            inference = MultiModalInference(self.model, self.config, self.tokenizer)

            for text in batch_texts:
                result = inference.generate(text=text)
                results.append(result['text'])

        return results


class StreamingInference:
    """
    流式推理器

    支持逐步输出生成结果
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[InferenceConfig] = None,
        tokenizer: Optional[Any] = None
    ):
        self.model = model
        self.config = config or InferenceConfig()
        self.tokenizer = tokenizer

        self.model.eval()

    def generate_stream(
        self,
        text: str,
        images: Optional[List] = None,
        callback: Optional[Callable] = None
    ) -> Iterator[str]:
        """
        流式生成

        参数:
            text: 输入文本
            images: 输入图像
            callback: 回调函数

        返回:
            生成文本的迭代器
        """
        if self.tokenizer:
            input_ids = self.tokenizer.encode(text, return_tensors='pt')
        else:
            input_ids = torch.tensor([[1]])

        device = self.config.device
        input_ids = input_ids.to(device)

        generated = input_ids.clone()
        past_key_values = None

        for step in range(self.config.max_length):
            if past_key_values is not None:
                outputs = self.model(
                    input_ids=generated[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True
                )
            else:
                outputs = self.model(input_ids=generated, images=images)

            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values if hasattr(outputs, 'past_key_values') else None

            if self.config.temperature != 1.0:
                logits /= self.config.temperature

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)

            if self.tokenizer:
                token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
            else:
                token_text = str(next_token[0].item())

            if callback:
                callback(token_text)

            yield token_text

            if next_token.item() == self.config.eos_token_id:
                break


def generate_multimodal(
    model: nn.Module,
    text: str,
    images: Optional[List] = None,
    tokenizer: Optional[Any] = None,
    **generation_kwargs
) -> str:
    """
    多模态生成函数

    参数:
        model: 多模态模型
        text: 输入文本
        images: 输入图像
        tokenizer: 分词器
        **generation_kwargs: 其他生成参数

    返回:
        生成的文本
    """
    config = InferenceConfig(**generation_kwargs)
    inference = MultiModalInference(model, config, tokenizer)

    result = inference.generate(text=text, images=images)

    return result['text']


def create_inference_engine(
    model: nn.Module,
    inference_type: str = "standard",
    tokenizer: Optional[Any] = None,
    **kwargs
) -> MultiModalInference:
    """
    创建推理引擎的工厂函数

    参数:
        model: 多模态模型
        inference_type: 推理类型 ('standard', 'batch', 'stream')
        tokenizer: 分词器
        **kwargs: 其他配置参数

    返回:
        推理引擎实例
    """
    if inference_type == "standard":
        config = InferenceConfig(**kwargs)
        return MultiModalInference(model, config, tokenizer)

    elif inference_type == "batch":
        config = InferenceConfig(**kwargs)
        max_batch_size = kwargs.get('max_batch_size', 32)
        return BatchInference(model, config, tokenizer, max_batch_size)

    elif inference_type == "stream":
        config = InferenceConfig(**kwargs)
        return StreamingInference(model, config, tokenizer)

    else:
        raise ValueError(f"Unknown inference type: {inference_type}")
