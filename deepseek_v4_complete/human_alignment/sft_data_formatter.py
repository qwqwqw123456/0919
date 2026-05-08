"""
SFT 数据格式化模块

SFT (Supervised Fine-Tuning) 数据格式化工具，用于将各种格式的训练数据
转换为模型可用的格式。

支持的数据格式:
1. 原始对话格式 (messages)
2. prompt-response 格式
3. 多轮对话格式
4. 带有系统提示的对话

特性:
1. 多种对话模板
2. 特殊 token 处理
3. 数据验证和清洗
4. 批量处理
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
import json
import os
import logging
from pathlib import Path
from tqdm import tqdm
import re
from collections import Counter
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SFTDataFormatterConfig:
    """SFT 数据格式化配置"""
    max_length: int = 2048
    max_prompt_length: int = 512
    max_response_length: int = 1536
    add_special_tokens: bool = True
    use_chat_template: bool = True
    system_message: Optional[str] = None
    separator: str = "\n"
    response_prefix: Optional[str] = None
    response_suffix: Optional[str] = None
    truncate_mode: str = 'middle'
    ignore_invalid: bool = True
    return_metadata: bool = False


class ConversationTemplate:
    """
    对话模板基类
    
    定义对话格式化为字符串的模板。
    """
    
    def __init__(
        self,
        system_template: str = "<|system|>\n{system_message}",
        user_template: str = "<|user|>\n{user_message}",
        assistant_template: str = "<|assistant|>\n{assistant_message}",
        separator: str = ""
    ):
        self.system_template = system_template
        self.user_template = user_template
        self.assistant_template = assistant_template
        self.separator = separator
    
    def format(self, messages: List[Dict[str, str]]) -> str:
        """
        格式化对话
        
        Args:
            messages: 消息列表，每条消息包含 'role' 和 'content' 字段
            
        Returns:
            格式化的字符串
        """
        parts = []
        
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if role == 'system':
                if self.system_template:
                    parts.append(self.system_template.format(system_message=content))
            elif role == 'user':
                parts.append(self.user_template.format(user_message=content))
            elif role == 'assistant':
                parts.append(self.assistant_template.format(assistant_message=content))
        
        return self.separator.join(parts)
    
    def get_roles(self) -> List[str]:
        """获取模板支持的 role 列表"""
        return ['system', 'user', 'assistant']


class ChatMLTemplate(ConversationTemplate):
    """ChatML 格式模板"""
    
    def __init__(self):
        super().__init__(
            system_template = "<|im_start|>system\n{system_message}<|im_end|>",
            user_template = "<|im_start|>user\n{user_message}<|im_end|>",
            assistant_template = "<|im_start|>assistant\n{assistant_message}<|im_end|>",
            separator = "\n"
        )


class Llama2Template(ConversationTemplate):
    """Llama2 格式模板"""
    
    def __init__(self):
        super().__init__(
            system_template = "[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
            user_template = "[INST] {user_message} [/INST]",
            assistant_template = " {assistant_message}</s>",
            separator = ""
        )


class QwenTemplate(ConversationTemplate):
    """Qwen 格式模板"""
    
    def __init__(self):
        super().__init__(
            system_template = "<|im_start|>system\n{system_message}<|im_end|>",
            user_template = "<|im_start|>user\n{user_message}<|im_end|>",
            assistant_template = "<|im_start|>assistant\n{assistant_message}<|im_end|>",
            separator = "\n"
        )


class SFTDataFormatter:
    """
    SFT 数据格式化器
    
    将各种格式的数据转换为 SFT 训练所需的格式。
    """
    
    TEMPLATES = {
        'chatml': ChatMLTemplate,
        'llama2': Llama2Template,
        'qwen': QwenTemplate,
        'default': ConversationTemplate
    }
    
    def __init__(
        self,
        tokenizer,
        config: Optional[SFTDataFormatterConfig] = None,
        template_name: str = 'chatml'
    ):
        self.tokenizer = tokenizer
        self.config = config or SFTDataFormatterConfig()
        
        if template_name in self.TEMPLATES:
            self.template = self.TEMPLATES[template_name]()
        else:
            self.template = ConversationTemplate()
        
        self.bos_token_id = getattr(tokenizer, 'bos_token_id', None) or 151643
        self.eos_token_id = getattr(tokenizer, 'eos_token_id', None) or 151643
        self.pad_token_id = getattr(tokenizer, 'pad_token_id', None) or 0
    
    def format_conversation(
        self,
        messages: List[Dict[str, str]],
        add_bos: bool = True,
        add_eos: bool = False
    ) -> str:
        """
        格式化对话为字符串
        
        Args:
            messages: 消息列表
            add_bos: 是否添加 bos token
            add_eos: 是否添加 eos token
            
        Returns:
            格式化的字符串
        """
        if self.config.use_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            if self.config.system_message and not any(m.get('role') == 'system' for m in messages):
                messages = [{'role': 'system', 'content': self.config.system_message}] + messages
            
            text = self.template.format(messages)
        
        if add_bos:
            text = self.tokenizer.bos_token + text
        if add_eos:
            text = text + self.tokenizer.eos_token
        
        return text
    
    def format_prompt_response(
        self,
        prompt: str,
        response: str,
        add_bos: bool = True,
        add_eos: bool = True
    ) -> str:
        """
        格式化 prompt-response 对
        
        Args:
            prompt: 输入提示
            response: 响应文本
            add_bos: 是否添加 bos token
            add_eos: 是否添加 eos token
            
        Returns:
            格式化的字符串
        """
        if self.config.response_prefix:
            response = self.config.response_prefix + response
        if self.config.response_suffix:
            response = response + self.config.response_suffix
        
        text = prompt + self.config.separator + response
        
        if add_bos:
            text = self.tokenizer.bos_token + text
        if add_eos:
            text = text + self.tokenizer.eos_token
        
        return text
    
    def encode_conversation(
        self,
        messages: List[Dict[str, str]],
        return_labels: bool = True,
        mask_response: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        编码对话为 tensors
        
        Args:
            messages: 消息列表
            return_labels: 是否返回 labels
            mask_response: 是否 mask 掉 prompt 部分只保留 response 的 labels
            
        Returns:
            包含 input_ids, attention_mask, labels 的字典
        """
        text = self.format_conversation(messages, add_bos=True, add_eos=False)
        
        encoding = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            max_length=self.config.max_length,
            truncation=True,
            return_tensors=None
        )
        
        input_ids = torch.tensor(encoding, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        if return_labels:
            if mask_response:
                prompt_len = self._estimate_prompt_len(messages)
                labels = input_ids.clone()
                labels[:prompt_len] = -100
            else:
                labels = input_ids.clone()
            
            result['labels'] = labels
        
        return result
    
    def encode_prompt_response(
        self,
        prompt: str,
        response: str,
        return_labels: bool = True,
        mask_prompt: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        编码 prompt-response 对
        
        Args:
            prompt: 输入提示
            response: 响应文本
            return_labels: 是否返回 labels
            mask_prompt: 是否 mask 掉 prompt 部分
            
        Returns:
            包含 input_ids, attention_mask, labels 的字典
        """
        text = self.format_prompt_response(prompt, response, add_bos=True, add_eos=True)
        
        encoding = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            max_length=self.config.max_length,
            truncation=True,
            return_tensors=None
        )
        
        input_ids = torch.tensor(encoding, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        if return_labels:
            if mask_prompt:
                prompt_encoding = self.tokenizer.encode(
                    prompt,
                    add_special_tokens=False,
                    max_length=self.config.max_prompt_length,
                    truncation=True
                )
                prompt_len = len(prompt_encoding)
                
                labels = input_ids.clone()
                labels[:prompt_len + 1] = -100
            else:
                labels = input_ids.clone()
            
            result['labels'] = labels
        
        return result
    
    def _estimate_prompt_len(self, messages: List[Dict[str, str]]) -> int:
        """估算 prompt 部分的长度"""
        prompt_text = self.template.format(messages[:-1]) if len(messages) > 1 else ""
        prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        return len(prompt_tokens) + 1
    
    def truncate_conversation(
        self,
        messages: List[Dict[str, str]],
        max_length: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        截断对话以适应最大长度
        
        Args:
            messages: 消息列表
            max_length: 最大长度
            
        Returns:
            截断后的消息列表
        """
        if max_length is None:
            max_length = self.config.max_length
        
        prompt_text = self.format_conversation(messages, add_bos=True, add_eos=False)
        prompt_len = len(self.tokenizer.encode(prompt_text, add_special_tokens=False))
        
        if prompt_len <= max_length:
            return messages
        
        if self.config.truncate_mode == 'head':
            return self._truncate_head(messages, max_length)
        elif self.config.truncate_mode == 'tail':
            return self._truncate_tail(messages, max_length)
        elif self.config.truncate_mode == 'middle':
            return self._truncate_middle(messages, max_length)
        else:
            return messages
    
    def _truncate_head(self, messages: List[Dict[str, str]], max_length: int) -> List[Dict[str, str]]:
        """从头截断对话"""
        while len(messages) > 1:
            messages = messages[1:]
            prompt_text = self.format_conversation(messages, add_bos=True, add_eos=False)
            prompt_len = len(self.tokenizer.encode(prompt_text, add_special_tokens=False))
            if prompt_len <= max_length:
                break
        return messages
    
    def _truncate_tail(self, messages: List[Dict[str, str]], max_length: int) -> List[Dict[str, str]]:
        """从尾截断对话"""
        while len(messages) > 1:
            messages = messages[:-1]
            prompt_text = self.format_conversation(messages, add_bos=True, add_eos=False)
            prompt_len = len(self.tokenizer.encode(prompt_text, add_special_tokens=False))
            if prompt_len <= max_length:
                break
        return messages
    
    def _truncate_middle(self, messages: List[Dict[str, str]], max_length: int) -> List[Dict[str, str]]:
        """从中间截断对话"""
        system_msg = []
        other_msgs = []
        
        for msg in messages:
            if msg.get('role') == 'system':
                system_msg.append(msg)
            else:
                other_msgs.append(msg)
        
        if len(other_msgs) <= 2:
            return messages
        
        other_msgs = other_msgs[1:-1]
        
        result = system_msg + [other_msgs[0]] if other_msgs else system_msg
        result = result + [other_msgs[-1]] if len(other_msgs) > 1 else result
        
        prompt_text = self.format_conversation(result, add_bos=True, add_eos=False)
        prompt_len = len(self.tokenizer.encode(prompt_text, add_special_tokens=False))
        
        if prompt_len <= max_length:
            return result
        
        return self._truncate_head(messages, max_length)


class SFTDataset(Dataset):
    """
    SFT 训练数据集
    
    支持多种数据格式的加载和格式化。
    """
    
    def __init__(
        self,
        data_path: Union[str, List[Dict[str, Any]]],
        tokenizer,
        formatter: Optional[SFTDataFormatter] = None,
        config: Optional[SFTDataFormatterConfig] = None,
        split: str = 'train'
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.config = config or SFTDataFormatterConfig()
        
        if formatter is None:
            self.formatter = SFTDataFormatter(tokenizer, self.config)
        else:
            self.formatter = formatter
        
        self.examples = self._load_data()
        self.split = split
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """加载数据"""
        examples = []
        
        if isinstance(self.data_path, list):
            examples = self.data_path
        elif os.path.exists(self.data_path):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    examples.append(json.loads(line.strip()))
        else:
            raise ValueError(f"Invalid data_path: {self.data_path}")
        
        if self.config.ignore_invalid:
            examples = [ex for ex in examples if self._is_valid(ex)]
        
        logger.info(f"Loaded {len(examples)} examples from {self.data_path}")
        return examples
    
    def _is_valid(self, example: Dict[str, Any]) -> bool:
        """验证数据是否有效"""
        if 'messages' in example:
            messages = example['messages']
            if not messages:
                return False
            if not any(m.get('role') == 'assistant' for m in messages):
                return False
            return True
        elif 'prompt' in example and 'response' in example:
            return bool(example.get('prompt')) and bool(example.get('response'))
        return False
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        if 'messages' in example:
            messages = example['messages']
            if self.config.system_message and not any(m.get('role') == 'system' for m in messages):
                messages = [{'role': 'system', 'content': self.config.system_message}] + messages
            
            return self.formatter.encode_conversation(messages)
        else:
            prompt = example.get('prompt', '')
            response = example.get('response', example.get('chosen', ''))
            
            return self.formatter.encode_prompt_response(prompt, response)


def sft_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int = 0
) -> Dict[str, torch.Tensor]:
    """
    SFT 数据批次整理函数
    
    Args:
        batch: 批次数据列表
        pad_token_id: 填充 token ID
        
    Returns:
        整理后的批次字典
    """
    max_len = max(item['input_ids'].size(0) for item in batch)
    
    input_ids = []
    attention_mask = []
    labels = []
    
    for item in batch:
        seq_len = item['input_ids'].size(0)
        pad_len = max_len - seq_len
        
        padded_input_ids = torch.cat([
            item['input_ids'],
            torch.full((pad_len,), pad_token_id, dtype=torch.long)
        ])
        padded_attention_mask = torch.cat([
            item['attention_mask'],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        padded_labels = torch.cat([
            item['labels'],
            torch.full((pad_len,), -100, dtype=torch.long)
        ])
        
        input_ids.append(padded_input_ids)
        attention_mask.append(padded_attention_mask)
        labels.append(padded_labels)
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'labels': torch.stack(labels)
    }


class DataValidator:
    """
    数据验证器
    
    验证 SFT 数据的质量。
    """
    
    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 8192,
        min_response_length: int = 1,
        check_repetition: bool = True,
        repetition_threshold: float = 0.7
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.min_response_length = min_response_length
        self.check_repetition = check_repetition
        self.repetition_threshold = repetition_threshold
    
    def validate(self, example: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        验证单个数据样本
        
        Args:
            example: 数据样本
            
        Returns:
            (is_valid, error_message)
        """
        if 'messages' in example:
            return self._validate_messages(example['messages'])
        elif 'prompt' in example and 'response' in example:
            return self._validate_prompt_response(example['prompt'], example['response'])
        else:
            return False, "Invalid data format"
    
    def _validate_messages(self, messages: List[Dict[str, str]]) -> Tuple[bool, Optional[str]]:
        """验证消息格式"""
        if not messages:
            return False, "Empty messages"
        
        has_assistant = any(m.get('role') == 'assistant' for m in messages)
        if not has_assistant:
            return False, "No assistant message"
        
        total_length = sum(len(m.get('content', '')) for m in messages)
        if total_length < self.min_length:
            return False, f"Total length {total_length} < {self.min_length}"
        
        if total_length > self.max_length:
            return False, f"Total length {total_length} > {self.max_length}"
        
        assistant_msgs = [m for m in messages if m.get('role') == 'assistant']
        for msg in assistant_msgs:
            if len(msg.get('content', '')) < self.min_response_length:
                return False, f"Response too short: {len(msg.get('content', ''))}"
        
        if self.check_repetition:
            for msg in assistant_msgs:
                if self._has_excessive_repetition(msg.get('content', '')):
                    return False, "Excessive repetition detected"
        
        return True, None
    
    def _validate_prompt_response(
        self,
        prompt: str,
        response: str
    ) -> Tuple[bool, Optional[str]]:
        """验证 prompt-response 对"""
        if not prompt or not response:
            return False, "Empty prompt or response"
        
        total_length = len(prompt) + len(response)
        if total_length < self.min_length:
            return False, f"Total length too short"
        
        if total_length > self.max_length:
            return False, f"Total length too long"
        
        if len(response) < self.min_response_length:
            return False, f"Response too short"
        
        if self.check_repetition and self._has_excessive_repetition(response):
            return False, "Excessive repetition detected"
        
        return True, None
    
    def _has_excessive_repetition(self, text: str, n: int = 10) -> bool:
        """检查是否有过度重复"""
        if len(text) < n:
            return False
        
        ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
        if not ngrams:
            return False
        
        counter = Counter(ngrams)
        most_common_count = counter.most_common(1)[0][1]
        
        repetition_ratio = most_common_count / len(ngrams)
        return repetition_ratio > self.repetition_threshold


class DataAugmenter:
    """
    数据增强器
    
    对 SFT 数据进行增强。
    """
    
    def __init__(
        self,
        augment_system: bool = True,
        system_messages: Optional[List[str]] = None,
        shuffle_responses: bool = False,
        add_noise: bool = False,
        noise_prob: float = 0.05
    ):
        self.augment_system = augment_system
        self.system_messages = system_messages or []
        self.shuffle_responses = shuffle_responses
        self.add_noise = add_noise
        self.noise_prob = noise_prob
    
    def augment(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        增强单个数据样本
        
        Args:
            example: 数据样本
            
        Returns:
            增强后的样本列表
        """
        results = [example.copy()]
        
        if self.augment_system and self.system_messages:
            augmented = self._add_system_message(example)
            if augmented:
                results.append(augmented)
        
        if self.shuffle_responses:
            shuffled = self._shuffle_conversation(example)
            if shuffled:
                results.append(shuffled)
        
        if self.add_noise:
            noised = self._add_noise(example)
            results.append(noised)
        
        return results
    
    def _add_system_message(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """添加系统消息"""
        if 'messages' not in example:
            return None
        
        messages = example['messages']
        if messages and messages[0].get('role') == 'system':
            return None
        
        system_msg = random.choice(self.system_messages)
        new_example = example.copy()
        new_example['messages'] = [{'role': 'system', 'content': system_msg}] + messages
        
        return new_example
    
    def _shuffle_conversation(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """打乱多轮对话"""
        if 'messages' not in example:
            return None
        
        messages = example['messages']
        if len(messages) <= 3:
            return None
        
        new_example = example.copy()
        new_example['messages'] = messages[:1] + random.sample(messages[1:], len(messages) - 1)
        
        return new_example
    
    def _add_noise(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """添加噪声"""
        new_example = example.copy()
        
        if 'messages' in example:
            messages = []
            for msg in example['messages']:
                if msg.get('role') == 'assistant':
                    content = msg['content']
                    words = content.split()
                    for i in range(len(words)):
                        if random.random() < self.noise_prob:
                            words[i] = self._get_typo(words[i])
                    msg = msg.copy()
                    msg['content'] = ' '.join(words)
                messages.append(msg)
            new_example['messages'] = messages
        elif 'response' in example:
            words = new_example['response'].split()
            for i in range(len(words)):
                if random.random() < self.noise_prob:
                    words[i] = self._get_typo(words[i])
            new_example['response'] = ' '.join(words)
        
        return new_example
    
    def _get_typo(self, word: str) -> str:
        """生成拼写错误"""
        if len(word) < 3:
            return word
        
        idx = random.randint(1, len(word) - 2)
        return word[:idx] + word[idx + 1] + word[idx] + word[idx + 2:]


def convert_format(
    input_path: str,
    output_path: str,
    input_format: str = 'raw',
    output_format: str = 'sft',
    tokenizer = None
):
    """
    转换数据格式
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        input_format: 输入格式 ('raw', 'sharegpt', 'anthropic')
        output_format: 输出格式 ('sft', 'dpo', 'prompt_response')
        tokenizer: 分词器
    """
    examples = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    
    converted = []
    
    if input_format == 'sharegpt':
        for ex in examples:
            messages = []
            for turn in ex.get('conversations', []):
                role = 'assistant' if turn.get('from') == 'gpt' else 'user'
                messages.append({
                    'role': role,
                    'content': turn.get('value', '')
                })
            converted.append({'messages': messages})
    
    elif input_format == 'anthropic':
        for ex in examples:
            messages = []
            if 'system' in ex:
                messages.append({'role': 'system', 'content': ex['system']})
            messages.append({'role': 'user', 'content': ex['input']})
            messages.append({'role': 'assistant', 'content': ex['output']})
            converted.append({'messages': messages})
    
    else:
        converted = examples
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in converted:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    logger.info(f"Converted {len(converted)} examples from {input_format} to {output_format}")


def prepare_sft_data(
    input_path: str,
    output_path: str,
    tokenizer,
    config: Optional[SFTDataFormatterConfig] = None,
    validation: bool = True,
    augmentation: bool = False
) -> int:
    """
    准备 SFT 训练数据
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        tokenizer: 分词器
        config: 格式化配置
        validation: 是否验证数据
        augmentation: 是否增强数据
        
    Returns:
        处理后的样本数量
    """
    formatter = SFTDataFormatter(tokenizer, config)
    validator = DataValidator() if validation else None
    augmenter = DataAugmenter() if augmentation else None
    
    examples = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    
    processed = []
    
    for ex in tqdm(examples, desc="Processing data"):
        if validation and validator:
            is_valid, error = validator.validate(ex)
            if not is_valid:
                logger.debug(f"Skipping invalid example: {error}")
                continue
        
        if augmentation and augmenter:
            augmented = augmenter.augment(ex)
            processed.extend(augmented)
        else:
            processed.append(ex)
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in processed:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    logger.info(f"Prepared {len(processed)} SFT examples to {output_path}")
    return len(processed)


if __name__ == '__main__':
    logger.info("SFT Data Formatter Module")
    logger.info("Import this module to use the SFTDataFormatter class")
