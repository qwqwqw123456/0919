import random
from typing import List, Dict, Any, Optional, Tuple
import json

class SampleBuilder:
    def __init__(self, tokenizer, max_seq_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def build_pretrain_sample(self, text: str, 
                             add_special_tokens: bool = True) -> Dict[str, List[int]]:
        tokens = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]
        
        labels = tokens[:]
        labels[0] = -100
        
        return {
            'input_ids': tokens,
            'labels': labels,
            'attention_mask': [1] * len(tokens)
        }
    
    def build_pair_sample(self, text_a: str, text_b: str, 
                         label: int = 0) -> Dict[str, Any]:
        tokens_a = self.tokenizer.encode(text_a, add_special_tokens=False)
        tokens_b = self.tokenizer.encode(text_b, add_special_tokens=False)
        
        max_available = self.max_seq_length - 3
        len_a = len(tokens_a)
        len_b = len(tokens_b)
        
        if len_a + len_b > max_available:
            if len_a > max_available // 2:
                tokens_a = tokens_a[:max_available // 2]
            if len_b > max_available - len(tokens_a):
                tokens_b = tokens_b[:max_available - len(tokens_a)]
        
        input_ids = [self.tokenizer.vocab['<s>']] + tokens_a + \
                    [self.tokenizer.vocab['</s>']] + tokens_b + \
                    [self.tokenizer.vocab['</s>']]
        
        return {
            'input_ids': input_ids,
            'token_type_ids': [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1),
            'attention_mask': [1] * len(input_ids),
            'label': label
        }
    
    def build_qa_sample(self, question: str, answer: str, 
                       context: Optional[str] = None) -> Dict[str, Any]:
        if context:
            text = f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"
        else:
            text = f"Question: {question}\nAnswer: {answer}"
        
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]
        
        return {
            'input_ids': tokens,
            'attention_mask': [1] * len(tokens),
            'question': question,
            'answer': answer,
            'context': context
        }
    
    def build_conversation_sample(self, conversations: List[Dict[str, str]],
                                 system_prompt: Optional[str] = None) -> Dict[str, Any]:
        text_parts = []
        
        if system_prompt:
            text_parts.append(f"<system>{system_prompt}</system>")
        
        for conv in conversations:
            role = conv.get('role', 'user')
            content = conv.get('content', '')
            
            if role == 'user':
                text_parts.append(f"<user>{content}</user>")
            elif role == 'assistant':
                text_parts.append(f"<assistant>{content}</assistant>")
            else:
                text_parts.append(f"{content}")
        
        full_text = '\n'.join(text_parts)
        tokens = self.tokenizer.encode(full_text, add_special_tokens=True)
        
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]
        
        labels = [-100] * len(tokens)
        last_assistant_start = full_text.rfind('<assistant>')
        
        if last_assistant_start != -1:
            prefix = full_text[:last_assistant_start]
            prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=True)
            start_idx = min(len(prefix_tokens), len(tokens))
            labels[start_idx:] = tokens[start_idx:]
        
        return {
            'input_ids': tokens,
            'labels': labels,
            'attention_mask': [1] * len(tokens),
            'conversations': conversations
        }
    
    def build_multimodal_sample(self, text: str, 
                               image_features: Optional[List[List[float]]] = None,
                               image_positions: Optional[List[int]] = None) -> Dict[str, Any]:
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        if image_features and image_positions:
            for pos in sorted(image_positions, reverse=True):
                tokens.insert(pos, self.tokenizer.vocab['<image>'])
        
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]
        
        return {
            'input_ids': tokens,
            'attention_mask': [1] * len(tokens),
            'image_features': image_features,
            'image_positions': image_positions
        }
    
    def pad_sample(self, sample: Dict[str, Any], pad_token_id: int = 0) -> Dict[str, Any]:
        max_len = self.max_seq_length
        
        for key in ['input_ids', 'labels', 'attention_mask', 'token_type_ids']:
            if key in sample:
                sample[key] = sample[key] + [pad_token_id] * (max_len - len(sample[key]))
        
        return sample
    
    def build_batch(self, samples: List[Dict[str, Any]], 
                   padding: bool = True) -> Dict[str, List[List[int]]]:
        batch = {
            'input_ids': [],
            'labels': [],
            'attention_mask': []
        }
        
        for sample in samples:
            if padding:
                sample = self.pad_sample(sample)
            
            batch['input_ids'].append(sample.get('input_ids', []))
            batch['labels'].append(sample.get('labels', []))
            batch['attention_mask'].append(sample.get('attention_mask', []))
            
            if 'token_type_ids' in sample:
                if 'token_type_ids' not in batch:
                    batch['token_type_ids'] = []
                batch['token_type_ids'].append(sample['token_type_ids'])
        
        return batch
    
    def truncate_pair(self, tokens_a: List[int], tokens_b: List[int]) -> Tuple[List[int], List[int]]:
        max_total = self.max_seq_length - 3
        
        while len(tokens_a) + len(tokens_b) > max_total:
            if len(tokens_a) > len(tokens_b):
                tokens_a = tokens_a[:-1]
            else:
                tokens_b = tokens_b[:-1]
        
        return tokens_a, tokens_b