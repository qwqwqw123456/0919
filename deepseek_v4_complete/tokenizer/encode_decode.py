from typing import List, Dict, Optional, Tuple, Union
import json
import os

class TokenizerEncodeDecode:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def encode_text(self, text: str, 
                   add_special_tokens: bool = True,
                   max_length: Optional[int] = None,
                   truncation: bool = True,
                   padding: bool = False,
                   pad_to_max_length: bool = False) -> Dict[str, List[int]]:
        tokens = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        
        if max_length is not None and truncation:
            tokens = tokens[:max_length]
        
        if padding or pad_to_max_length:
            if max_length is None:
                max_length = len(tokens)
            padding_length = max_length - len(tokens)
            if padding_length > 0:
                tokens = tokens + [self.tokenizer.vocab['<pad>']] * padding_length
        
        attention_mask = [1] * len(tokens)
        if padding:
            attention_mask = attention_mask + [0] * (max_length - len(tokens)) if max_length else attention_mask
        
        return {
            'input_ids': tokens,
            'attention_mask': attention_mask
        }
    
    def encode_batch(self, texts: List[str],
                    add_special_tokens: bool = True,
                    max_length: Optional[int] = None,
                    truncation: bool = True,
                    padding: bool = True) -> Dict[str, List[List[int]]]:
        encoded = []
        for text in texts:
            encoded.append(self.encode_text(text, add_special_tokens, max_length, truncation, padding=False))
        
        if max_length is None:
            max_length = max(len(e['input_ids']) for e in encoded)
        
        input_ids = []
        attention_masks = []
        
        for e in encoded:
            ids = e['input_ids']
            mask = e['attention_mask']
            
            if padding:
                padding_length = max_length - len(ids)
                ids = ids + [self.tokenizer.vocab['<pad>']] * padding_length
                mask = mask + [0] * padding_length
            
            input_ids.append(ids)
            attention_masks.append(mask)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks
        }
    
    def decode_tokens(self, ids: List[int],
                     skip_special_tokens: bool = True,
                     clean_up_tokenization_spaces: bool = True) -> str:
        text = self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        
        if clean_up_tokenization_spaces:
            text = self._clean_up_tokenization(text)
        
        return text
    
    def decode_batch(self, ids_batch: List[List[int]],
                    skip_special_tokens: bool = True,
                    clean_up_tokenization_spaces: bool = True) -> List[str]:
        return [
            self.decode_tokens(ids, skip_special_tokens, clean_up_tokenization_spaces)
            for ids in ids_batch
        ]
    
    def _clean_up_tokenization(self, text: str) -> str:
        text = text.replace(' .', '.').replace(' ,', ',').replace(' !', '!').replace(' ?', '?')
        text = text.replace(" ' ", "'").replace(" n't", "n't").replace(" 's", "'s")
        text = text.replace(" \n", "\n").replace("\n ", "\n")
        text = ' '.join(text.split())
        return text
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.tokenizer.vocab.get(token, self.tokenizer.vocab['<unk>']) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.tokenizer.id_to_token.get(id_, '<unk>') for id_ in ids]
    
    def create_padding_mask(self, input_ids: List[List[int]], pad_token_id: int) -> List[List[int]]:
        return [[1 if token != pad_token_id else 0 for token in seq] for seq in input_ids]
    
    def create_position_ids(self, input_ids: List[List[int]], pad_token_id: int) -> List[List[int]]:
        position_ids = []
        for seq in input_ids:
            pos = []
            current_pos = 0
            for token in seq:
                if token == pad_token_id:
                    pos.append(0)
                else:
                    pos.append(current_pos)
                    current_pos += 1
            position_ids.append(pos)
        return position_ids