import json
import os
from typing import List, Dict, Optional, Tuple
import regex as re

class DeepSeekTokenizer:
    def __init__(self, vocab_file: str, merges_file: str = None):
        self.vocab = self._load_vocab(vocab_file)
        self.merges = self._load_merges(merges_file) if merges_file else {}
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.byte_encoder = self._build_byte_encoder()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
    def _load_vocab(self, vocab_file: str) -> Dict[str, int]:
        if vocab_file.endswith('.json'):
            with open(vocab_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        vocab = {}
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                token = line.strip().split()[0] if ' ' in line else line.strip()
                vocab[token] = idx
        return vocab
    
    def _load_merges(self, merges_file: str) -> Dict[Tuple[str, str], str]:
        merges = {}
        with open(merges_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 3:
                    merges[(parts[0], parts[1])] = parts[2]
        return merges
    
    def _build_byte_encoder(self) -> Dict[int, str]:
        base = [chr(i) for i in range(256)]
        for i in range(256):
            if i >= 33 and i <= 126:
                base[i] = chr(i)
            else:
                base[i] = f'<0x{i:02X}>'
        return {i: base[i] for i in range(256)}
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        text = self._clean_text(text)
        tokens = self._byte_level_encode(text)
        tokens = self._bpe_encode(tokens)
        if add_special_tokens:
            tokens = [self.vocab['<s>']] + tokens + [self.vocab['</s>']]
        return tokens
    
    def _clean_text(self, text: str) -> str:
        text = text.replace('\r\n', '\n')
        text = text.replace('\r', '\n')
        return text
    
    def _byte_level_encode(self, text: str) -> List[str]:
        tokens = []
        for char in text:
            cp = ord(char)
            if cp < 256:
                tokens.append(self.byte_encoder[cp])
            else:
                tokens.extend(self._encode_unicode(char))
        return tokens
    
    def _encode_unicode(self, char: str) -> List[str]:
        cp = ord(char)
        if cp < 0x80:
            return [self.byte_encoder[cp]]
        elif cp < 0x800:
            return [
                self.byte_encoder[0xC0 | (cp >> 6)],
                self.byte_encoder[0x80 | (cp & 0x3F)]
            ]
        elif cp < 0x10000:
            return [
                self.byte_encoder[0xE0 | (cp >> 12)],
                self.byte_encoder[0x80 | ((cp >> 6) & 0x3F)],
                self.byte_encoder[0x80 | (cp & 0x3F)]
            ]
        else:
            return [
                self.byte_encoder[0xF0 | (cp >> 18)],
                self.byte_encoder[0x80 | ((cp >> 12) & 0x3F)],
                self.byte_encoder[0x80 | ((cp >> 6) & 0x3F)],
                self.byte_encoder[0x80 | (cp & 0x3F)]
            ]
    
    def _bpe_encode(self, tokens: List[str]) -> List[int]:
        while len(tokens) > 1:
            best_score = float('inf')
            best_pair = None
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges:
                    best_pair = pair
                    best_score = self.merges.get(pair, float('inf'))
                    break
            if best_pair is None:
                break
            new_token = self.merges[best_pair]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return [self.vocab.get(t, self.vocab['<unk>']) for t in tokens]
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for id_ in ids:
            if id_ in self.id_to_token:
                token = self.id_to_token[id_]
                if skip_special_tokens and token in ['<s>', '</s>', '<pad>', '<unk>']:
                    continue
                tokens.append(token)
        text = self._bpe_decode(tokens)
        text = self._byte_level_decode(text)
        return text
    
    def _bpe_decode(self, tokens: List[str]) -> str:
        return ''.join(tokens).replace('@@ ', '')
    
    def _byte_level_decode(self, text: str) -> str:
        result = []
        i = 0
        while i < len(text):
            if text[i:i+4] == '<0x' and i + 6 <= len(text) and text[i+5] == '>':
                try:
                    byte = int(text[i+3:i+5], 16)
                    result.append(chr(byte))
                    i += 6
                except ValueError:
                    result.append(text[i])
                    i += 1
            else:
                result.append(text[i])
                i += 1
        return ''.join(result)
    
    def get_vocab_size(self) -> int:
        return len(self.vocab)
    
    def tokenize(self, text: str, add_special_tokens: bool = True) -> List[str]:
        ids = self.encode(text, add_special_tokens)
        return [self.id_to_token.get(id_, '<unk>') for id_ in ids]