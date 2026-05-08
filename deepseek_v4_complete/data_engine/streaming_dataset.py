import json
import os
from typing import List, Dict, Any, Optional, Iterator
from torch.utils.data import IterableDataset
import random

class StreamingDataset(IterableDataset):
    def __init__(self, data_paths: List[str],
                 transform: Optional[callable] = None,
                 shuffle: bool = True,
                 buffer_size: int = 10000):
        self.data_paths = data_paths
        self.transform = transform
        self.shuffle = shuffle
        self.buffer_size = buffer_size
    
    def _read_file(self, file_path: str) -> Iterator[Dict[str, Any]]:
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
        elif ext == '.jsonl':
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    yield {'text': line.strip()}
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def _stream_files(self) -> Iterator[Dict[str, Any]]:
        if self.shuffle:
            random.shuffle(self.data_paths)
        
        for file_path in self.data_paths:
            for item in self._read_file(file_path):
                yield item
    
    def __iter__(self):
        if self.shuffle:
            buffer = []
            for item in self._stream_files():
                buffer.append(item)
                if len(buffer) >= self.buffer_size:
                    random.shuffle(buffer)
                    for item in buffer:
                        if self.transform:
                            item = self.transform(item)
                        yield item
                    buffer = []
            
            if buffer:
                random.shuffle(buffer)
                for item in buffer:
                    if self.transform:
                        item = self.transform(item)
                    yield item
        else:
            for item in self._stream_files():
                if self.transform:
                    item = self.transform(item)
                yield item

class ConcatDataset(IterableDataset):
    def __init__(self, datasets: List[IterableDataset], weights: Optional[List[float]] = None):
        self.datasets = datasets
        self.weights = weights if weights else [1.0] * len(datasets)
        self.normalized_weights = [w / sum(self.weights) for w in self.weights]
    
    def __iter__(self):
        iterators = [iter(ds) for ds in self.datasets]
        active = [True] * len(self.datasets)
        
        while any(active):
            idx = random.choices(range(len(self.datasets)), weights=self.normalized_weights)[0]
            
            if active[idx]:
                try:
                    yield next(iterators[idx])
                except StopIteration:
                    active[idx] = False

class DynamicDataset(IterableDataset):
    def __init__(self, base_dir: str,
                 file_pattern: str = '*.jsonl',
                 transform: Optional[callable] = None,
                 shuffle: bool = True):
        self.base_dir = base_dir
        self.file_pattern = file_pattern
        self.transform = transform
        self.shuffle = shuffle
        self.refresh()
    
    def refresh(self):
        import glob
        self.data_paths = glob.glob(os.path.join(self.base_dir, self.file_pattern))
    
    def __iter__(self):
        self.refresh()
        if self.shuffle:
            random.shuffle(self.data_paths)
        
        for file_path in self.data_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            if self.transform:
                                item = self.transform(item)
                            yield item
                        except json.JSONDecodeError:
                            continue

class TextDataset(IterableDataset):
    def __init__(self, text_files: List[str],
                 block_size: int = 2048,
                 tokenizer = None,
                 shuffle: bool = True):
        self.text_files = text_files
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.shuffle = shuffle
    
    def _read_text(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.text_files)
        
        for file_path in self.text_files:
            text = self._read_text(file_path)
            
            if self.tokenizer:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                
                for i in range(0, len(tokens), self.block_size):
                    block = tokens[i:i + self.block_size]
                    if len(block) >= self.block_size // 2:
                        yield {'input_ids': block, 'length': len(block)}
            else:
                for i in range(0, len(text), self.block_size):
                    block = text[i:i + self.block_size]
                    if len(block) >= self.block_size // 2:
                        yield {'text': block, 'length': len(block)}