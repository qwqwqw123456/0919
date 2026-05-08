import re
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass

@dataclass
class FilterResult:
    passed: bool
    reason: str
    score: float = 0.0

class DataFilterRule:
    def __init__(self):
        self.rules = []
    
    def add_rule(self, name: str, func: Callable[[Any], FilterResult], weight: float = 1.0):
        self.rules.append({'name': name, 'func': func, 'weight': weight})
    
    def filter(self, data: Any) -> FilterResult:
        total_score = 0.0
        total_weight = 0.0
        failed_reasons = []
        
        for rule in self.rules:
            result = rule['func'](data)
            if not result.passed:
                failed_reasons.append(f"{rule['name']}: {result.reason}")
            total_score += result.score * rule['weight']
            total_weight += rule['weight']
        
        if failed_reasons:
            return FilterResult(passed=False, reason='; '.join(failed_reasons), score=0.0)
        
        avg_score = total_score / total_weight if total_weight > 0 else 0.0
        return FilterResult(passed=True, reason='All rules passed', score=avg_score)
    
    def batch_filter(self, data_list: List[Any]) -> List[FilterResult]:
        return [self.filter(data) for data in data_list]

class TextFilter:
    def __init__(self):
        self.min_length = 10
        self.max_length = 1000000
        self.min_char_diversity = 0.3
        self.max_repeat_ratio = 0.5
        self.max_url_ratio = 0.1
        self.max_special_char_ratio = 0.3
        
        self.low_quality_patterns = [
            re.compile(r'^[a-zA-Z0-9]{1,5}$'),
            re.compile(r'^(yes|no|ok|okay|sure|thanks|thank you)$', re.IGNORECASE),
            re.compile(r'^[\s\p{P}]+$'),
        ]
        
        self.invalid_patterns = [
            re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]'),
            re.compile(r'https?://[^\s]+'),
        ]
    
    def check_length(self, text: str) -> FilterResult:
        length = len(text.strip())
        if length < self.min_length:
            return FilterResult(passed=False, reason=f'Too short: {length} chars')
        if length > self.max_length:
            return FilterResult(passed=False, reason=f'Too long: {length} chars')
        return FilterResult(passed=True, reason='Length OK', score=1.0)
    
    def check_char_diversity(self, text: str) -> FilterResult:
        if len(text) == 0:
            return FilterResult(passed=False, reason='Empty text')
        
        unique_chars = len(set(text))
        diversity = unique_chars / len(text)
        
        if diversity < self.min_char_diversity:
            return FilterResult(passed=False, reason=f'Low character diversity: {diversity:.2f}')
        return FilterResult(passed=True, reason='Diversity OK', score=min(diversity * 2, 1.0))
    
    def check_repeat_ratio(self, text: str) -> FilterResult:
        if len(text) < 10:
            return FilterResult(passed=True, reason='Too short to check', score=1.0)
        
        max_repeat = 0
        for i in range(len(text) - 3):
            substr = text[i:i+4]
            count = text.count(substr)
            if count > 1:
                ratio = (count * 4) / len(text)
                max_repeat = max(max_repeat, ratio)
        
        if max_repeat > self.max_repeat_ratio:
            return FilterResult(passed=False, reason=f'High repeat ratio: {max_repeat:.2f}')
        return FilterResult(passed=True, reason='Repeat ratio OK', score=1.0)
    
    def check_special_chars(self, text: str) -> FilterResult:
        special_chars = re.findall(r'[^\w\s\u4e00-\u9fff]', text)
        ratio = len(special_chars) / len(text) if len(text) > 0 else 0
        
        if ratio > self.max_special_char_ratio:
            return FilterResult(passed=False, reason=f'Too many special chars: {ratio:.2f}')
        return FilterResult(passed=True, reason='Special chars OK', score=1.0)
    
    def check_invalid_patterns(self, text: str) -> FilterResult:
        for pattern in self.invalid_patterns:
            if pattern.search(text):
                return FilterResult(passed=False, reason=f'Contains invalid content')
        return FilterResult(passed=True, reason='No invalid patterns', score=1.0)
    
    def check_low_quality(self, text: str) -> FilterResult:
        text_lower = text.strip().lower()
        for pattern in self.low_quality_patterns:
            if pattern.match(text_lower):
                return FilterResult(passed=False, reason='Low quality content')
        return FilterResult(passed=True, reason='Quality OK', score=1.0)
    
    def check_language(self, text: str, allowed_langs: List[str] = None) -> FilterResult:
        if allowed_langs is None:
            return FilterResult(passed=True, reason='No language restriction', score=1.0)
        
        chinese_count = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_count = len(re.findall(r'[a-zA-Z]', text))
        
        if chinese_count > english_count * 2:
            lang = 'zh'
        elif english_count > chinese_count * 2:
            lang = 'en'
        else:
            lang = 'mixed'
        
        if lang not in allowed_langs:
            return FilterResult(passed=False, reason=f'Language not allowed: {lang}')
        return FilterResult(passed=True, reason=f'Language OK: {lang}', score=1.0)
    
    def filter(self, text: str, allowed_langs: List[str] = None) -> FilterResult:
        rules = [
            ('length', lambda: self.check_length(text)),
            ('diversity', lambda: self.check_char_diversity(text)),
            ('repeat', lambda: self.check_repeat_ratio(text)),
            ('special_chars', lambda: self.check_special_chars(text)),
            ('invalid', lambda: self.check_invalid_patterns(text)),
            ('quality', lambda: self.check_low_quality(text)),
            ('language', lambda: self.check_language(text, allowed_langs)),
        ]
        
        for name, func in rules:
            result = func()
            if not result.passed:
                return FilterResult(passed=False, reason=f'{name}: {result.reason}')
        
        return FilterResult(passed=True, reason='All checks passed', score=0.8)

class DocumentFilter:
    def __init__(self):
        self.text_filter = TextFilter()
    
    def filter_document(self, doc: Dict[str, Any]) -> FilterResult:
        if 'text' not in doc:
            return FilterResult(passed=False, reason='Missing text field')
        
        text_result = self.text_filter.filter(doc['text'])
        if not text_result.passed:
            return text_result
        
        if 'metadata' in doc and isinstance(doc['metadata'], dict):
            if 'source' in doc['metadata'] and doc['metadata']['source'] == '':
                return FilterResult(passed=False, reason='Empty source')
        
        return FilterResult(passed=True, reason='Document OK', score=text_result.score)