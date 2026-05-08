import re
import json
from typing import Dict, List, Any, Optional
import html

class RawDataCleaner:
    def __init__(self):
        self.url_pattern = re.compile(r'https?://[^\s]+')
        self.email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
        self.phone_pattern = re.compile(r'(\+?86)?1[3-9]\d{9}')
        self.html_tag_pattern = re.compile(r'<[^>]+>')
        self.special_char_pattern = re.compile(r'[^\w\s\p{P}]+', re.UNICODE)
        self.multiple_space_pattern = re.compile(r'\s+')
        self.multiple_newline_pattern = re.compile(r'\n{3,}')
    
    def clean_text(self, text: str) -> str:
        text = self._decode_html_entities(text)
        text = self._remove_urls(text)
        text = self._remove_emails(text)
        text = self._remove_phone_numbers(text)
        text = self._remove_html_tags(text)
        text = self._normalize_whitespace(text)
        text = self._clean_special_chars(text)
        text = self._trim_text(text)
        return text
    
    def _decode_html_entities(self, text: str) -> str:
        return html.unescape(text)
    
    def _remove_urls(self, text: str) -> str:
        return self.url_pattern.sub('', text)
    
    def _remove_emails(self, text: str) -> str:
        return self.email_pattern.sub('', text)
    
    def _remove_phone_numbers(self, text: str) -> str:
        return self.phone_pattern.sub('', text)
    
    def _remove_html_tags(self, text: str) -> str:
        return self.html_tag_pattern.sub('', text)
    
    def _normalize_whitespace(self, text: str) -> str:
        text = self.multiple_space_pattern.sub(' ', text)
        text = self.multiple_newline_pattern.sub('\n\n', text)
        return text
    
    def _clean_special_chars(self, text: str) -> str:
        return self.special_char_pattern.sub('', text)
    
    def _trim_text(self, text: str) -> str:
        return text.strip()
    
    def clean_json(self, json_str: str) -> Optional[Dict[str, Any]]:
        try:
            data = json.loads(json_str)
            return self.clean_dict(data)
        except json.JSONDecodeError:
            return None
    
    def clean_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {}
        for key, value in data.items():
            if isinstance(value, str):
                cleaned[key] = self.clean_text(value)
            elif isinstance(value, dict):
                cleaned[key] = self.clean_dict(value)
            elif isinstance(value, list):
                cleaned[key] = self.clean_list(value)
            else:
                cleaned[key] = value
        return cleaned
    
    def clean_list(self, data: List[Any]) -> List[Any]:
        cleaned = []
        for item in data:
            if isinstance(item, str):
                cleaned.append(self.clean_text(item))
            elif isinstance(item, dict):
                cleaned.append(self.clean_dict(item))
            elif isinstance(item, list):
                cleaned.append(self.clean_list(item))
            else:
                cleaned.append(item)
        return cleaned
    
    def remove_empty_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in data.items() if v not in [None, '', [], {}]}
    
    def validate_text_length(self, text: str, min_length: int = 10, max_length: int = 1_000_000) -> bool:
        return min_length <= len(text.strip()) <= max_length
    
    def detect_language(self, text: str) -> str:
        chinese_count = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_count = len(re.findall(r'[a-zA-Z]', text))
        
        if chinese_count > english_count * 2:
            return 'zh'
        elif english_count > chinese_count * 2:
            return 'en'
        else:
            return 'mixed'
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        return [self.clean_text(text) for text in texts]
    
    def filter_empty(self, texts: List[str]) -> List[str]:
        return [t for t in texts if t.strip()]
    
    def remove_duplicates(self, texts: List[str]) -> List[str]:
        seen = set()
        result = []
        for text in texts:
            if text not in seen:
                seen.add(text)
                result.append(text)
        return result