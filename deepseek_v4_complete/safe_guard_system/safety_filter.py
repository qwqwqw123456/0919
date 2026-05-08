import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json

@dataclass
class SafetyResult:
    safe: bool
    reason: str
    category: Optional[str] = None
    confidence: float = 0.0

class SafetyFilter:
    def __init__(self, config_path: Optional[str] = None):
        self.patterns = self._load_patterns(config_path)
        self.keyword_lists = self._load_keywords(config_path)
    
    def _load_patterns(self, config_path: Optional[str]) -> Dict[str, List[re.Pattern]]:
        patterns = {
            'toxic': [
                re.compile(r'\b(fuck|shit|damn|bitch|bastard|asshole|cunt|dick)\b', re.IGNORECASE),
                re.compile(r'\b(kill|murder|rape|torture)\b', re.IGNORECASE),
                re.compile(r'\b(terrorist|terrorism)\b', re.IGNORECASE),
            ],
            'hate': [
                re.compile(r'\b(nigger|chink|gook|jap|spic)\b', re.IGNORECASE),
                re.compile(r'\b(racist|sexist|homophobe)\b', re.IGNORECASE),
                re.compile(r'\b(die|go to hell)\b', re.IGNORECASE),
            ],
            'violence': [
                re.compile(r'\b(shoot|stab|stabbed|assault)\b', re.IGNORECASE),
                re.compile(r'\b(bomb|explode|explosion)\b', re.IGNORECASE),
                re.compile(r'\b(weapon|gun|knife)\b', re.IGNORECASE),
            ],
            'illegal': [
                re.compile(r'\b(drug|drugs|cocaine|heroin|meth)\b', re.IGNORECASE),
                re.compile(r'\b(steal|rob|robbery|theft)\b', re.IGNORECASE),
                re.compile(r'\b(piracy|pirate|cracked)\b', re.IGNORECASE),
            ],
            'privacy': [
                re.compile(r'\b(password|credit card|ssn|social security)\b', re.IGNORECASE),
                re.compile(r'\b(bank account|account number)\b', re.IGNORECASE),
                re.compile(r'\b(email|phone|address)\b', re.IGNORECASE),
            ],
            'political': [
                re.compile(r'\b(communist|fascist|nazism)\b', re.IGNORECASE),
                re.compile(r'\b(propaganda|brainwash)\b', re.IGNORECASE),
            ],
        }
        
        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if 'patterns' in config:
                        for category, pattern_list in config['patterns'].items():
                            if category not in patterns:
                                patterns[category] = []
                            for pattern in pattern_list:
                                patterns[category].append(re.compile(pattern, re.IGNORECASE))
            except Exception:
                pass
        
        return patterns
    
    def _load_keywords(self, config_path: Optional[str]) -> Dict[str, List[str]]:
        keywords = {
            'toxic': ['fuck', 'shit', 'damn', 'bitch', 'bastard', 'asshole'],
            'hate': ['nigger', 'chink', 'racist', 'sexist', 'homophobic'],
            'violence': ['kill', 'murder', 'rape', 'shoot', 'stab', 'bomb'],
            'illegal': ['drug', 'steal', 'rob', 'pirate', 'crack'],
            'privacy': ['password', 'credit card', 'ssn', 'bank account'],
        }
        
        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if 'keywords' in config:
                        keywords.update(config['keywords'])
            except Exception:
                pass
        
        return keywords
    
    def check_text(self, text: str) -> SafetyResult:
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    return SafetyResult(
                        safe=False,
                        reason=f"Contains {category} content",
                        category=category,
                        confidence=0.9
                    )
        
        for category, keywords in self.keyword_lists.items():
            text_lower = text.lower()
            for keyword in keywords:
                if keyword in text_lower:
                    return SafetyResult(
                        safe=False,
                        reason=f"Contains {category} keyword: {keyword}",
                        category=category,
                        confidence=0.7
                    )
        
        if self._check_prompt_injection(text):
            return SafetyResult(
                safe=False,
                reason="Potential prompt injection detected",
                category="injection",
                confidence=0.8
            )
        
        if self._check_excessive_length(text):
            return SafetyResult(
                safe=False,
                reason="Text exceeds maximum length",
                category="length",
                confidence=1.0
            )
        
        return SafetyResult(safe=True, reason="Text is safe")
    
    def _check_prompt_injection(self, text: str) -> bool:
        injection_patterns = [
            re.compile(r'ignore.*previous.*instructions?', re.IGNORECASE),
            re.compile(r'forget.*previous.*prompt', re.IGNORECASE),
            re.compile(r'you.*are.*not.*a.*language.*model', re.IGNORECASE),
            re.compile(r'roleplay.*as.*someone.*else', re.IGNORECASE),
            re.compile(r'pretend.*to.*be', re.IGNORECASE),
            re.compile(r'system.*prompt.*override', re.IGNORECASE),
            re.compile(r'<system>.*</system>', re.IGNORECASE),
        ]
        
        for pattern in injection_patterns:
            if pattern.search(text):
                return True
        return False
    
    def _check_excessive_length(self, text: str, max_length: int = 10000) -> bool:
        return len(text) > max_length
    
    def check_batch(self, texts: List[str]) -> List[SafetyResult]:
        return [self.check_text(text) for text in texts]
    
    def get_filtered_text(self, text: str, replacement: str = '[REDACTED]') -> str:
        result = self.check_text(text)
        if result.safe:
            return text
        
        filtered_text = text
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                filtered_text = pattern.sub(replacement, filtered_text)
        
        for category, keywords in self.keyword_lists.items():
            text_lower = filtered_text.lower()
            for keyword in keywords:
                filtered_text = filtered_text.replace(keyword, replacement)
                filtered_text = filtered_text.replace(keyword.capitalize(), replacement)
        
        return filtered_text

class ContentModeration:
    def __init__(self):
        self.safety_filter = SafetyFilter()
    
    def moderate(self, text: str) -> Dict[str, Any]:
        result = self.safety_filter.check_text(text)
        
        return {
            'safe': result.safe,
            'reason': result.reason,
            'category': result.category,
            'confidence': result.confidence,
            'filtered_text': self.safety_filter.get_filtered_text(text) if not result.safe else text
        }
    
    def moderate_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.moderate(text) for text in texts]
    
    def should_block(self, text: str) -> bool:
        result = self.safety_filter.check_text(text)
        return not result.safe