import random
import re
from typing import List, Dict, Any, Optional
import numpy as np

class DataAugmenter:
    def __init__(self):
        self.synonym_dict = {
            'happy': ['joyful', 'glad', 'cheerful', 'delighted', 'pleased'],
            'sad': ['unhappy', 'sorrowful', 'melancholy', 'gloomy', 'depressed'],
            'big': ['large', 'huge', 'enormous', 'giant', 'massive'],
            'small': ['little', 'tiny', 'minuscule', 'petite', 'compact'],
            'fast': ['quick', 'rapid', 'swift', 'speedy', 'hasty'],
            'slow': ['unhurried', 'leisurely', 'gradual', 'sluggish', 'tardy'],
            'good': ['excellent', 'great', 'fine', 'wonderful', 'superb'],
            'bad': ['terrible', 'awful', 'poor', 'dreadful', 'horrible'],
            'look': ['see', 'watch', 'view', 'observe', 'notice'],
            'say': ['speak', 'tell', 'talk', 'utter', 'mention'],
            'go': ['leave', 'depart', 'travel', 'move', 'proceed'],
            'come': ['arrive', 'approach', 'enter', 'appear', 'reach'],
            'make': ['create', 'produce', 'build', 'construct', 'generate'],
            'take': ['grab', 'seize', 'obtain', 'get', 'acquire'],
            'give': ['provide', 'offer', 'grant', 'donate', 'supply'],
            'use': ['utilize', 'employ', 'apply', 'exploit', 'leverage'],
            'know': ['understand', 'comprehend', 'realize', 'recognize', 'grasp'],
            'think': ['consider', 'believe', 'suppose', 'assume', 'imagine'],
            'want': ['desire', 'wish', 'need', 'crave', 'long for'],
            'like': ['enjoy', 'prefer', 'admire', 'fancy', 'love'],
        }
        
        self.stop_words = ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                          'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                          'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
                          'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                          'from', 'as', 'into', 'through', 'during', 'before', 'after',
                          'above', 'below', 'between', 'under', 'again', 'further', 'then',
                          'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                          'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
                          'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                          'just', 'or', 'and', 'but', 'if', 'because', 'while', 'although',
                          'though', 'that', 'which', 'who', 'whom', 'this', 'these', 'those']
    
    def random_synonym_replacement(self, text: str, ratio: float = 0.1) -> str:
        words = text.split()
        new_words = []
        
        for word in words:
            lower_word = word.lower()
            if lower_word in self.synonym_dict and random.random() < ratio:
                synonyms = self.synonym_dict[lower_word]
                new_word = random.choice(synonyms)
                if word[0].isupper():
                    new_word = new_word.capitalize()
                new_words.append(new_word)
            else:
                new_words.append(word)
        
        return ' '.join(new_words)
    
    def random_deletion(self, text: str, ratio: float = 0.1) -> str:
        words = text.split()
        new_words = [word for word in words if random.random() > ratio]
        return ' '.join(new_words) if new_words else text
    
    def random_insertion(self, text: str, ratio: float = 0.1) -> str:
        words = text.split()
        new_words = words.copy()
        insert_count = int(len(words) * ratio)
        
        for _ in range(insert_count):
            pos = random.randint(0, len(new_words))
            random_word = random.choice(list(self.synonym_dict.keys()))
            new_words.insert(pos, random_word)
        
        return ' '.join(new_words)
    
    def random_swap(self, text: str, ratio: float = 0.1) -> str:
        words = text.split()
        new_words = words.copy()
        swap_count = int(len(words) * ratio)
        
        for _ in range(swap_count):
            if len(new_words) < 2:
                break
            i, j = random.sample(range(len(new_words)), 2)
            new_words[i], new_words[j] = new_words[j], new_words[i]
        
        return ' '.join(new_words)
    
    def back_translation(self, text: str, translator = None) -> str:
        if translator is None:
            return text
        try:
            translated = translator.translate(text, dest='zh')
            back_translated = translator.translate(translated.text, dest='en')
            return back_translated.text
        except Exception:
            return text
    
    def random_case_change(self, text: str, ratio: float = 0.1) -> str:
        chars = list(text)
        new_chars = []
        
        for char in chars:
            if random.random() < ratio:
                if char.islower():
                    new_chars.append(char.upper())
                elif char.isupper():
                    new_chars.append(char.lower())
                else:
                    new_chars.append(char)
            else:
                new_chars.append(char)
        
        return ''.join(new_chars)
    
    def add_noise(self, text: str, noise_level: float = 0.01) -> str:
        chars = list(text)
        new_chars = []
        
        for char in chars:
            if random.random() < noise_level:
                new_char = chr(random.randint(97, 122)) if char.isalpha() else char
                new_chars.append(new_char)
            else:
                new_chars.append(char)
        
        return ''.join(new_chars)
    
    def shuffle_sentences(self, text: str) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) > 1:
            random.shuffle(sentences)
            return ' '.join(sentences)
        return text
    
    def augment(self, text: str, methods: List[str] = None, **kwargs) -> str:
        if methods is None:
            methods = ['synonym', 'deletion', 'insertion', 'swap']
        
        augmented = text
        
        if 'synonym' in methods:
            augmented = self.random_synonym_replacement(augmented, kwargs.get('synonym_ratio', 0.1))
        
        if 'deletion' in methods:
            augmented = self.random_deletion(augmented, kwargs.get('deletion_ratio', 0.05))
        
        if 'insertion' in methods:
            augmented = self.random_insertion(augmented, kwargs.get('insertion_ratio', 0.05))
        
        if 'swap' in methods:
            augmented = self.random_swap(augmented, kwargs.get('swap_ratio', 0.05))
        
        if 'case' in methods:
            augmented = self.random_case_change(augmented, kwargs.get('case_ratio', 0.1))
        
        if 'noise' in methods:
            augmented = self.add_noise(augmented, kwargs.get('noise_level', 0.01))
        
        if 'shuffle' in methods:
            augmented = self.shuffle_sentences(augmented)
        
        if 'back_translate' in methods and kwargs.get('translator'):
            augmented = self.back_translation(augmented, kwargs.get('translator'))
        
        return augmented
    
    def augment_batch(self, texts: List[str], methods: List[str] = None, **kwargs) -> List[str]:
        return [self.augment(text, methods, **kwargs) for text in texts]
    
    def mixup(self, text1: str, text2: str, alpha: float = 0.5) -> str:
        words1 = text1.split()
        words2 = text2.split()
        
        len1 = int(len(words1) * alpha)
        len2 = int(len(words2) * (1 - alpha))
        
        mixed = words1[:len1] + words2[:len2]
        return ' '.join(mixed)
    
    def cutmix(self, text1: str, text2: str, cut_ratio: float = 0.3) -> str:
        words1 = text1.split()
        words2 = text2.split()
        
        if len(words1) < 3 or len(words2) < 3:
            return text1
        
        cut_len = int(len(words1) * cut_ratio)
        start_pos = random.randint(0, len(words1) - cut_len)
        
        insert_len = min(cut_len, len(words2))
        insert_pos = random.randint(0, len(words2) - insert_len)
        
        result = words1[:start_pos] + words2[insert_pos:insert_pos + insert_len] + words1[start_pos + cut_len:]
        return ' '.join(result)