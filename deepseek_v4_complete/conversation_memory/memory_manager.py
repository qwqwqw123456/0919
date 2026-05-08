import json
import hashlib
from typing import List, Dict, Optional, Any
from datetime import datetime
from collections import OrderedDict

class MemoryItem:
    def __init__(self, content: str, timestamp: Optional[datetime] = None, 
                 metadata: Optional[Dict[str, Any]] = None):
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        hash_input = f"{self.content}{self.timestamp.isoformat()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        return cls(
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )

class ConversationMemory:
    def __init__(self, max_items: int = 100):
        self.max_items = max_items
        self.items: OrderedDict[str, MemoryItem] = OrderedDict()
    
    def add_item(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        item = MemoryItem(content, metadata=metadata)
        self.items[item.id] = item
        
        if len(self.items) > self.max_items:
            self.items.popitem(last=False)
    
    def get_items(self, limit: Optional[int] = None) -> List[MemoryItem]:
        items = list(self.items.values())
        if limit:
            items = items[-limit:]
        return items
    
    def search(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        results = []
        for item in self.items.values():
            score = self._compute_similarity(query, item.content)
            if score > 0.3:
                results.append((score, item))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in results[:top_k]]
    
    def _compute_similarity(self, query: str, content: str) -> float:
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        if not query_words:
            return 0.0
        return len(query_words & content_words) / len(query_words)
    
    def clear(self):
        self.items.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'max_items': self.max_items,
            'items': [item.to_dict() for item in self.items.values()]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMemory':
        memory = cls(max_items=data.get('max_items', 100))
        for item_data in data.get('items', []):
            item = MemoryItem.from_dict(item_data)
            memory.items[item.id] = item
        return memory

class LongTermMemory:
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self.memories: Dict[str, ConversationMemory] = {}
    
    def get_conversation_memory(self, conversation_id: str) -> ConversationMemory:
        if conversation_id not in self.memories:
            self.memories[conversation_id] = ConversationMemory()
        return self.memories[conversation_id]
    
    def save_memory(self, conversation_id: str):
        if self.storage_path and conversation_id in self.memories:
            memory = self.memories[conversation_id]
            file_path = f"{self.storage_path}/{conversation_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(memory.to_dict(), f, ensure_ascii=False, indent=2)
    
    def load_memory(self, conversation_id: str) -> Optional[ConversationMemory]:
        if self.storage_path:
            file_path = f"{self.storage_path}/{conversation_id}.json"
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    memory = ConversationMemory.from_dict(data)
                    self.memories[conversation_id] = memory
                    return memory
            except FileNotFoundError:
                return None
        return None
    
    def delete_memory(self, conversation_id: str):
        if conversation_id in self.memories:
            del self.memories[conversation_id]
    
    def get_all_conversations(self) -> List[str]:
        return list(self.memories.keys())

class MemoryAugmenter:
    def __init__(self, long_term_memory: LongTermMemory):
        self.long_term_memory = long_term_memory
    
    def augment_prompt(self, conversation_id: str, prompt: str, 
                      max_context_items: int = 5) -> str:
        memory = self.long_term_memory.get_conversation_memory(conversation_id)
        relevant_items = memory.search(prompt, top_k=max_context_items)
        
        context = ""
        for item in relevant_items:
            role = item.metadata.get('role', 'user')
            content = item.content
            context += f"<memory_{role}>{content}</memory_{role}>\n"
        
        if context:
            return f"{context}\n{prompt}"
        return prompt
    
    def update_memory(self, conversation_id: str, message: Dict[str, str]):
        memory = self.long_term_memory.get_conversation_memory(conversation_id)
        memory.add_item(
            content=message['content'],
            metadata={'role': message.get('role', 'user')}
        )
        self.long_term_memory.save_memory(conversation_id)