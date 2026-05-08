from typing import Dict, List, Optional

class SpecialTokens:
    def __init__(self):
        self.special_tokens = {
            '<s>': 0,
            '</s>': 1,
            '<pad>': 2,
            '<unk>': 3,
            '<mask>': 4,
            '<bos>': 0,
            '<eos>': 1,
            '<image>': 128000,
            '</image>': 128001,
            '<audio>': 128002,
            '</audio>': 128003,
            '<video>': 128004,
            '</video>': 128005,
            '<func_call>': 128006,
            '</func_call>': 128007,
            '<system>': 128008,
            '</system>': 128009,
            '<user>': 128010,
            '</user>': 128011,
            '<assistant>': 128012,
            '</assistant>': 128013,
            '<document>': 128014,
            '</document>': 128015,
            '<search>': 128016,
            '</search>': 128017,
            '<math>': 128018,
            '</math>': 128019,
            '<code>': 128020,
            '</code>': 128021,
            '<table>': 128022,
            '</table>': 128023,
            '<link>': 128024,
            '</link>': 128025,
            '<timestamp>': 128026,
            '</timestamp>': 128027,
        }
        
        self.token_to_id = self.special_tokens
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
    
    def get_special_token(self, token: str) -> Optional[int]:
        return self.special_tokens.get(token)
    
    def get_token_by_id(self, id_: int) -> Optional[str]:
        return self.id_to_token.get(id_)
    
    def is_special_token(self, token: str) -> bool:
        return token in self.special_tokens
    
    def is_special_token_id(self, id_: int) -> bool:
        return id_ in self.id_to_token
    
    def add_special_token(self, token: str, id_: Optional[int] = None) -> int:
        if token in self.special_tokens:
            return self.special_tokens[token]
        
        if id_ is None:
            id_ = max(self.special_tokens.values()) + 1
        
        self.special_tokens[token] = id_
        self.id_to_token[id_] = token
        return id_
    
    def remove_special_token(self, token: str) -> bool:
        if token in self.special_tokens:
            id_ = self.special_tokens[token]
            del self.special_tokens[token]
            del self.id_to_token[id_]
            return True
        return False
    
    def get_all_special_tokens(self) -> List[str]:
        return list(self.special_tokens.keys())
    
    def get_all_special_token_ids(self) -> List[int]:
        return list(self.special_tokens.values())
    
    def get_pad_token(self) -> str:
        return '<pad>'
    
    def get_pad_token_id(self) -> int:
        return self.special_tokens['<pad>']
    
    def get_bos_token(self) -> str:
        return '<s>'
    
    def get_bos_token_id(self) -> int:
        return self.special_tokens['<s>']
    
    def get_eos_token(self) -> str:
        return '</s>'
    
    def get_eos_token_id(self) -> int:
        return self.special_tokens['</s>']
    
    def get_unk_token(self) -> str:
        return '<unk>'
    
    def get_unk_token_id(self) -> int:
        return self.special_tokens['<unk>']
    
    def get_mask_token(self) -> str:
        return '<mask>'
    
    def get_mask_token_id(self) -> int:
        return self.special_tokens['<mask>']
    
    def get_image_tokens(self) -> Dict[str, int]:
        return {
            '<image>': self.special_tokens['<image>'],
            '</image>': self.special_tokens['</image>']
        }
    
    def get_audio_tokens(self) -> Dict[str, int]:
        return {
            '<audio>': self.special_tokens['<audio>'],
            '</audio>': self.special_tokens['</audio>']
        }
    
    def get_conversation_tokens(self) -> Dict[str, int]:
        return {
            '<system>': self.special_tokens['<system>'],
            '</system>': self.special_tokens['</system>'],
            '<user>': self.special_tokens['<user>'],
            '</user>': self.special_tokens['</user>'],
            '<assistant>': self.special_tokens['<assistant>'],
            '</assistant>': self.special_tokens['</assistant>'],
        }
    
    def get_format_tokens(self) -> Dict[str, int]:
        return {
            '<code>': self.special_tokens['<code>'],
            '</code>': self.special_tokens['</code>'],
            '<math>': self.special_tokens['<math>'],
            '</math>': self.special_tokens['</math>'],
            '<table>': self.special_tokens['<table>'],
            '</table>': self.special_tokens['</table>'],
        }
    
    def build_vocab(self) -> Dict[str, int]:
        return self.special_tokens.copy()