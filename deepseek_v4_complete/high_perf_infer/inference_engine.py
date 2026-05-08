import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass
import time

@dataclass
class GenerateConfig:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    stop_tokens: List[int] = None
    eos_token_id: int = 1

class InferenceEngine:
    def __init__(self, model, tokenizer, device: str = 'cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        
        if hasattr(self.model, 'config'):
            self.config = self.model.config
        else:
            self.config = None
    
    @torch.no_grad()
    def generate(self, input_ids: Union[List[int], torch.Tensor],
                 config: GenerateConfig = None) -> List[int]:
        if config is None:
            config = GenerateConfig()
        
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        
        generated = input_ids
        past_kv = None
        
        for _ in range(config.max_new_tokens):
            if past_kv is not None:
                input_ids = generated[:, -1:]
            
            outputs = self.model(input_ids, past_kv=past_kv, use_cache=True)
            
            if isinstance(outputs, tuple):
                logits, past_kv = outputs[0], outputs[-1]
            else:
                logits = outputs
                past_kv = None
            
            next_token_logits = logits[:, -1, :]
            
            if config.temperature != 1.0:
                next_token_logits = next_token_logits / config.temperature
            
            if config.repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token in generated[i]:
                        next_token_logits[i, token] /= config.repetition_penalty
            
            if config.top_k > 0:
                top_k_values, top_k_indices = torch.topk(next_token_logits, config.top_k)
                mask = torch.ones_like(next_token_logits) * float('-inf')
                mask.scatter_(1, top_k_indices, top_k_values)
                next_token_logits = mask
            
            if config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > config.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits.scatter_(1, indices_to_remove, float('-inf'))
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=-1)
            
            if config.stop_tokens and next_token.item() in config.stop_tokens:
                break
            
            if config.eos_token_id != -1 and next_token.item() == config.eos_token_id:
                break
        
        return generated[0].tolist()
    
    @torch.no_grad()
    def batch_generate(self, input_ids_list: List[List[int]],
                       config: GenerateConfig = None) -> List[List[int]]:
        if config is None:
            config = GenerateConfig()
        
        max_len = max(len(ids) for ids in input_ids_list)
        padded_ids = [ids + [self.tokenizer.vocab['<pad>']] * (max_len - len(ids)) 
                      for ids in input_ids_list]
        
        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([[1] * len(ids) + [0] * (max_len - len(ids)) 
                                      for ids in input_ids_list], device=self.device)
        
        batch_size = input_ids.size(0)
        generated = input_ids
        past_kv = None
        
        for _ in range(config.max_new_tokens):
            if past_kv is not None:
                input_ids = generated[:, -1:]
            
            outputs = self.model(input_ids, past_kv=past_kv, use_cache=True)
            
            if isinstance(outputs, tuple):
                logits, past_kv = outputs[0], outputs[-1]
            else:
                logits = outputs
                past_kv = None
            
            next_token_logits = logits[:, -1, :]
            
            if config.temperature != 1.0:
                next_token_logits = next_token_logits / config.temperature
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=-1)
            
            if config.eos_token_id != -1:
                finished = (next_token == config.eos_token_id).all()
                if finished:
                    break
        
        results = []
        for i in range(batch_size):
            seq = generated[i].tolist()
            if config.eos_token_id in seq:
                seq = seq[:seq.index(config.eos_token_id) + 1]
            results.append(seq)
        
        return results
    
    def chat(self, messages: List[Dict[str, str]], 
             config: GenerateConfig = None) -> str:
        if config is None:
            config = GenerateConfig()
        
        text = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                text += f"<system>{content}</system>"
            elif role == 'user':
                text += f"<user>{content}</user>"
            elif role == 'assistant':
                text += f"<assistant>{content}</assistant>"
        
        input_ids = self.tokenizer.encode(text)
        generated_ids = self.generate(input_ids, config)
        
        response = self.tokenizer.decode(generated_ids[len(input_ids):])
        return response.strip()
    
    def get_performance_stats(self, input_text: str, 
                              config: GenerateConfig = None) -> Dict[str, float]:
        if config is None:
            config = GenerateConfig()
        
        input_ids = self.tokenizer.encode(input_text)
        
        start_time = time.time()
        generated_ids = self.generate(input_ids, config)
        total_time = time.time() - start_time
        
        tokens_generated = len(generated_ids) - len(input_ids)
        tokens_per_second = tokens_generated / total_time
        
        return {
            'total_time': total_time,
            'tokens_generated': tokens_generated,
            'tokens_per_second': tokens_per_second
        }

class TensorRTInferenceEngine(InferenceEngine):
    def __init__(self, model_path: str, tokenizer):
        super().__init__(None, tokenizer)
        self._init_tensorrt(model_path)
    
    def _init_tensorrt(self, model_path: str):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            self.trt_logger = trt.Logger(trt.Logger.WARNING)
            with open(model_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            self.context = self.engine.create_execution_context()
            self.input_binding = self.engine.get_binding_index("input_ids")
            self.output_binding = self.engine.get_binding_index("logits")
            
            self.d_input = cuda.mem_alloc(1 * 1024 * 4)
            self.d_output = cuda.mem_alloc(1 * self.tokenizer.get_vocab_size() * 4)
            
            print("TensorRT engine initialized successfully")
        except ImportError:
            print("TensorRT not available, falling back to PyTorch")
            self.engine = None
    
    @torch.no_grad()
    def generate(self, input_ids: Union[List[int], torch.Tensor],
                 config: GenerateConfig = None) -> List[int]:
        if self.engine is None:
            return super().generate(input_ids, config)
        
        if config is None:
            config = GenerateConfig()
        
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.int32, device='cuda')
        
        generated = input_ids.tolist()
        
        for _ in range(config.max_new_tokens):
            input_tensor = torch.tensor([generated[-1024:]], dtype=torch.int32, device='cuda')
            
            import pycuda.driver as cuda
            cuda.memcpy_htod(self.d_input, input_tensor.data_ptr())
            
            bindings = [int(self.d_input), int(self.d_output)]
            self.context.execute_v2(bindings)
            
            output = torch.empty(1, len(generated[-1024:]), self.tokenizer.get_vocab_size(), 
                               dtype=torch.float32, device='cuda')
            cuda.memcpy_dtoh(output.data_ptr(), self.d_output)
            
            next_token_logits = output[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            
            generated.append(next_token)
            
            if config.eos_token_id != -1 and next_token == config.eos_token_id:
                break
        
        return generated