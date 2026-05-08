import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
import uvicorn
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "deepseek-v4"
    messages: List[Message]
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    stream: bool = False

class CompletionRequest(BaseModel):
    model: str = "deepseek-v4"
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    stream: bool = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Union[str, Dict[str, str]]]]
    usage: Dict[str, int]

class CompletionResponse(BaseModel):
    id: str
    object: str = "text.completion"
    created: int
    model: str
    choices: List[Dict[str, Union[str, int]]]
    usage: Dict[str, int]

class APIError(BaseModel):
    error: Dict[str, str]

class APIServer:
    def __init__(self, inference_engine, host: str = "0.0.0.0", port: int = 8000):
        self.inference_engine = inference_engine
        self.host = host
        self.port = port
        self.app = FastAPI(title="DeepSeek V4 API", version="1.0")
        self._setup_routes()
        self._setup_cors()
        
        self.request_count = 0
        self.total_tokens = 0
    
    def _setup_cors(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy"}
        
        @self.app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
        async def chat_completions(request: ChatCompletionRequest):
            try:
                messages = [{"role": m.role, "content": m.content} for m in request.messages]
                
                config = type('obj', (object,), {
                    'max_new_tokens': request.max_tokens,
                    'temperature': request.temperature,
                    'top_p': request.top_p,
                    'top_k': request.top_k,
                    'repetition_penalty': request.repetition_penalty
                })
                
                response = self.inference_engine.chat(messages, config)
                
                self.request_count += 1
                self.total_tokens += len(response.split())
                
                return ChatCompletionResponse(
                    id=f"chatcmpl-{self.request_count}",
                    created=0,
                    model=request.model,
                    choices=[{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response
                        },
                        "finish_reason": "stop"
                    }],
                    usage={
                        "prompt_tokens": sum(len(m.content.split()) for m in request.messages),
                        "completion_tokens": len(response.split()),
                        "total_tokens": 0
                    }
                )
            except Exception as e:
                logger.error(f"Error in chat completions: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/v1/completions", response_model=CompletionResponse)
        async def completions(request: CompletionRequest):
            try:
                input_ids = self.inference_engine.tokenizer.encode(request.prompt)
                
                config = type('obj', (object,), {
                    'max_new_tokens': request.max_tokens,
                    'temperature': request.temperature,
                    'top_p': request.top_p,
                    'top_k': request.top_k,
                    'repetition_penalty': request.repetition_penalty
                })
                
                generated_ids = self.inference_engine.generate(input_ids, config)
                response = self.inference_engine.tokenizer.decode(generated_ids[len(input_ids):])
                
                self.request_count += 1
                
                return CompletionResponse(
                    id=f"cmpl-{self.request_count}",
                    created=0,
                    model=request.model,
                    choices=[{
                        "text": response,
                        "index": 0,
                        "finish_reason": "stop"
                    }],
                    usage={
                        "prompt_tokens": len(input_ids),
                        "completion_tokens": len(generated_ids) - len(input_ids),
                        "total_tokens": len(generated_ids)
                    }
                )
            except Exception as e:
                logger.error(f"Error in completions: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/v1/models")
        async def list_models():
            return {
                "object": "list",
                "data": [{
                    "id": "deepseek-v4",
                    "object": "model",
                    "created": 0,
                    "owned_by": "deepseek"
                }]
            }
        
        @self.app.get("/metrics")
        async def get_metrics():
            return {
                "request_count": self.request_count,
                "total_tokens": self.total_tokens
            }
    
    def run(self):
        logger.info(f"Starting API server on {self.host}:{self.port}")
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )

async def main():
    server = APIServer(None)
    server.run()

if __name__ == "__main__":
    asyncio.run(main())