# 调用 vLLM 的封装，需安装 vllm
class VLLMInference:
    def __init__(self, model_path):
        from vllm import LLM
        self.llm = LLM(model=model_path)
    def generate(self, prompts):
        return self.llm.generate(prompts)
