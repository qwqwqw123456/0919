class HybridEngine:
    def __init__(self, mamba_model, attn_model):
        self.mamba = mamba_model
        self.attn = attn_model
    def generate(self, prompt):
        if len(prompt) > 100000:
            return self.mamba(prompt)
        else:
            return self.attn(prompt)
