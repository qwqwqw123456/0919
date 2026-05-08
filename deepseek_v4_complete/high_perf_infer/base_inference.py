class BaseInference:
    def __init__(self, model):
        self.model = model
    def generate(self, prompt):
        return self.model.generate(prompt)
