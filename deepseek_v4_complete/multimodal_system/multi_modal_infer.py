class MultiModalInference:
    def __init__(self, model):
        self.model = model
    def generate(self, text, image=None):
        return self.model.generate(text, image)
