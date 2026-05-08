class SelfRewarding:
    def __init__(self, model):
        self.model = model
    def generate_with_score(self, prompt):
        responses = [self.model.generate(prompt) for _ in range(4)]
        scores = [self.model.score(prompt, r) for r in responses]
        best = responses[scores.index(max(scores))]
        return best
