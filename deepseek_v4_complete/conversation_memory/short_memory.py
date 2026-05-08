class ShortMemory:
    def __init__(self, max_len=10):
        self.history = []
        self.max_len = max_len
    def add(self, user, assistant):
        self.history.append((user, assistant))
        if len(self.history) > self.max_len:
            self.history.pop(0)
    def get_context(self):
        return "\n".join([f"User: {u}\nAssistant: {a}" for u,a in self.history])
