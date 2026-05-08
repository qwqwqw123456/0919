class MemoryThinkPool:
    def __init__(self, capacity=100):
        self.pool = []
        self.capacity = capacity
    def add(self, thought):
        if len(self.pool) >= self.capacity:
            self.pool.pop(0)
        self.pool.append(thought)
