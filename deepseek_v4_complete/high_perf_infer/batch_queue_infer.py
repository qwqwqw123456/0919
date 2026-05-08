import queue

class BatchQueueInference:
    def __init__(self, model):
        self.model = model
        self.queue = queue.Queue()
    def process(self):
        while not self.queue.empty():
            prompt = self.queue.get()
            yield self.model.generate(prompt)
