class LongMemoryDB:
    def __init__(self):
        self.db = {}
    def store(self, key, value):
        self.db[key] = value
    def retrieve(self, key):
        return self.db.get(key)
