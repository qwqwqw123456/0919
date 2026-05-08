class RateLimiter:
    def __init__(self, max_per_minute=60):
        self.max = max_per_minute
        self.counts = {}
    def is_allowed(self, user_id):
        self.counts[user_id] = self.counts.get(user_id, 0) + 1
        return self.counts[user_id] <= self.max
