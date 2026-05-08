class SensitiveWordDetector:
    def __init__(self, words=None):
        self.words = words or []
    def detect(self, text):
        for w in self.words:
            if w in text:
                return True
        return False
