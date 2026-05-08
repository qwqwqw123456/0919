class ConstitutionGuard:
    def __init__(self, rules):
        self.rules = rules
    def check(self, text):
        for rule in self.rules:
            if rule in text:
                return False
        return True
