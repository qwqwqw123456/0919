class ResultFilter:
    def filter(self, results):
        return [r for r in results if len(r.get("snippet","")) > 10]
