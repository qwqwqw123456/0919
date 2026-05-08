class ReportGenerator:
    def generate(self, results):
        return json.dumps(results, indent=2)
