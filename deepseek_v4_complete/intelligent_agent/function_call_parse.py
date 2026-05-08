import json
class FunctionCallParser:
    def parse(self, llm_output):
        try:
            return json.loads(llm_output)
        except:
            return None
