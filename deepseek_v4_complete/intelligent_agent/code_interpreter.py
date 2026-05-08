class CodeInterpreter:
    def execute(self, code):
        try:
            exec(code)
            return "Success"
        except Exception as e:
            return str(e)
