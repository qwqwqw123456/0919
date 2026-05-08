class ToolManager:
    def __init__(self):
        self.tools = {}
    def register(self, name, func):
        self.tools[name] = func
    def call(self, name, *args, **kwargs):
        return self.tools[name](*args, **kwargs)
