class CoreAgent:
    def __init__(self, model, tools):
        self.model = model
        self.tools = tools
    def run(self, task):
        # 简化：直接调用工具
        for tool_name, tool_fn in self.tools.items():
            if tool_name in task:
                return tool_fn(task)
        return self.model.generate(task)
