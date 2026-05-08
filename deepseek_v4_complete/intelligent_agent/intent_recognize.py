class IntentRecognizer:
    def recognize(self, text):
        if "搜索" in text: return "search"
        if "执行代码" in text: return "execute_code"
        return "chat"
