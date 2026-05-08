class RefuseReplyStrategy:
    def apply(self, text):
        if "违规" in text:
            return "抱歉，我不能回答该问题。"
        return text
