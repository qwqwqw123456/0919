class SFTDataFormatter:
    @staticmethod
    def format_conversation(messages):
        prompt = ""
        for msg in messages:
            if msg["role"] == "user":
                prompt += f"USER: {msg['content']}
"
            elif msg["role"] == "assistant":
                prompt += f"ASSISTANT: {msg['content']}
"
        return prompt
