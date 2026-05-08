class MemoryCompressor:
    def compress(self, memories, max_tokens=1000):
        return " ".join(memories)[:max_tokens]
