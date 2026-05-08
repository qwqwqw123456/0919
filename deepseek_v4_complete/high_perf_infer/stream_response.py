class StreamResponse:
    def stream(self, generator):
        for token in generator:
            yield token
