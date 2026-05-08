import math

class BasicMetricCalc:
    def perplexity(self, loss):
        return math.exp(loss)
