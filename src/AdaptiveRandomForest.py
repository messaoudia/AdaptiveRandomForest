class AdaptiveRandomForest:
    def __init__(self, m, n, delta_w=0.0001, delta_d=0.00001):
        self.m = m
        self.n = n
        self.delta_w = delta_w
        self.delta_d = delta_d
