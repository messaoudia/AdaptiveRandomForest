from collections import defaultdict

class AdaptiveRandomForest:
    def __init__(self, m, n, delta_w=0.0001, delta_d=0.00001):
        self.m = m
        self.n = n
        self.delta_w = delta_w
        self.delta_d = delta_d

    def create_trees(self, n):
        return dict()

    def create_tree(self):
        return dict()

    def init_weights(self, n):
        return defaultdict(float)

    def change_detector(self, t, x, y):
        return False

    def learning_performance(self, t, y_predicted, y):
        return 0.0

    def train(self):
        T = self.create_trees(self.n)
        W = self.init_weights(self.n)
        B = defaultdict()

        while has_next(S):
            x,y = next(S)
            for t in T:
                y_predicted = predict(t, x)
                W[t] = self.learning_performance(t=W(t), y_predicted=y_predicted, y=y)
                rftt = RFTreeTrain(m, t, x, y)
                if self.change_detector(self.delta_w, t, x, y):
                    b = self.create_tree()
                    B[t] = b

                if self.change_detector(self.delta_d, t, x, y):
                    t = B[t]


            for key, value in B.items():
                rftt = RFTreeTrain(m, b, x, y)



