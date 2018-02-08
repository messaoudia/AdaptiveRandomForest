from collections import defaultdict
from src import RFTree
from skmultiflow.classification.core.driftdetection.adwin import ADWIN

class AdaptiveRandomForest:

    def __init__(self, m, n, delta_w=0.0001, delta_d=0.00001):
        """
        Constructor
        :param m: maximum feature evaluated per split
        :param n: total number of trees
        :param delta_w: warning threshold
        :param delta_d: drift threshold
        """
        self.m = m
        self.n = n
        self.delta_w = delta_w
        self.delta_d = delta_d

    def create_trees(self, n):
        trees = defaultdict(int)
        for i in enumerate(n-1):
            trees[i] = self.create_tree()
        return trees

    def create_tree(self):
        return RFTree()

    def init_weights(self, n):
        weights =

        return defaultdict(float)

    def change_detector(self, t, x, y):
        ADWIN(self.delta_d)
        return False

    def learning_performance(self, t, y_predicted, y):
        nb_seen = 0
        nb_good_prediction = 0

        return nb_seen/nb_good_prediction

    def train(self):
        T = self.create_trees(self.n)
        W = self.init_weights(self.n)
        B = defaultdict()
        adwin_d = ADWIN(delta=self.delta_d)
        adwin_w = ADWIN(delta=self.delta_w)

        while has_next(S):
            x, y = next(S)
            for t in T:
                y_predicted = t.predict(t, x)
                W[t] = self.learning_performance(t=W(t), y_predicted=y_predicted, y=y)

                correct_prediction = y == y_predicted

                rftt = RFTree(self.m, t, x, y)
                adwin_d.add_element(correct_prediction)
                if self.change_detector(self.delta_w, t, x, y):
                    b = self.create_tree()
                    B[t] = b

                if self.change_detector(self.delta_d, t, x, y):
                    t = B[t]

            for key, value in B.items():
                rftt = RFTree(m, b, x, y)

    def predict(self, X):
        return list()
