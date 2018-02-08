from collections import defaultdict
from src import ARFHoeffdingTree
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
        for i in range(n):
            trees[i] = self.create_tree()
        return trees

    def create_tree(self):
        return ARFHoeffdingTree(self.m)

    def init_weights(self, n):
        return defaultdict(float)

    def change_detector(self, t, x, y):
        ADWIN(self.delta_d)
        return False

    def learning_performance(self, idx, y_predicted, y):
        # well predicted
        if y == y_predicted:
            self.W[idx][0] += 1
        # not well predicted
        else:
            self.W[idx][0] += 0

        self.W[idx][1] += 1

        nb_good_prediction = self.W[idx][0]
        nb_seen = self.W[idx][1]

        return nb_seen/nb_good_prediction

    def train(self):
        T = self.create_trees(self.n)
        self.W = self.init_weights(self.n)
        B = defaultdict()
        adwin_d = ADWIN(delta=self.delta_d)
        adwin_w = ADWIN(delta=self.delta_w)

        new_tree = list()
        index_to_replace = list()

        while has_next(S):
            x, y = next(S)

            # first tree => idx = 0, second tree => idx = 1 ...
            idx = 0
            for t in T:
                y_predicted = t.predict(t, x)
                self.W[idx] = self.learning_performance(idx=idx, y_predicted=y_predicted, y=y)

                correct_prediction = (y == y_predicted)

                t.rf_tree_train(self.m, x, y)
                adwin_d.add_element(correct_prediction)
                if self.change_detector(self.delta_w, t, x, y):
                    b = self.create_tree()
                    B[idx] = b

                if self.change_detector(self.delta_d, t, x, y):
                    new_tree.append(B[idx])
                    index_to_replace.append(idx)

                idx += 1

            # t ← B(t)
            for key, index in enumerate(index_to_replace):
                T[index] = new_tree[index]

            new_tree.clear()
            index_to_replace.clear()

            for key, value in B.items():
                value.rf_tree_train(self.m, x, y)

    def predict(self, X):
        return list()
