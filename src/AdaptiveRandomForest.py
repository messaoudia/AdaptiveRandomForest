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

    def create_trees(self):
        trees = defaultdict(lambda: ARFHoeffdingTree(self.m))
        for i in range(self.n):
            trees[i] = self.create_tree()
        return trees

    def create_tree(self):
        return ARFHoeffdingTree(self.m)

    def init_weights(self):
        return defaultdict(list)

    def learning_performance(self, idx, y_predicted, y):
        # well predicted
        if y == y_predicted:
            self.W[idx][0] += 1

        self.W[idx][1] += 1

    def train(self, X, y):
        self.T = self.create_trees()
        self.W = self.init_weights()
        B = defaultdict()

        adwin_d = ADWIN(delta=self.delta_d)
        adwin_w = ADWIN(delta=self.delta_w)

        new_tree = list()
        index_to_replace = list()

        for stream in range(len(X)):
            X_ = X[stream]
            y_ = y[stream]

            # first tree => idx = 0, second tree => idx = 1 ...
            idx = 0
            for key, t in enumerate(self.T.items()):
                y_predicted = t.predict(X_)
                self.learning_performance(idx=key, y_predicted=y_predicted, y=y_)

                correct_prediction = (y_ == y_predicted)

                t.rf_tree_train(self.m, X_, y_)
                adwin_w.add_element(correct_prediction)
                if adwin_w.detected_change():
                    if B.get(key, None) is None:
                        b = self.create_tree()
                        B[key] = b
                else:
                    if B.get(key, None) is not None:
                        B.pop(key)

                adwin_d.add_element(correct_prediction)
                if adwin_d.detected_change():
                    new_tree.append(B[key])
                    index_to_replace.append(key)

            # t ← B(t)
            for key, index in enumerate(index_to_replace):
                T[index] = new_tree[index]

            new_tree.clear()
            index_to_replace.clear()

            for key, value in B.items():
                value.rf_tree_train(X_, y_)


    def predict(self):

        predictions = defaultdict(float)
        for index, tree in self.T:
            predictions[index] += self.W[index][0]/self.W[index][1]

        max

        return predictions
