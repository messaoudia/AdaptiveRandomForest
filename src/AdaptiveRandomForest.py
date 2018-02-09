from collections import defaultdict
from src.ARFHoeffdingTree import ARFHoeffdingTree
from skmultiflow.classification.core.driftdetection.adwin import ADWIN
import numpy as np

class AdaptiveRandomForest:

    def __init__(self, m, n, delta_w=0.0001, delta_d=0.00001, predict_method="avg"):
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
        self.Trees = None
        self.Weights = None
        self.predict_method = predict_method

    def create_trees(self):
        trees = defaultdict(lambda: ARFHoeffdingTree(self.m))
        for i in range(self.n):
            trees[i] = self.create_tree()
        return trees

    def create_tree(self):
        return ARFHoeffdingTree(self.m)

    def init_weights(self):
        l = list()
        l.append(0)
        l.append(0)
        return defaultdict(lambda: l)

    def learning_performance(self, idx, y_predicted, y):
        # well predicted
        if y == y_predicted[0]:
            self.Weights[idx][0] += 1

        self.Weights[idx][1] += 1

    def partial_fit(self, X, y, classes=None):
        self.Trees = self.create_trees()
        self.Weights = self.init_weights()
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
            for key, tree in self.Trees.items():
                y_predicted = tree.predict(X_)
                self.learning_performance(idx=key, y_predicted=y_predicted, y=y_)

                correct_prediction = (y_ == y_predicted[0])

                tree.rf_tree_train(np.asarray([X_]), np.asarray([y_]))
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

            # tree ← B(tree)
            for key, index in enumerate(index_to_replace):
                self.Trees[key] = new_tree[key]

            new_tree.clear()
            index_to_replace.clear()

            for key, value in B.items():
                value.rf_tree_train(X_, y_)

    def predict(self, X):
        best_class = -1
        # average weight
        if self.predict_method == "avg":
            predictions = defaultdict(float)
            predictions_count = defaultdict(int)

            for index, val in enumerate(self.Trees.items()):
                tree = self.Trees[index]
                y = tree.predict(X)
                predictions[y[0]] += self.Weights[index][0] / self.Weights[index][1]
                predictions_count[y[0]] += 1

            max_weight = -1.0
            for key, weight in predictions.items():
                w = predictions[key]/predictions_count[key]
                if best_class != key and w > max_weight:
                    max_weight = w
                    best_class = key

        # TODO majority class
        elif self.predict_method == "mc":
            return best_class

        p = list()
        p.append(best_class)
        return p
