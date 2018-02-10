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
        self.delta_warning = delta_w
        self.delta_drift = delta_d


        self.predict_method = predict_method

        self.Trees = self.create_trees()
        self.Weights = self.init_weights()
        self.B = defaultdict()

        self.adwin_warning = ADWIN(delta=self.delta_warning)
        self.adwin_drift = ADWIN(delta=self.delta_drift)

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

        new_tree = list()
        index_to_replace = list()
        rows, cols = X.shape

        for stream in range(rows):
            X_ = X[stream, :]
            y_ = y[stream]

            # first tree => idx = 0, second tree => idx = 1 ...
            idx = 0
            for key, tree in self.Trees.items():
                y_predicted = tree.predict(np.asarray([X_]))
                self.learning_performance(idx=key, y_predicted=y_predicted, y=y_)

                correct_prediction = (y_ == y_predicted[0])

                tree.rf_tree_train(np.asarray([X_]), np.asarray([y_]))
                self.adwin_warning.add_element(correct_prediction)
                if self.adwin_warning.detected_change():
                    if self.B.get(key, None) is None:
                        b = self.create_tree()
                        self.B[key] = b
                else:
                    if self.B.get(key, None) is not None:
                        self.B.pop(key)

                self.adwin_drift.add_element(correct_prediction)
                if self.adwin_drift.detected_change():
                    if self.B.get(key, None) is None: # Added condition, there is some problem here, we detected a drift before warning
                        b = self.create_tree() # Also too many trees is being created
                        self.B[key] = b
                    new_tree.append(self.B[key])
                    index_to_replace.append(key)

            # tree ← B(tree)
            for key, index in enumerate(index_to_replace):
                self.Trees[key] = new_tree[key]

            new_tree.clear()
            index_to_replace.clear()

            for key, value in self.B.items():
                value.rf_tree_train(np.asarray([X_]), np.asarray([y_])) # Changed

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
