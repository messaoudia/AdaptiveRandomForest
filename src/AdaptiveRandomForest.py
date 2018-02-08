from collections import defaultdict
from src.ARFHoeffdingTree import ARFHoeffdingTree
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
        self.Trees = None
        self.Weights = None

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
            self.Weights[idx][0] += 1

        self.Weights[idx][1] += 1

    def partial_fit(self, X, y):
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
            for key, tree in enumerate(self.Trees.items()):
                y_predicted = tree.predict(X_)
                self.learning_performance(idx=key, y_predicted=y_predicted, y=y_)

                correct_prediction = (y_ == y_predicted)

                tree.rf_tree_train(self.m, X_, y_)
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
                self.Trees[index] = new_tree[index]

            new_tree.clear()
            index_to_replace.clear()

            for key, value in B.items():
                value.rf_tree_train(X_, y_)

        return self

    def predict(self, X):
        method = "avg"
        best_class = -1
        # average weight
        if method == "avg":
            predictions = defaultdict(lambda: defaultdict(float))
            predictions_count = dict(int)

            for index, tree in self.Trees:
                y = tree.predict(X)
                predictions[y] += self.Weights[index][0] / self.Weights[index][1]
                predictions_count[y] += 1

            max_weight = -1.0
            for key, weight in enumerate(predictions.items()):
                if best_class != key and weight/predictions_count[key] > max_weight:
                    max_weight = weight
                    best_class = key

        # TODO majority class
        elif method == "mc":
            return best_class

        p = list()
        p.append(best_class)
        return p
