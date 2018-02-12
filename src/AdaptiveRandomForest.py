from collections import defaultdict, Counter
from src.ARFHoeffdingTree import ARFHoeffdingTree
import numpy as np


class AdaptiveRandomForest:
    """ AdaptiveRandomForest or ARF

        An Adaptive Random Forest is a classification algorithm that want to make
        Random Forest, which is not a stream algorithm, be again among the best classifier in streaming
        In this code you will find the implementation of the ARF described on :

            Adaptive random forests for evolving data stream classification
            Heitor M. Gomes, Albert Bifet, Jesse Read, Jean Paul Barddal,
            Fabricio Enembreck, Bernhard Pfharinger, Geoff Holmes, Talel Abdessalem


        Parameters
        ----------

        nb_features: Int
            The number of features a leaf should observe.

        nb_trees: Int
            The number of trees that the forest should contain

        predict_method: String
            Prediction method: either Majority Classifier "mc", Average "avg"

        """

    def __init__(self, nb_features=5, nb_trees=100, predict_method="mc", pretrain_size=1000, delta_w=0.01, delta_d=0.001):
        """
        Constructor
        :param predict_method:
        :type predict_method:
        :param nb_features: maximum feature evaluated per split
        :param nb_trees: total number of trees
        """
        self.m = nb_features
        self.n = nb_trees
        self.predict_method = predict_method
        self.pretrain_size = pretrain_size
        self.delta_d = delta_d
        self.delta_w = delta_w

        self.Trees = self.create_trees()
        self.Weights = self.init_weights()
        self.B = defaultdict()
        self.number_of_instances_seen = 0

    def create_trees(self):
        """
        Create nb_trees, trees
        :return: a dictionnary of trees
        :rtype: Dictionnary
        """
        trees = defaultdict(lambda: ARFHoeffdingTree(self.m, self.delta_w, self.delta_d))
        for i in range(self.n):
            trees[i] = self.create_tree()
        return trees

    def create_tree(self):
        """
        Create a ARF Hoeffding tree
        :return: a tree
        :rtype: ARFHoeffdingTree
        """
        return ARFHoeffdingTree(self.m,self.delta_w, self.delta_d)

    def init_weights(self):
        """
        Init weight of the trees. Weight is 1 per default
        :return: a dictionnary of weight, where each weight is associated to 1 ARF Hoeffding Tree
        :rtype: Dictionnary
        """
        l = list()
        l.append(1)
        l.append(1)
        return defaultdict(lambda: l)

    def learning_performance(self, idx, y_predicted, y):
        """
        Compute the learning performance of one tree at the index "idx"
        :param idx: index of the tree in the dictionnary
        :type idx: Int
        :param y_predicted: Prediction result
        :type y_predicted: Int
        :param y: The real y, from the training
        :type y: Int
        :return: /
        :rtype: /
        """
        # if well predicted, count th
        if y == y_predicted[0]:
            self.Weights[idx][0] += 1

        self.Weights[idx][1] += 1

    def partial_fit(self, X, y, classes=None):
        """
        Partial fit over X and y arrays
        :param X: Features
        :type X: Numpy.ndarray of shape (n_samples, n_features)
        :param y: Classes
        :type y: Vector
        :return:
        :rtype:
        """
        new_tree = list()
        index_to_replace = list()
        rows, cols = X.shape

        for stream in range(rows):
            X_ = X[stream, :]
            y_ = y[stream]
            self.number_of_instances_seen += 1

            # first tree => idx = 0, second tree => idx = 1 ...

            for key, tree in self.Trees.items():
                if self.number_of_instances_seen > self.pretrain_size:
                    y_predicted = tree.predict(np.asarray([X_]))
                    self.learning_performance(idx=key, y_predicted=y_predicted, y=y_)
                    if y_ == y_predicted[0]:
                        correct_prediction = 1
                    else:
                        correct_prediction = 0
                    tree.adwin_warning.add_element(correct_prediction)
                    tree.adwin_drift.add_element(correct_prediction)
                    if tree.adwin_warning.detected_change():
                        if self.B.get(key, None) is None:
                            b = self.create_tree()
                            self.B[key] = b
                    else:
                        if self.B.get(key, None) is not None:
                            self.B.pop(key)

                    if tree.adwin_drift.detected_change():
                        if self.B.get(key, None) is None:
                            # Added condition, there is some problem here, we detected a drift before warning
                            b = self.create_tree()  # Also too many trees is being created
                            self.B[key] = b
                        new_tree.append(self.B[key])
                        index_to_replace.append(key)
                tree.rf_tree_train(np.asarray([X_]), np.asarray([y_]))

            for key, value in self.B.items():
                value.rf_tree_train(np.asarray([X_]), np.asarray([y_]))  # Changed

            # tree ← B(tree)
            for key, index in enumerate(index_to_replace):
                self.Trees[index] = new_tree[key]
                self.B.pop(index)
                self.Weights[index][0] = 1
                self.Weights[index][1] = 1

            new_tree.clear()
            index_to_replace.clear()

    def predict(self, X):
        """
        Predicts the label of the X instance(s)

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
        All the samples we want to predict the label for.

        Returns
        -------
        list
        A list containing the predicted labels for all instances in X.
        """
        r, _ = X.shape
        predictions_result = list()

        for row in range(r):
            X_ = X[row]

            best_class = -1
            # average weight
            predictions = defaultdict(float)
            predictions_count = defaultdict(int)

            if self.predict_method == "avg":

                global_weight = 0.0

                for key, tree in self.Trees.items():
                    y_predicted = tree.predict(np.asarray([X_]))
                    learning_perf = self.Weights[key][0] / self.Weights[key][1]
                    predictions[y_predicted[0]] += learning_perf
                    global_weight += learning_perf
                    # predictions_count[y_predicted[0]] += 1

                max_weight = -1.0
                for key, weight in predictions.items():
                    w = predictions[key] / global_weight
                    if best_class != key and w > max_weight:
                        max_weight = w
                        best_class = key

            elif self.predict_method == "mc":
                for key, tree in self.Trees.items():
                    y_predicted = tree.predict(np.asarray([X_]))
                    predictions_count[y_predicted[0]] += 1
                max_value = -1.0

                for key, value in predictions_count.items():
                    if value > max_value:
                        best_class = key
                        max_value = value

            predictions_result.append(best_class)

        return predictions_result
