from skmultiflow.classification.core.attribute_class_observers.gaussian_numeric_attribute_class_observer import \
    GaussianNumericAttributeClassObserver
from skmultiflow.classification.core.attribute_class_observers.nominal_attribute_class_observer import \
    NominalAttributeClassObserver
from skmultiflow.classification.core.driftdetection.adwin import ADWIN
from skmultiflow.classification.core.utils.utils import do_naive_bayes_prediction
from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree, MAJORITY_CLASS, NAIVE_BAYES
import random
import numpy as np


class ARFHoeffdingTree (HoeffdingTree):

    def predict_proba(self, X):
        pass

    def score(self, X, y):
        pass

    def fit(self, X, y, classes=None, weight=None):
        pass

    def __init__(self, m, delta_w=0.01, delta_d=0.001,):
        super().__init__()
        self.m = m
        self.remove_poor_atts = None
        self.no_preprune = True
        self.delta_warning = delta_w
        self.delta_drift = delta_d
        self.adwin_warning = ADWIN(delta=self.delta_warning)
        self.adwin_drift = ADWIN(delta=self.delta_drift)
        self.leaf_prediction = 'nb'
        self.grace_period = 50

    @staticmethod
    def is_randomizable():
        return True

    def rf_tree_train(self, X, y):
        w = np.random.poisson(6, len(X))
        if w > 0:
            self.partial_fit(X, y, weight=w)

    def _new_learning_node(self, initial_class_observations=None):
        if initial_class_observations is None:
            initial_class_observations = {}
        if self._leaf_prediction == MAJORITY_CLASS:
            return self.RandomLearningNode(initial_class_observations, self.m)
        elif self._leaf_prediction == NAIVE_BAYES:
            return self.LearningNodeNB(initial_class_observations, self.m)
        else:
            return self.LearningNodeNBAdaptive(initial_class_observations, self.m)

    class RandomLearningNode(HoeffdingTree.ActiveLearningNode):

        """A Hoeffding Tree node that supports growth."""
        def __init__(self, initial_class_observations, m):
            super().__init__(initial_class_observations)
            self.list_attributes = []
            self.num_attributes = m
            self._is_initialized = False
            self._attribute_observers = []

        def learn_from_instance(self, X, y, weight, ht):
            """ learn_from_instance
            Update the node with the supplied instance.

            Parameters
            ----------
            X: The attributes for updating the node
            y: The class
            weight: The instance's weight
            ht: The Hoeffding Tree

            """
            if self.num_attributes <= 0 or self.num_attributes > len(X) - 1:
                self.num_attributes = len(X) - 1
            if not self._is_initialized:
                self._attribute_observers = [None] * self.num_attributes
                self._is_initialized = True
            if y not in self._observed_class_distribution:
                self._observed_class_distribution[y] = 0.0
            self._observed_class_distribution[y] += weight

            if not self.list_attributes:
                self.list_attributes = []
                while len(self.list_attributes) < self.num_attributes:
                    attribute_index = random.randint(0, len(X)-1)
                    if attribute_index not in self.list_attributes:
                        self.list_attributes.append(attribute_index)

            for i in range(self.num_attributes):
                attr_index = self.list_attributes[i]
                obs = self._attribute_observers[i]
                if obs is None:
                    if i in ht.nominal_attributes:
                        obs = NominalAttributeClassObserver()
                    else:
                        obs = GaussianNumericAttributeClassObserver()
                    self._attribute_observers[i] = obs
                obs.observe_attribute_class(X[attr_index], int(y), weight)

    class LearningNodeNB(RandomLearningNode):

        def __init__(self, initial_class_observations, m):
            super().__init__(initial_class_observations, m)

        def get_class_votes(self, X, ht):
            observed_x = []
            for attr_index in self.list_attributes:
                observed_x.append(X[attr_index])
            if self.get_weight_seen() >= ht.nb_threshold:
                return do_naive_bayes_prediction(np.asarray(observed_x), self._observed_class_distribution, self._attribute_observers)
            else:
                return super().get_class_votes(np.asarray(observed_x), ht)

        def disable_attribute(self, att_index):
            # Should not disable poor attributes, they are used in NB calculation
            pass

    class LearningNodeNBAdaptive(LearningNodeNB):

        def __init__(self, initial_class_observations, m):
            super().__init__(initial_class_observations, m)
            self._mc_correct_weight = 0.0
            self._nb_correct_weight = 0.0

        def learn_from_instance(self, X, y, weight, ht):

            if self._observed_class_distribution == {}:
                # All classes equal, default to class 0
                if 0 == y:
                    self._mc_correct_weight += weight
            elif max(self._observed_class_distribution, key=self._observed_class_distribution.get) == y:
                self._mc_correct_weight += weight
            nb_prediction = do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)
            if max(nb_prediction, key=nb_prediction.get) == y:
                self._nb_correct_weight += weight
            super().learn_from_instance(X, y, weight, ht)

        def get_class_votes(self, X, ht):
            observed_x = []
            for attr_index in self.list_attributes:
                observed_x.append(X[attr_index])
            if self._mc_correct_weight > self._nb_correct_weight:
                return self._observed_class_distribution
            return do_naive_bayes_prediction(np.asarray(observed_x), self._observed_class_distribution, self._attribute_observers)
