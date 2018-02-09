from skmultiflow.classification.core.attribute_class_observers.gaussian_numeric_attribute_class_observer import \
    GaussianNumericAttributeClassObserver
from skmultiflow.classification.core.attribute_class_observers.nominal_attribute_class_observer import \
    NominalAttributeClassObserver
from skmultiflow.classification.core.utils.utils import do_naive_bayes_prediction
from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree, MAJORITY_CLASS, NAIVE_BAYES
from xlsxwriter.contenttypes import overrides
import random
import numpy as np


class ARFHoeffdingTree (HoeffdingTree):

    def __init__(self, m):
        super().__init__()
        self.m = m
        self.remove_poor_atts = None
        self.no_preprune = True

    @staticmethod
    def is_randomizable():
        return True

    def rf_tree_train(self, X, y):
        self.partial_fit(X, y, weight=np.random.poisson(6, 1))

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

        def learn_from_instance(self, X, y, weight, arf_ht):
            """ learn_from_instance
            Update the node with the supplied instance.

            Parameters
            ----------
            X: The attributes for updating the node
            y: The class
            weight: The instance's weight
            ht: The Hoeffding Tree

            """
            if not self._is_initialized:
                self._attribute_observers = [None] * len(X)
                self._is_initialized = True
            if y not in self._observed_class_distribution:
                self._observed_class_distribution[y] = 0.0
            self._observed_class_distribution[y] += weight

            if not self.list_attributes:
                self.list_attributes = []
                for i in range(self.num_attributes):
                    is_unique = False
                    while not is_unique:
                        self.list_attributes.append(random.randint(0, len(X)-1))
                        is_unique = True
                        for j in range(i):
                            if self.list_attributes[i] == self.list_attributes[j]:
                                is_unique = False
                        if not is_unique:
                            break

            for i in range(self.num_attributes):
                attr_index = self.list_attributes[i]
                obs = self._attribute_observers[i]
                if obs is None:
                    if i in arf_ht.nominal_attributes:
                        obs = NominalAttributeClassObserver()
                    else:
                        obs = GaussianNumericAttributeClassObserver()
                    self._attribute_observers[i] = obs
                obs.observe_attribute_class(X[attr_index], int(y), weight)

    class LearningNodeNB(RandomLearningNode):

        def __init__(self, initial_class_observations, m):
            super().__init__(initial_class_observations, m)

        def get_class_votes(self, X, arf_ht):
            if self.get_weight_seen() >= arf_ht.nb_threshold:
                return do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)
            else:
                return super().get_class_votes(X, arf_ht)

        def disable_attribute(self, att_index):
            # Should not disable poor attributes, they are used in NB calculation
            pass

    class LearningNodeNBAdaptive(LearningNodeNB):

        def __init__(self, initial_class_observations, m):
            super().__init__(initial_class_observations, m)
            self._mc_correct_weight = 0.0
            self._nb_correct_weight = 0.0

        def learn_from_instance(self, X, y, weight, arf_ht):

            if self._observed_class_distribution == {}:
                # All classes equal, default to class 0
                if 0 == y:
                    self._mc_correct_weight += weight
            elif max(self._observed_class_distribution, key=self._observed_class_distribution.get) == y:
                self._mc_correct_weight += weight
            nb_prediction = do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)
            if max(nb_prediction, key=nb_prediction.get) == y:
                self._nb_correct_weight += weight
            super().learn_from_instance(X, y, weight, arf_ht)

        def get_class_votes(self, X, arf_ht):
            if self._mc_correct_weight > self._nb_correct_weight:
                return self._observed_class_distribution
            return do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)
