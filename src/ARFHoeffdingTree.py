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
    """
    ARFHoeffding Tree

    A Hoeffding tree is an incremental, anytime decision tree induction algorithm that is capable of learning from
    massive data streams, assuming that the distribution generating examples does not change over time. Hoeffding trees
    exploit the fact that a small sample can often be enough to choose an optimal splitting attribute. This idea is
    supported mathematically by the Hoeffding bound, which quantifies the number of observations (in our case, examples)
    needed to estimate some statistics within a prescribed precision (in our case, the goodness of an attribute).

    A theoretically appealing feature of Hoeffding Trees not shared by other incremental decision tree learners is that
    it has sound guarantees of performance. Using the Hoeffding bound one can show that its output is asymptotically
    nearly identical to that of a non-incremental learner using infinitely many examples.

    ARFHoeffding tree is based on Hoeffding tree and it has two main differences. Whenever a new node is created, a
    subset of m random attributes is chosen and split attempts are limited to that subset.
    Second difference is that there is no early tree prunning.

        See for details:
        G. Hulten, L. Spencer, and P. Domingos. Mining time-changing data streams.
        In KDD’01, pages 97–106, San Francisco, CA, 2001. ACM Press.

        Implementation based on:
        Albert Bifet, Geoff Holmes, Richard Kirkby, Bernhard Pfahringer (2010);
        MOA: Massive Online Analysis; Journal of Machine Learning Research 11: 1601-1604

    Parameters
    ----------
    m: Int
        Number of random attributes for split on each node

    grace_period: Int
        The number of instances a leaf should observe between split attempts.

    delta_w: float
        Warning threshold of change detection for ADWIN change detector

    delta_d: float
        Change threshold of change detection for ADWIN change detector

    no_pre_prune: Boolean
        If True, disable pre-pruning. Default: True

    leaf_prediction: String
        Prediction mechanism used at leafs.
        'mc' - Majority Class
        'nb' - Naive Bayes
        'nba' - Naive BAyes Adaptive

    Other attributes for Hoeffding Tree:

    HoeffdingTree.max_byte_size: Int
        Maximum memory consumed by the tree.

    HoeffdingTree.memory_estimate_period: Int
        How many instances between memory consumption checks.

    HoeffdingTree.split_criterion: String
        Split criterion to use.
        'gini' - Gini
        'info_gain' - Information Gain

    HoeffdingTree.split_confidence: Float
        Allowed error in split decision, a value closer to 0 takes longer to decide.

    HoeffdingTree.tie_threshold: Float
        Threshold below which a split will be forced to break ties.

    HoeffdingTree.binary_split: Boolean
        If True only allow binary splits.

    HoeffdingTree.stop_mem_management: Boolean
        If True, stop growing as soon as memory limit is hit.

    HoeffdingTree.remove_poor_atts: Boolean
        If True, disable poor attributes.

    HoeffdingTree.nb_threshold: Int
        The number of instances a leaf should observe before permitting Naive Bayes.

    HoeffdingTree.nominal_attributes: List
        List of Nominal attributes


    """
    def __init__(self, m, delta_w, delta_d, grace_period=50, leaf_prediction='nb', no_pre_prune=True):
        super().__init__()
        self.m = m
        self.remove_poor_atts = None
        self.no_preprune = no_pre_prune
        self.delta_warning = delta_w
        self.delta_drift = delta_d
        self.adwin_warning = ADWIN(delta=self.delta_warning)
        self.adwin_drift = ADWIN(delta=self.delta_drift)
        self.leaf_prediction = leaf_prediction
        self.grace_period = grace_period

    def predict_proba(self, X):
        pass

    def score(self, X, y):
        pass

    def fit(self, X, y, classes=None, weight=None):
        pass

    @staticmethod
    def is_randomizable():
        return True

    def rf_tree_train(self, X, y):
        """
        This function calculates Poisson(6) and assigns this as a weight of instance.
        If Poisson(6) returns zero, it doesn't use this instance for training.

        :param X: Array
            Input vector
        :param y: Array
            True value of class for X
        """
        w = np.random.poisson(6, len(X))
        if w > 0:
            self.partial_fit(X, y, weight=w)

    def _new_learning_node(self, initial_class_observations=None):
        """
        Depending on the method used for leaf prediction we define three types of nodes.

        :param initial_class_observations:
        :return: Node of appropriate type
        """
        if initial_class_observations is None:
            initial_class_observations = {}
        if self._leaf_prediction == MAJORITY_CLASS:
            return self.RandomLearningNode(initial_class_observations, self.m)
        elif self._leaf_prediction == NAIVE_BAYES:
            return self.LearningNodeNB(initial_class_observations, self.m)
        else:
            return self.LearningNodeNBAdaptive(initial_class_observations, self.m)

    class RandomLearningNode(HoeffdingTree.ActiveLearningNode):

        """A Hoeffding Tree node that supports growth. Used when leaf prediction is Majority class."""
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
            # Check if the value of m has proper value
            # If it is less or equal to zero or larger then total number of attributes in X
            # Replace it by total number of X
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
        """
        Node used when leaf prediction is Naive Bayes.
        """
        def __init__(self, initial_class_observations, m):
            super().__init__(initial_class_observations, m)

        def get_class_votes(self, X, ht):
            """
            :param X: Array
                Input vector of attributes
            :param ht: ARFHoeffdingTree
                ARF Hoeffding Tree to update.
            :return: Class votes
            """
            # Create a list of attributes that are observed
            observed_x = []
            for attr_index in self.list_attributes:
                observed_x.append(X[attr_index])
            if self.get_weight_seen() >= ht.nb_threshold:
                # Do the prediction only on observed attributes
                return do_naive_bayes_prediction(np.asarray(observed_x), self._observed_class_distribution,
                                                 self._attribute_observers)
            else:
                return super().get_class_votes(np.asarray(observed_x), ht)

        def disable_attribute(self, att_index):
            # Should not disable poor attributes, they are used in NB calculation
            pass

    class LearningNodeNBAdaptive(LearningNodeNB):
        """
        Node used when leaf prediction is Naive Bayes Adaptive.
        """
        def __init__(self, initial_class_observations, m):
            super().__init__(initial_class_observations, m)
            self._mc_correct_weight = 0.0
            self._nb_correct_weight = 0.0

        def learn_from_instance(self, X, y, weight, ht):
            """
            :param X: Array
                Input vector of attributes
            :param y: Array
                True value of class for X
            :param weight: Int
                Weight of the instance (Poisson(6))
            :param ht: ARFHoeffdingTree
                ARF Hoeffding Tree to update
            """
            # Create a list of attributes that are observed
            observed_x = []
            for attr_index in self.list_attributes:
                observed_x.append(X[attr_index])

            if self._observed_class_distribution == {}:
                # All classes equal, default to class 0
                if 0 == y:
                    self._mc_correct_weight += weight
            elif max(self._observed_class_distribution, key=self._observed_class_distribution.get) == y:
                self._mc_correct_weight += weight
            # Do the prediction only on observed attributes
            nb_prediction = do_naive_bayes_prediction(np.asarray(observed_x), self._observed_class_distribution,
                                                      self._attribute_observers)
            if max(nb_prediction, key=nb_prediction.get) == y:
                self._nb_correct_weight += weight
            super().learn_from_instance(np.asarray(observed_x), y, weight, ht)

        def get_class_votes(self, X, ht):
            """
            :param X: Array
                Input vector of attributes
            :param ht: ARFHoeffdingTree
                ARF Hoeffding Tree to update
            :return: Class votes
            """
            # Create a list of attributes that are observed
            observed_x = []
            for attr_index in self.list_attributes:
                observed_x.append(X[attr_index])
            if self._mc_correct_weight > self._nb_correct_weight:
                return self._observed_class_distribution
            # Do the prediction only on observed attributes
            return do_naive_bayes_prediction(np.asarray(observed_x), self._observed_class_distribution,
                                             self._attribute_observers)
