# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert
from typing import Optional
import warnings

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from recpack.algorithms.base import Algorithm, TimeAwareAlgorithm
from recpack.matrix import InteractionMatrix
from recpack.util import get_top_K_values


class Random(Algorithm):
    """Uniform random algorithm, each item has an equal chance of getting recommended.

    Simple baseline, recommendations are sampled uniformly without replacement
    from the items that were interacted with in the matrix provided to fit.
    Scores are given based on sampling rank, such that the items first
    in the sample has the highest score

    :param K: How many items to sample for recommendation, defaults to 200
    :type K: int, optional
    :param seed: Seed for the random number generator used, defaults to None
    :type seed: int, optional
    :param use_only_interacted_items: Should only items visited in the training
        matrix be used to recommend from. If False all items will be recommended
        uniformly at random.
        Defaults to True.
    :type use_only_interacted_items: boolean, optional
    """

    def __init__(self, K: Optional[int] = 200, seed: Optional[int] = None, use_only_interacted_items: bool = True):
        super().__init__()
        self.items = None
        self.K = K  # TODO Allow K to be set to zero?
        self.use_only_interacted_items = use_only_interacted_items

        if seed is None:
            seed = np.random.get_state()[1][0]
        self.seed = seed
        self.rand_gen = np.random.default_rng(seed=self.seed)

    def _fit(self, X: csr_matrix) -> "Random":
        if self.use_only_interacted_items:
            self.items_ = list(set(X.nonzero()[1]))
        else:
            self.items_ = list(np.arange(X.shape[1]))

        if self.K is not None and len(self.items_) < self.K:
            warnings.warn("K is larger than the number of items.", UserWarning)

        return self

    def _predict(self, X: csr_matrix) -> csr_matrix:
        # For each user choose random K items, and generate a score for these items
        # Then create a matrix with the scores on the right indices
        users = list(set(X.nonzero()[0]))

        num_items = X.shape[1]
        K = min(len(self.items_), self.K) if self.K is not None else self.K
        # Generate random scores for all items
        random_scores = self.rand_gen.random((len(users), num_items))

        # Filter out only allowed items
        allowed_items = np.zeros(num_items)
        allowed_items[self.items_] = 1
        # Get top K of allowed items per user
        top_scores = get_top_K_values(csr_matrix(random_scores * allowed_items), K=K)

        X_pred = csr_matrix(X.shape)
        X_pred[users] = top_scores

        return X_pred


class Popularity(Algorithm):
    """Baseline algorithm recommending the most popular items in training data.

    During training the occurrences of each item is counted,
    and then normalized by dividing each count by the max count over items.
    As a result, all users are recommended the same items and all scores are between zero and one.

    :param K: How many items to recommend when predicting, defaults to 200
    :type K: int, optional
    """

    def __init__(self, K: int = 200):
        super().__init__()
        self.K = K

    def _fit(self, X: csr_matrix) -> "Popularity":
        # Get popularity score for every item
        interaction_counts = X.sum(axis=0).A[0]
        sorted_scores = interaction_counts / interaction_counts.max()

        num_items = X.shape[1]
        if num_items < self.K:
            warnings.warn("K is larger than the number of items.", UserWarning)

        K = min(self.K, num_items)
        ind = np.argpartition(sorted_scores, -K)[-K:]
        a = np.zeros(X.shape[1])
        a[ind] = sorted_scores[ind]
        self.sorted_scores_ = a
        return self

    def _predict(self, X: csr_matrix) -> csr_matrix:
        """For each user predict the K most popular items"""

        users = list(set(X.nonzero()[0]))

        X_pred = lil_matrix(X.shape)
        X_pred[users] = self.sorted_scores_

        return X_pred.tocsr()


class TimeAwarePopularity(TimeAwareAlgorithm):
    """Baseline algorithm recommending the most popular items,
    where each interaction is weighted by its recency.

    During training each interaction is given an exponentially decayed weight

    .. math::

        w(t) = e^{- \\frac{t_{max} - t}{decay}}

    where :math:`t_{max}` is the maximal timestamp in the training data.
    The score of an item is the sum of the weights of its interactions,
    normalized by the maximal score over items,
    so all scores fall in the interval [0, 1].

    If ``decay`` is None or not strictly positive,
    every interaction gets weight 1,
    which makes the ranking identical to the plain
    :class:`Popularity` algorithm.

    :param K: How many items to recommend when predicting, defaults to 200
    :type K: int, optional
    :param decay: Time scale of the exponential decay, in the same unit
        as the timestamps in the InteractionMatrix
        (e.g. ``decay=86400.`` is a one day time scale for epoch timestamps in seconds).
        If None, no decay is applied. Defaults to None.
    :type decay: float, optional
    """

    def __init__(self, K: int = 200, decay: Optional[float] = None):
        super().__init__()
        self.K = K
        self.decay = decay

    def _fit(self, X: InteractionMatrix) -> "TimeAwarePopularity":
        timestamps = X.timestamps

        if self.decay is not None and self.decay > 0:
            weights = np.exp(-(timestamps.max() - timestamps) / self.decay)
            item_scores = weights.groupby(level=1).sum()
        else:
            # Weight 1 per interaction, equivalent to plain Popularity counts.
            item_scores = timestamps.groupby(level=1).count()

        num_items = X.shape[1]
        scores = np.zeros(num_items)
        scores[item_scores.index.values] = item_scores.values
        scores = scores / scores.max()

        if num_items < self.K:
            warnings.warn("K is larger than the number of items.", UserWarning)

        K = min(self.K, num_items)
        ind = np.argpartition(scores, -K)[-K:]
        a = np.zeros(num_items)
        a[ind] = scores[ind]
        self.sorted_scores_ = a
        return self

    def _predict(self, X: InteractionMatrix) -> csr_matrix:
        """For each active user predict the K most popular scores"""

        users = list(X.active_users)

        X_pred = lil_matrix(X.shape)
        X_pred[users] = self.sorted_scores_

        return X_pred.tocsr()
