# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging

import numpy as np
import scipy.sparse
from scipy.sparse import csr_array

from recpack.metrics.base import ListwiseMetricK
from recpack.metrics.util import sparse_divide_nonzero


logger = logging.getLogger("recpack")


class RecallK(ListwiseMetricK):
    """Computes the fraction of true interactions that made it into
    the Top-K recommendations.

    Recall per user is computed as:

    .. math::

        \\text{Recall}(u) = \\frac{\\sum\\limits_{i \\in \\text{Top-K}(u)} y^{true}_{u,i} }{\\sum\\limits_{j \\in I} y^{true}_{u,j}}

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    def __init__(self, K):
        super().__init__(K)

    def _calculate(self, y_true: csr_array, y_pred_top_K: csr_array) -> None:
        scores = scipy.sparse.lil_array(y_pred_top_K.shape)

        # Elementwise multiplication of top K predicts and true interactions
        scores[y_pred_top_K.multiply(y_true).astype(bool)] = 1

        scores = scores.tocsr()

        self.scores_ = csr_array(sparse_divide_nonzero(scores, csr_array(y_true.sum(axis=1)[:, None])).sum(axis=1)[:, None])

        return


def recall_k(y_true, y_pred, k=50):
    r = RecallK(K=k)
    r.calculate(y_true, y_pred)

    return r.value


class CalibratedRecallK(ListwiseMetricK):
    """Computes number of Top-K recommendations that were hits divided
    by the minimum of K and number of true interactions of the user.

    This differs from recall as we know it in that it accounts for when K < #true,
    because we can't expect a list of K recommendations to cover more than K true interactions.

    .. math::

        \\text{CalibratedRecall}(u) = \\frac{\\sum\\limits_{i \\in \\text{topK}(u)} y^{true}_{u,i} }{\\text{min}(\\sum\\limits_{j \\in I} y^{true}_{u,j}, K)}

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    def __init__(self, K):
        super().__init__(K)

    def _calculate(self, y_true: csr_array, y_pred_top_K: csr_array) -> None:
        scores = scipy.sparse.lil_array(y_pred_top_K.shape)

        # Elementwise multiplication of top K predicts and true interactions
        scores[y_pred_top_K.multiply(y_true).astype(bool)] = 1
        scores = scores.tocsr()

        optimal = csr_array(np.minimum(y_true.sum(axis=1), self.K)[:, None])

        self.scores_ = csr_array(sparse_divide_nonzero(scores, optimal).sum(axis=1)[:, None])


def calibrated_recall_k(y_true, y_pred, k):
    r = CalibratedRecallK(K=k)
    r.calculate(y_true, y_pred)

    return r.value
