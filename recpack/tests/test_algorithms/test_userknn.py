# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Marouan El Marnissi

import math

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from recpack.algorithms import UserKNN


@pytest.fixture(scope="function")
def data():
    values = [1] * 7
    users = [0, 0, 1, 1, 2, 2, 2]
    items = [1, 2, 0, 2, 0, 1, 2]
    d = csr_matrix((values, (users, items)), shape=(3, 3))

    return d

@pytest.fixture(scope="function")
def data_empty_row():
    values = [1] * 5
    users = [0, 0, 1, 1, 1]
    items = [1, 2, 0, 1, 2]
    d = csr_matrix((values, (users, items)), shape=(3, 3))

    return d


def test_user_knn(data):

    algo = UserKNN(K=2)

    algo.fit(data)

    # diagonal is set to zero because self similarity is removed
    expected_similarities = np.array(
        [
            [0, 0.5, 2 / math.sqrt(6)],
            [0.5, 0, 2 / math.sqrt(6)],
            [2 / math.sqrt(6), 2 / math.sqrt(6), 0]
        ]
    )
    np.testing.assert_almost_equal(algo.similarity_matrix_.toarray(), expected_similarities)

    # Make sure the similarities recommended are the cosine similarities as computed.
    # If we create users with a single item seen in order.
    _in = csr_matrix(([1, 1, 1], ([0, 1, 2], [0, 1, 2])), shape=(3, 3))
    result = algo.predict(_in)

    np.testing.assert_almost_equal(result.toarray(), expected_similarities)

    # Make sure similarities are added correctly.
    _in = csr_matrix(([1, 1], ([0, 1], [0, 0])), shape=(3, 1))
    expected_out = [[0.5],
                    [0.5],
                    [4 / math.sqrt(6)]]
    result = algo.predict(_in)
    np.testing.assert_almost_equal(result.toarray(), expected_out)

def test_user_knn_empty_row(data_empty_row):
    algo = UserKNN(K=2)

    algo.fit(data_empty_row)
    # user without interactions should have a similarity score of 0 with every other user
    expected_similarities = np.array([[0, 2 / math.sqrt(6), 0], [2 / math.sqrt(6), 0, 0], [0, 0, 0]])
    np.testing.assert_almost_equal(algo.similarity_matrix_.toarray(), expected_similarities)