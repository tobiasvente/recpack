# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from recpack.algorithms import Popularity, Random, TimeAwarePopularity
from recpack.matrix import InteractionMatrix


def test_random(X_in):
    seed = 42
    K = 2
    algo = Random(K=K, seed=seed)
    algo.fit(X_in)

    X_pred = algo.predict(X_in)

    assert X_pred.nnz == len(set(X_pred.nonzero()[0])) * K


def test_random_use_only_interacted_items(X_in):
    algo1 = Random(K=2, use_only_interacted_items=True)
    algo2 = Random(K=2, use_only_interacted_items=False)
    algo1.fit(X_in)
    algo2.fit(X_in)
    assert len(algo1.items_) < len(algo2.items_)


def test_random_K_is_None(X_in):
    algo1 = Random(K=None, use_only_interacted_items=True)
    algo2 = Random(K=None, use_only_interacted_items=False)
    algo1.fit(X_in)
    algo2.fit(X_in)
    assert len(algo1.items_) < len(algo2.items_)

    X_pred = algo1.predict(X_in)

    assert len(set(X_pred.nonzero()[1]).difference(set(X_in.nonzero()[1]))) == 0


def test_popularity():
    item_i = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    user_i = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    values = [1] * 10
    train_data = csr_matrix((values, (user_i, item_i)))
    algo = Popularity(K=20)

    algo.fit(train_data)

    _in = csr_matrix(([1, 1], ([0, 1], [1, 1])), shape=(5, 5))
    prediction = algo.predict(_in)

    # All users in _in get the same recommendations
    np.testing.assert_almost_equal(prediction[0, :].toarray(), prediction[1, :].toarray())
    # The most popular item is ranked highest
    assert prediction[0, 4] > prediction[0, 3]
    # Users who were not in _in do not receive any recommendations
    assert prediction[2, :].nnz == 0


def test_popularity_K_larger_than_num_items():
    item_i = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    user_i = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    values = [1] * 10
    train_data = csr_matrix((values, (user_i, item_i)))
    algo = Popularity(K=20)
    with warnings.catch_warnings(record=True) as w:
        algo.fit(train_data)
        assert len(w) > 0
        assert "K is larger than the number of items." in str(w[-1].message)


def _make_timed_interaction_matrix(users, items, timestamps, shape):
    df = pd.DataFrame(
        {
            InteractionMatrix.USER_IX: users,
            InteractionMatrix.ITEM_IX: items,
            InteractionMatrix.TIMESTAMP_IX: timestamps,
        }
    )
    return InteractionMatrix(
        df,
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        timestamp_ix=InteractionMatrix.TIMESTAMP_IX,
        shape=shape,
    )


def test_time_aware_popularity_no_decay_matches_popularity():
    item_i = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    user_i = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    timestamps = list(range(10))
    shape = (5, 5)

    X = _make_timed_interaction_matrix(user_i, item_i, timestamps, shape)

    algo = TimeAwarePopularity(K=5, decay=None)
    algo.fit(X)

    pop = Popularity(K=5)
    pop.fit(csr_matrix(([1] * 10, (user_i, item_i)), shape=shape))

    # Without decay every interaction gets weight 1,
    # so scores match plain count-based popularity.
    np.testing.assert_almost_equal(algo.sorted_scores_, pop.sorted_scores_)


def test_time_aware_popularity_decay_boosts_recent_items():
    # Item 1 has three old interactions, item 2 has two recent ones.
    user_i = [0, 1, 2, 3, 4]
    item_i = [1, 1, 1, 2, 2]
    timestamps = [0, 0, 0, 100, 100]
    shape = (5, 5)

    X = _make_timed_interaction_matrix(user_i, item_i, timestamps, shape)

    no_decay = TimeAwarePopularity(K=5, decay=None)
    no_decay.fit(X)
    # Without decay, raw counts win: item 1 > item 2.
    assert no_decay.sorted_scores_[1] > no_decay.sorted_scores_[2]

    strong_decay = TimeAwarePopularity(K=5, decay=10.0)
    strong_decay.fit(X)
    # With strong decay the old interactions are worth almost nothing,
    # so the recently popular item 2 ranks first.
    assert strong_decay.sorted_scores_[2] > strong_decay.sorted_scores_[1]


def test_time_aware_popularity_rejects_csr_matrix(X_in):
    algo = TimeAwarePopularity(K=2)

    with pytest.raises(TypeError) as type_error:
        algo.fit(X_in)

    assert type_error.match(".* requires Interaction Matrix as input.")


def test_time_aware_popularity_rejects_matrix_without_timestamps(X_in):
    algo = TimeAwarePopularity(K=2)

    with pytest.raises(ValueError) as value_error:
        algo.fit(InteractionMatrix.from_csr_matrix(X_in))

    assert value_error.match(".* requires timestamp information in the InteractionMatrix.")


def test_time_aware_popularity_predict_rejects_matrix_without_timestamps(X_in):
    user_i = [0, 1, 2, 3, 4]
    item_i = [1, 1, 1, 2, 2]
    timestamps = [0, 0, 0, 100, 100]

    X = _make_timed_interaction_matrix(user_i, item_i, timestamps, (10, 5))

    algo = TimeAwarePopularity(K=2)
    algo.fit(X)

    with pytest.raises(ValueError) as value_error:
        algo.predict(InteractionMatrix.from_csr_matrix(X_in))

    assert value_error.match(".* requires timestamp information in the InteractionMatrix.")
