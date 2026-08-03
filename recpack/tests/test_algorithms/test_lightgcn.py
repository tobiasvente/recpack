# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import os

import numpy as np
import pytest
from scipy.sparse import csr_matrix
import torch

from recpack.algorithms import LightGCN
from recpack.algorithms.lightgcn import LightGCNModule


@pytest.fixture(scope="function")
def X_in_for_pairwise():
    pv_users, pv_items, pv_values = (
        [0, 0, 0, 1, 1, 1, 3, 3, 4, 4],
        [0, 1, 2, 0, 1, 2, 3, 4, 3, 4],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    )

    pv = csr_matrix((pv_values, (pv_users, pv_items)), shape=(10, 5))

    return pv


def test_lightgcn(X_in):
    a = LightGCN(num_components=2, num_layers=2, max_epochs=2, batch_size=1)
    a.fit(X_in, (X_in, X_in))

    pred = a.predict(X_in)

    # Users should be the exact same.
    assert set(pred.nonzero()[0]) == set(X_in.nonzero()[0])


def test_lightgcn_topK(X_in):
    a = LightGCN(num_components=2, num_layers=2, max_epochs=2, batch_size=1, predict_topK=1)

    a.fit(X_in, (X_in, X_in))

    pred = a.predict(X_in)

    assert set(pred.nonzero()[0]) == set(X_in.nonzero()[0])
    # Each user should receive a single recommendation
    assert pred.nonzero()[1].shape[0] == len(set(X_in.nonzero()[0]))


def test_lightgcn_w_interaction_mat(X_in_interaction_m):
    a = LightGCN(num_components=2, num_layers=2, max_epochs=2, batch_size=1)
    a.fit(X_in_interaction_m, (X_in_interaction_m, X_in_interaction_m))

    pred = a.predict(X_in_interaction_m)

    # Users should be the exact same.
    assert set(pred.nonzero()[0]) == set(X_in_interaction_m.active_users)


def test_lightgcn_normalized_adjacency(X_in_for_pairwise):
    module = LightGCNModule(
        X_in_for_pairwise.shape[0], X_in_for_pairwise.shape[1], X_in_for_pairwise, num_components=2, num_layers=1
    )

    adjacency = module.norm_adjacency.to_dense().numpy()

    num_users = X_in_for_pairwise.shape[0]

    # The adjacency matrix is symmetric.
    np.testing.assert_almost_equal(adjacency, adjacency.T)

    # The user-user and item-item blocks are zero.
    assert not adjacency[:num_users, :num_users].any()
    assert not adjacency[num_users:, num_users:].any()

    # User 0 interacted with items 0, 1, 2: degree 3.
    # Item 0 was interacted with by users 0, 1: degree 2.
    np.testing.assert_almost_equal(adjacency[0, num_users + 0], 1 / (np.sqrt(3) * np.sqrt(2)))

    # Users without interactions have empty rows.
    assert not adjacency[2, :].any()


@pytest.mark.parametrize("seed", list(range(1, 6)))
def test_lightgcn_pairwise_ranking(X_in_for_pairwise, seed):
    """Tests that the pairwise ranking of 2 items is correctly computed."""

    a = LightGCN(
        num_components=4,
        num_layers=2,
        max_epochs=10,
        batch_size=10,
        seed=seed,
        learning_rate=0.2,
        sample_size=50,
    )
    a.fit(X_in_for_pairwise, (X_in_for_pairwise, X_in_for_pairwise))
    pred = a.predict(X_in_for_pairwise)

    # The interaction graph has two disconnected components:
    # users {0, 1} with items {0, 1, 2}, and users {3, 4} with items {3, 4}.
    # Items in the user's component should be scored above the others.
    assert pred[1, 2] > pred[1, 4]
    assert pred[1, 1] > pred[1, 4]
    assert pred[1, 0] > pred[1, 4]
    assert pred[1, 2] > pred[1, 3]
    assert pred[1, 1] > pred[1, 3]
    assert pred[1, 0] > pred[1, 3]

    assert pred[3, 3] > pred[3, 0]
    assert pred[3, 4] > pred[3, 0]
    assert pred[3, 3] > pred[3, 1]
    assert pred[3, 4] > pred[3, 1]


def test_lightgcn_save_and_load(X_in_for_pairwise):
    a = LightGCN(
        num_components=4,
        num_layers=2,
        max_epochs=1,
        batch_size=2,
        seed=42,
        learning_rate=0.05,
        save_best_to_file=True,
    )

    a.fit(X_in_for_pairwise, (X_in_for_pairwise, X_in_for_pairwise))

    assert os.path.isfile(a.filename)

    b = LightGCN(
        num_components=4,
        num_layers=2,
        max_epochs=40,
        batch_size=2,
        seed=42,
        learning_rate=0.05,
        save_best_to_file=True,
    )

    b.load(a.filename)

    np.testing.assert_array_equal(
        a.predict(X_in_for_pairwise).toarray(),
        b.predict(X_in_for_pairwise).toarray(),
    )

    # Cleanup
    os.remove(a.filename)


def test_lightgcn_seed_is_set():
    a = LightGCN()
    assert hasattr(a, "seed")

    b = LightGCN(seed=42)
    assert b.seed == 42


def test_lightgcn_in_registry():
    from recpack.pipelines.registries import ALGORITHM_REGISTRY

    assert "LightGCN" in ALGORITHM_REGISTRY
