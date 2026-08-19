# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import os

import numpy as np
import pytest
from scipy.sparse import csr_array
import torch

from recpack.algorithms import BPRKNN, BPRMF
from recpack.algorithms.bpr import BPRKNNModule, FactorizedBPRKNNModule, MFModule


@pytest.fixture(scope="function")
def X_in_for_pairwise():
    pv_users, pv_items, pv_values = (
        [0, 0, 0, 1, 1, 1, 3, 3, 4, 4],
        [0, 1, 2, 0, 1, 2, 3, 4, 3, 4],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    )

    pv = csr_array((pv_values, (pv_users, pv_items)), shape=(10, 5))

    return pv


@pytest.fixture(scope="function")
def X_in_for_bprknn():
    return csr_array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ]
    )


def test_bprmf(X_in):
    a = BPRMF(num_components=2, max_epochs=2, batch_size=1)
    a.fit(X_in, (X_in, X_in))

    pred = a.predict(X_in)

    # Users should be the exact same.
    assert set(pred.nonzero()[0]) == set(X_in.nonzero()[0])


def test_bprmf_topK(X_in):
    a = BPRMF(num_components=2, max_epochs=2, batch_size=1, predict_topK=1)

    a.fit(X_in, (X_in, X_in))

    pred = a.predict(X_in)

    assert set(pred.nonzero()[0]) == set(X_in.nonzero()[0])
    # Each user should receive a single recommendation
    assert pred.nonzero()[1].shape[0] == len(set(X_in.nonzero()[0]))


def test_bprmf_w_interaction_mat(X_in_interaction_m):
    a = BPRMF(num_components=2, max_epochs=2, batch_size=1)
    a.fit(X_in_interaction_m, (X_in_interaction_m, X_in_interaction_m))

    pred = a.predict(X_in_interaction_m)

    # Users should be the exact same.
    assert set(pred.nonzero()[0]) == set(X_in_interaction_m.active_users)


@pytest.mark.parametrize("seed", list(range(1, 25)))
def test_bprmf_pairwise_ranking(X_in_for_pairwise, seed):
    """Tests that the pairwise ranking of 2 items is correctly computed."""

    a = BPRMF(
        num_components=4,
        max_epochs=10,
        batch_size=10,
        seed=seed,
        learning_rate=0.5,
        sample_size=50,
    )
    a.fit(X_in_for_pairwise, (X_in_for_pairwise, X_in_for_pairwise))
    pred = a.predict(X_in_for_pairwise)

    # Negative example scores should be lower than positive
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


def test_bprmf_save_and_load(X_in_for_pairwise):
    a = BPRMF(
        num_components=4,
        max_epochs=1,
        batch_size=2,
        seed=42,
        learning_rate=0.05,
        save_best_to_file=True,
    )

    a.fit(X_in_for_pairwise, (X_in_for_pairwise, X_in_for_pairwise))

    assert os.path.isfile(a.filename)

    b = BPRMF(
        num_components=4,
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

    # TODO cleanup
    os.remove(a.filename)


def test_bprmf_forward(X_in_for_pairwise):
    a = MFModule(3, 3, 2)

    U = torch.LongTensor([0, 1])
    I = torch.LongTensor([0, 2])

    res_1 = a.forward(U, I)
    res_2 = a.forward(U[1], I[1])

    assert res_2 == res_1[1, 1]


def test_bprmf_bad_stopping_criterion(X_in):
    with pytest.raises(ValueError):
        BPRMF(stopping_criterion="not_a_correct_value")


def test_bprmf_recall_stopping_criterion(X_in):
    a = BPRMF(num_components=2, max_epochs=2, batch_size=1, stopping_criterion="recall")
    a.fit(X_in, (X_in, X_in))


def test_bprknn(X_in):
    """Tests that BPRKNN fits and predicts scores for every active user."""
    a = BPRKNN(K=None, max_epochs=2, batch_size=1)
    a.fit(X_in, (X_in, X_in))

    pred = a.predict(X_in)

    assert set(pred.nonzero()[0]) == set(X_in.nonzero()[0])


def test_bprknn_topK(X_in):
    """Tests that BPRKNN retains only the requested number of predictions per user."""
    a = BPRKNN(K=None, max_epochs=2, batch_size=1, predict_topK=1)
    a.fit(X_in, (X_in, X_in))

    pred = a.predict(X_in)

    assert set(pred.nonzero()[0]) == set(X_in.nonzero()[0])
    assert pred.nonzero()[1].shape[0] == len(set(X_in.nonzero()[0]))


def test_bprknn_w_interaction_mat(X_in_interaction_m):
    """Tests that BPRKNN accepts an InteractionMatrix as input."""
    a = BPRKNN(K=None, max_epochs=2, batch_size=1)
    a.fit(X_in_interaction_m, (X_in_interaction_m, X_in_interaction_m))

    pred = a.predict(X_in_interaction_m)

    assert set(pred.nonzero()[0]) == set(X_in_interaction_m.active_users)

@pytest.mark.parametrize("seed", list(range(1, 25)))
def test_bprknn_pairwise_ranking(X_in_for_bprknn, seed):
    """Tests that direct BPRKNN ranks related items above unrelated items."""
    algorithm = BPRKNN(K=None, max_epochs=20, learning_rate=0.1, seed=seed)

    algorithm.fit(X_in_for_bprknn, (X_in_for_bprknn, X_in_for_bprknn))
    prediction_input = csr_array(
        [
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    predictions = algorithm.predict(prediction_input)

    assert predictions[0, 1] > predictions[0, 2]
    assert predictions[0, 1] > predictions[0, 3]


def test_bprknn_similarities(X_in_for_bprknn):
    """Tests that direct BPRKNN learns reproducible, symmetric similarities with a zero diagonal."""
    first = BPRKNN(K=None, max_epochs=3, seed=42).fit(
        X_in_for_bprknn, (X_in_for_bprknn, X_in_for_bprknn)
    )
    second = BPRKNN(K=None, max_epochs=3, seed=42).fit(
        X_in_for_bprknn, (X_in_for_bprknn, X_in_for_bprknn)
    )

    np.testing.assert_array_equal(first.similarity_matrix_.toarray(), second.similarity_matrix_.toarray())
    np.testing.assert_array_equal(first.similarity_matrix_.diagonal(), 0)
    np.testing.assert_array_equal(first.similarity_matrix_.toarray(), first.similarity_matrix_.toarray().T)


def test_bprknn_top_k_neighbours(X_in_for_bprknn):
    """Tests that direct BPRKNN retains at most K neighbours per item."""
    algorithm = BPRKNN(K=1, max_epochs=3, seed=42).fit(
        X_in_for_bprknn, (X_in_for_bprknn, X_in_for_bprknn)
    )

    assert np.all(np.diff(algorithm.similarity_matrix_.indptr) <= 1)


def test_bprknn_save_and_load(X_in_for_bprknn):
    """Tests that a saved BPRKNN model produces identical scores after loading."""
    a = BPRKNN(
        K=None,
        max_epochs=1,
        batch_size=2,
        seed=42,
        learning_rate=0.05,
        save_best_to_file=True,
    )
    a.fit(X_in_for_bprknn, (X_in_for_bprknn, X_in_for_bprknn))

    assert os.path.isfile(a.filename)

    b = BPRKNN(K=None)
    b.load(a.filename)

    np.testing.assert_array_equal(
        a.predict(X_in_for_bprknn).toarray(),
        b.predict(X_in_for_bprknn).toarray(),
    )

    os.remove(a.filename)


def test_bprknn_forward():
    """Tests that batched and selected BPRKNNModule scores are consistent."""
    model = BPRKNNModule(num_items=3)
    with torch.no_grad():
        # set similarity matrix
        model.item_similarity_.copy_(torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]]))
    history = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    items = torch.LongTensor([0, 2])

    scores = model.score_history(history, items)
    selected_score = model.score_history(history[1:2], items[1:2])

    assert selected_score[0, 0] == scores[1, 1]


def test_bprknn_bad_stopping_criterion():
    """Tests that BPRKNN rejects an unknown stopping criterion."""
    with pytest.raises(ValueError):
        BPRKNN(stopping_criterion="unknown")


def test_bprknn_recall_stopping_criterion(X_in):
    """Tests that BPRKNN can train using recall as its stopping criterion."""
    a = BPRKNN(K=None, max_epochs=2, batch_size=1, stopping_criterion="recall")
    a.fit(X_in, (X_in, X_in))


def test_factorized_scores():
    """Tests that factorized BPRKNN scores match an explicitly constructed similarity matrix."""
    model = FactorizedBPRKNNModule(num_items=3, num_components=2)
    with torch.no_grad():
        model.item_embedding_.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]))

    history = torch.tensor([[1.0, 1.0, 0.0]])
    items = torch.LongTensor([0, 2])

    scores = model.score_history(history, items)

    # H H^T with its diagonal removed is [[0, 0, 1], [0, 0, 2], [1, 2, 0]].
    torch.testing.assert_close(scores, torch.tensor([[0.0, 3.0]]))


def test_factorized_parameter_count():
    """Tests that factorized BPRKNN stores a linear number of item parameters."""
    model = FactorizedBPRKNNModule(num_items=5, num_components=2)

    assert sum(parameter.numel() for parameter in model.parameters()) == 5 * 2


def test_factorized_ranking(X_in_for_bprknn):
    """Tests that factorized BPRKNN ranks related items above unrelated items."""
    algorithm = BPRKNN(
        similarity_mode="factorized",
        num_components=2,
        max_epochs=20,
        learning_rate=0.1,
        seed=42,
    )

    algorithm.fit(X_in_for_bprknn, (X_in_for_bprknn, X_in_for_bprknn))
    predictions = algorithm.predict(csr_array([[1, 0, 0, 0]] * 4))

    assert predictions[0, 1] > predictions[0, 2]
    assert predictions[0, 1] > predictions[0, 3]
    assert algorithm.item_embedding_.shape == (4, 2)
    assert not hasattr(algorithm, "similarity_matrix_")


def test_bprknn_invalid_mode():
    """Tests that BPRKNN rejects an unknown similarity representation."""
    with pytest.raises(ValueError):
        BPRKNN(similarity_mode="unknown")
