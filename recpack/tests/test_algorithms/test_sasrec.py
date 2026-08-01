# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
import torch

from recpack.algorithms import SASRec
from recpack.matrix import InteractionMatrix
from recpack.tests.test_algorithms.util import assert_changed, assert_same
from recpack.tests.test_algorithms.test_loss_functions import sigmoid


@pytest.fixture(scope="function")
def cyclic_sessions():
    """Sessions following the deterministic cycle 0 -> 1 -> 2 -> 3 -> 0 -> ...

    16 users, with all combinations of starting offset (0-3)
    and session length (5-8). The next item of every session
    is fully determined by its last item.
    """
    rows = []
    uid = 0
    for offset in range(4):
        for length in (5, 6, 7, 8):
            for t in range(length):
                rows.append((uid, (offset + t) % 4, t))
            uid += 1
    df = pd.DataFrame(rows, columns=["uid", "iid", "ts"])
    return InteractionMatrix(df, "iid", "uid", timestamp_ix="ts")


@pytest.fixture(scope="function")
def sasrec():
    return SASRec(
        seed=42,
        batch_size=3,
        num_components=16,
        num_blocks=1,
        num_heads=1,
        max_len=10,
        dropout=0.0,
        num_negatives=10,
        max_epochs=30,
        learning_rate=0.05,
        keep_last=True,
    )


@pytest.fixture(scope="function")
def sasrec_topK():
    return SASRec(
        seed=42,
        batch_size=3,
        num_components=16,
        num_blocks=1,
        num_heads=1,
        max_len=10,
        dropout=0.0,
        num_negatives=10,
        max_epochs=30,
        learning_rate=0.05,
        keep_last=True,
        predict_topK=1,
    )


def test_sasrec_compute_loss(sasrec):
    output = torch.FloatTensor(
        [
            [[0, 0.1, 0.8, 0.1, 0], [0, 0.8, 0.2, 0, 0]],
            [[0, 0.1, 0.8, 0.1, 0], [0, 0.8, 0.2, 0, 0]],
            [[0, 0.1, 0.8, 0.1, 0], [0, 0, 0, 0, 1]],
        ]
    )

    targets_chunk = torch.LongTensor([[2, 1], [2, 1], [2, 4]])

    negatives_chunk = torch.LongTensor([[[1, 3], [2, 2]], [[1, 3], [2, 3]], [[3, 0], [3, 3]]])

    true_input_mask = torch.BoolTensor([[True, True], [True, True], [True, True]])

    loss = sasrec._compute_loss(output, targets_chunk, negatives_chunk, true_input_mask)

    # 6 positives and 12 negatives -> 18 scores.
    expected_loss = (
        -(
            # positives: log(sigmoid(positive score))
            np.log(sigmoid(0.8))
            + np.log(sigmoid(0.8))
            + np.log(sigmoid(0.8))
            + np.log(sigmoid(0.8))
            + np.log(sigmoid(0.8))
            + np.log(sigmoid(1.0))
            # negatives: log(1 - sigmoid(negative score))
            + np.log(1 - sigmoid(0.1))
            + np.log(1 - sigmoid(0.1))
            + np.log(1 - sigmoid(0.2))
            + np.log(1 - sigmoid(0.2))
            + np.log(1 - sigmoid(0.1))
            + np.log(1 - sigmoid(0.1))
            + np.log(1 - sigmoid(0.2))
            + np.log(1 - sigmoid(0.0))
            + np.log(1 - sigmoid(0.1))
            + np.log(1 - sigmoid(0.0))
            + np.log(1 - sigmoid(0.0))
            + np.log(1 - sigmoid(0.0))
        )
        / 18
    )
    np.testing.assert_almost_equal(loss, expected_loss, decimal=6)

    # Mask out the middle sequence: 4 positives and 8 negatives remain.
    true_input_mask = torch.BoolTensor([[True, True], [False, False], [True, True]])
    loss = sasrec._compute_loss(output, targets_chunk, negatives_chunk, true_input_mask)

    expected_loss = (
        -(
            np.log(sigmoid(0.8))
            + np.log(sigmoid(0.8))
            + np.log(sigmoid(0.8))
            + np.log(sigmoid(1.0))
            + np.log(1 - sigmoid(0.1))
            + np.log(1 - sigmoid(0.1))
            + np.log(1 - sigmoid(0.2))
            + np.log(1 - sigmoid(0.2))
            + np.log(1 - sigmoid(0.1))
            + np.log(1 - sigmoid(0.0))
            + np.log(1 - sigmoid(0.0))
            + np.log(1 - sigmoid(0.0))
        )
        / 12
    )
    np.testing.assert_almost_equal(loss, expected_loss, decimal=6)


def test_sasrec_truncate_seq_batch(sasrec):
    sasrec.max_len = 3
    # pad_token is normally set in _init_model; pick one
    # that does not collide with the item ids used below.
    sasrec.pad_token = pad = 99

    # Two sequences: one longer than max_len, one shorter (right-padded).
    seq = torch.LongTensor([[1, 2, 3, 4, 5], [1, 2, pad, pad, pad]])
    aligned = torch.LongTensor([[10, 20, 30, 40, 50], [10, 20, pad, pad, pad]])
    negatives = aligned.unsqueeze(-1)

    t_seq, t_aligned, t_negatives = sasrec._truncate_seq_batch(seq, aligned, negatives)

    # The long sequence keeps its most recent max_len items.
    np.testing.assert_array_equal(t_seq[0].numpy(), [3, 4, 5])
    np.testing.assert_array_equal(t_aligned[0].numpy(), [30, 40, 50])
    np.testing.assert_array_equal(t_negatives[0].squeeze(-1).numpy(), [30, 40, 50])

    # The short sequence is unchanged, apart from the width.
    np.testing.assert_array_equal(t_seq[1].numpy(), [1, 2, pad])

    # Batches within max_len are returned untouched.
    short = torch.LongTensor([[1, 2], [3, pad]])
    (untouched,) = sasrec._truncate_seq_batch(short)
    assert untouched is short


def test_sasrec_training_epoch(sasrec, matrix_sessions):
    device = sasrec.device
    sasrec._init_model(matrix_sessions)

    # Each training epoch should update the parameters
    for _ in range(5):
        params = [np for np in sasrec.model_.named_parameters() if np[1].requires_grad]
        params_before = [(name, p.clone()) for (name, p) in params]

        sasrec._train_epoch(matrix_sessions)
        assert_changed(params_before, params, device)


def test_sasrec_evaluation_epoch(sasrec, matrix_sessions):
    device = sasrec.device

    sasrec.fit(matrix_sessions, (matrix_sessions, matrix_sessions))

    # Model evaluation should have no effect on parameters
    for _ in range(5):
        params = [np for np in sasrec.model_.named_parameters() if np[1].requires_grad]
        params_before = [(name, p.clone()) for (name, p) in params]

        sasrec._evaluate(matrix_sessions, matrix_sessions)
        assert_same(params_before, params, device)


def test_sasrec_predict(sasrec, matrix_sessions):
    sasrec.fit(matrix_sessions, (matrix_sessions, matrix_sessions))

    X_pred = sasrec.predict(matrix_sessions)

    # Prediction matrix should have same shape as input matrix
    assert isinstance(X_pred, csr_matrix)
    assert X_pred.shape == matrix_sessions.shape

    # All users with a history should have predictions
    assert set(matrix_sessions.values.nonzero()[0]) == set(X_pred.nonzero()[0])

    # All items should have a score
    assert len(set(X_pred.nonzero()[1])) == matrix_sessions.shape[1]


def test_sasrec_learns_sequential_pattern(cyclic_sessions):
    """SASRec should learn a deterministic next-item pattern.

    Every session follows the cycle 0 -> 1 -> 2 -> 3 -> 0,
    so the top recommendation for each user must be the successor
    of the last item in their session.
    """
    sasrec = SASRec(
        seed=42,
        batch_size=8,
        num_components=16,
        num_blocks=1,
        num_heads=1,
        max_len=10,
        dropout=0.0,
        num_negatives=1,
        max_epochs=60,
        learning_rate=0.05,
        keep_last=True,
    )
    sasrec.fit(cyclic_sessions, (cyclic_sessions, cyclic_sessions))

    top_item = sasrec.predict(cyclic_sessions).toarray().argmax(axis=1)

    uid = 0
    for offset in range(4):
        for length in (5, 6, 7, 8):
            expected_next = (offset + length) % 4
            assert top_item[uid] == expected_next
            uid += 1


def test_sasrec_predict_topK(sasrec_topK, matrix_sessions):
    sasrec_topK.fit(matrix_sessions, (matrix_sessions, matrix_sessions))

    X_pred = sasrec_topK.predict(matrix_sessions)

    # Prediction matrix should have same shape as input matrix
    assert isinstance(X_pred, csr_matrix)
    assert X_pred.shape == matrix_sessions.shape

    # All users with a history should have predictions
    assert set(matrix_sessions.values.nonzero()[0]) == set(X_pred.nonzero()[0])

    # Each user should receive only a single recommendation
    assert X_pred.nonzero()[1].shape[0] == len(set(matrix_sessions.nonzero()[0]))


def test_sasrec_truncated_history_predict(sasrec, matrix_sessions):
    # With a max_len smaller than the longest session,
    # prediction should still work on the most recent items.
    sasrec.max_len = 4
    sasrec.fit(matrix_sessions, (matrix_sessions, matrix_sessions))

    X_pred = sasrec.predict(matrix_sessions)

    assert set(matrix_sessions.values.nonzero()[0]) == set(X_pred.nonzero()[0])


def test_fit_no_interaction_matrix(sasrec, mat):
    with pytest.raises(TypeError):
        sasrec.fit(mat.binary_values, (mat, mat))
    with pytest.raises(TypeError):
        sasrec.fit(mat, (mat.binary_values, mat))
    with pytest.raises(TypeError):
        sasrec.fit(mat, (mat, mat.binary_values))


def test_fit_no_timestamps(sasrec, mat):
    with pytest.raises(ValueError):
        sasrec.fit(mat.eliminate_timestamps(), (mat, mat))
    with pytest.raises(ValueError):
        sasrec.fit(mat, (mat.eliminate_timestamps(), mat))
    with pytest.raises(ValueError):
        sasrec.fit(mat, (mat, mat.eliminate_timestamps()))


def test_predict_no_interaction_matrix(sasrec, mat):
    sasrec.fit(mat, (mat, mat))
    with pytest.raises(TypeError):
        sasrec.predict(mat.binary_values)


def test_predict_no_timestamps(sasrec, mat):
    sasrec.fit(mat, (mat, mat))
    with pytest.raises(ValueError):
        sasrec.predict(mat.eliminate_timestamps())


def test_sasrec_in_registry():
    from recpack.pipelines.registries import ALGORITHM_REGISTRY

    assert "SASRec" in ALGORITHM_REGISTRY
