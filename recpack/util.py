# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix

logger = logging.getLogger("recpack")


def to_tuple(el):
    """Return an object as a tuple.

    Existing tuples are returned unchanged; every other object is wrapped in a
    one-element tuple.

    :param el: Object to return as a tuple.
    :type el: Any
    :return: ``el`` itself if it is a tuple, otherwise ``(el,)``.
    :rtype: tuple
    """
    if type(el) == tuple:
        return el
    else:
        return (el,)


def df_to_sparse(df, item_ix, user_ix, value_ix=None, shape=None):
    """Construct a sparse user-item matrix from a pandas DataFrame.

    Each row in ``df`` becomes an entry at ``(user, item)``.  When a value
    column is supplied, its values are stored in the matrix.  Otherwise every
    row contributes one.  Duplicate user-item pairs are summed by SciPy, so an
    unweighted DataFrame produces interaction counts for duplicate pairs.

    :param df: Interaction data containing user and item identifier columns.
    :type df: pandas.DataFrame
    :param item_ix: Name of the column containing zero-based item indices.
    :type item_ix: str
    :param user_ix: Name of the column containing zero-based user indices.
    :type user_ix: str
    :param value_ix: Optional name of the values column. If the column is not
        present, ones are used and a warning is logged.
    :type value_ix: str, optional
    :param shape: Explicit ``(number_of_users, number_of_items)`` shape. If
        omitted, the shape is inferred from the largest user and item indices.
    :type shape: tuple(int, int), optional
    :return: Sparse matrix with users on rows and items on columns.
    :rtype: scipy.sparse.csr_matrix

    .. note::
        An explicit ``shape`` is required for an empty DataFrame because no
        maximum user or item index is available for inference.
    """
    if value_ix is not None and value_ix in df:
        values = df[value_ix]
    else:
        if value_ix is not None:
            # value_ix provided, but not in df
            logger.warning(f"Value column {value_ix} not found in dataframe. Using ones instead.")

        num_entries = df.shape[0]
        # Scipy sums up the entries when an index-pair occurs more than once,
        # resulting in the actual counts being stored. Neat!
        values = np.ones(num_entries)

    indices = list(zip(*df.loc[:, [user_ix, item_ix]].values))

    if indices == []:
        indices = [[], []]  # Empty zip does not evaluate right

    if shape is None:
        shape = df[user_ix].max() + 1, df[item_ix].max() + 1
    sparse_matrix = csr_matrix((values, indices), shape=shape, dtype=values.dtype)

    return sparse_matrix


def get_top_K_ranks(X: csr_matrix, K: Optional[int] = None) -> csr_matrix:
    """Returns a matrix of ranks assigned to the largest K values in X.

    Selects K largest values for every row in X and assigns a rank to each.
    Rank one corresponds to the largest stored value. If ``K`` is omitted, all
    stored values are ranked. Empty rows remain empty.

    :param X: Matrix from which we will select K values in every row.
    :type X: csr_matrix
    :param K: Maximum number of stored values to select per row. ``None``
        selects all stored values.
    :type K: int, optional
    :return: Matrix with ranks at the selected coordinates and zeros elsewhere.
    :rtype: csr_matrix
    """
    U, I, V = [], [], []
    for row_ix, (le, ri) in enumerate(zip(X.indptr[:-1], X.indptr[1:])):
        K_row_pick = min(K, ri - le) if K is not None else ri - le

        if K_row_pick != 0:

            top_k_row = X.indices[le + np.argpartition(X.data[le:ri], list(range(-K_row_pick, 0)))[-K_row_pick:]]

            for rank, col_ix in enumerate(reversed(top_k_row)):
                U.append(row_ix)
                I.append(col_ix)
                V.append(rank + 1)

    X_top_K = csr_matrix((V, (U, I)), shape=X.shape)

    return X_top_K


def get_top_K_values(X: csr_matrix, K: Optional[int] = None) -> csr_matrix:
    """Returns a matrix of only the K largest values for every row in X.

    Selects the top-K items for every user (which is equal to the K nearest neighbours.)
    In case of a tie for the last position, the item with the largest index of the tied items is used.

    :param X: Matrix from which we will select K values in every row.
    :type X: csr_matrix
    :param K: Maximum number of stored values to retain per row. ``None``
        retains all stored values.
    :type K: int, optional
    :return: Matrix containing the selected input values at their original
        coordinates and zeros elsewhere.
    :rtype: csr_matrix
    """
    top_K_ranks = get_top_K_ranks(X, K)
    top_K_ranks[top_K_ranks > 0] = 1  # ranks to binary

    return top_K_ranks.multiply(X)  # elementwise multiplication


def to_binary(X: csr_matrix) -> csr_matrix:
    """Converts a matrix to binary by setting all non-zero values to 1.

    The returned matrix retains the input dtype and shape. The input matrix is
    not modified.

    :param X: Matrix to convert to binary.
    :type X: csr_matrix
    :return: Binary matrix with the same shape and dtype as ``X``.
    :rtype: csr_matrix
    """
    X_binary = X.astype(bool).astype(X.dtype)

    return X_binary
