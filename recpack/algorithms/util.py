# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from inspect import isgenerator
from itertools import islice
from typing import Iterator, List, Iterable, Union

import numpy as np
from scipy.sparse import csr_matrix
import torch

from recpack.matrix import InteractionMatrix, Matrix, to_binary


def swish(x):
    """Apply the element-wise swish activation function.

    Swish is defined as :math:`x \\cdot \\operatorname{sigmoid}(x)` and preserves
    the shape of the input tensor.

    :param x: Input tensor.
    :type x: torch.Tensor
    :return: Tensor containing the element-wise activation values.
    :rtype: torch.Tensor
    """
    return x.mul(torch.sigmoid(x))


def log_norm_pdf(x, mu, logvar):
    """Evaluate an element-wise normal log-density.

    ``logvar`` is the logarithm of the variance. All inputs follow PyTorch's
    broadcasting rules; no reduction is applied.

    :param x: Values at which to evaluate the density.
    :type x: torch.Tensor
    :param mu: Mean of the normal distribution.
    :type mu: torch.Tensor
    :param logvar: Log-variance of the normal distribution.
    :type logvar: torch.Tensor
    :return: Element-wise log-density values.
    :rtype: torch.Tensor
    """
    return -0.5 * (logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


def naive_sparse2tensor(data: csr_matrix) -> torch.Tensor:
    """Naively converts sparse csr_matrix to torch Tensor.

    This conversion materializes the complete dense matrix and should therefore
    only be used when it fits in memory. Values are converted to ``float32``.

    :param data: CSR matrix to convert
    :type data: csr_matrix
    :return: Dense Torch tensor representation of the matrix.
    :rtype: torch.Tensor
    """
    return torch.FloatTensor(data.toarray())


def naive_tensor2sparse(tensor: torch.Tensor) -> csr_matrix:
    """Converts torch Tensor to sparse csr_matrix.

    The tensor is detached from autograd before conversion. It must be on the
    CPU; move CUDA tensors to the CPU before calling this function.

    :param tensor: Torch Tensor representation of the matrix to convert.
    :type tensor: torch.Tensor
    :return: CSR matrix representation of the matrix.
    :rtype: csr_matrix
    """
    return csr_matrix(tensor.detach().numpy())


def get_users(data: Matrix) -> List[int]:
    """Return the indices of users with at least one interaction.

    :param data: Interaction data whose rows represent users.
    :type data: recpack.matrix.Matrix
    :return: Unique row indices containing nonzero entries. Ordering is not
        guaranteed.
    :rtype: List[int]
    """
    return list(set(data.nonzero()[0]))


def get_batches(iterable: Iterable, batch_size=1000) -> Iterator[List]:
    """Get batches from an iterable.

    The final batch might contain less than batch_size entries, as it will be the remainder.

    :param iterable: List of values that will be split into batches of size `batch_size`
    :type iterable: Iterable
    :param batch_size: Size of each batch, defaults to 1000
    :type batch_size: int, optional
    :yield: Iterator of lists of values
    :rtype: Iterator[List]
    """
    if not isgenerator(iterable):
        iterable = iter(iterable)

    while True:

        batch = list(islice(iterable, 0, batch_size))
        if batch:
            yield batch
        else:
            break


def sample_rows(*args: Matrix, sample_size: int = 1000) -> List[Matrix]:
    """Samples rows from the matrices

    Rows are sampled from the nonzero rows in the first csr_matrix argument.
    The return value will contain a matrix for each of the matrix arguments, with only the sampled rows nonzero.

    The same rows are selected from every input. Sampling is without
    replacement and only considers nonzero rows in the first matrix. For an
    :class:`~recpack.matrix.InteractionMatrix`, interactions of unselected
    users are removed; sparse inputs retain their original shape with
    unselected rows set to zero.

    :param args: Matrices from which to select the same user rows. At least one
        matrix is required, and all matrices are expected to use the same user
        row indices.
    :type args: recpack.matrix.Matrix
    :param sample_size: Maximum number of rows to sample, defaults to 1000
    :type sample_size: int, optional
    :return: Sampled copies in the same order as the input matrices.
    :rtype: List[Matrix]
    """
    nonzero_users = list(set(args[0].nonzero()[0]))
    users = np.random.choice(nonzero_users, size=min(sample_size, len(nonzero_users)), replace=False)
    sampled_matrices = []

    for mat in args:
        if type(mat) == InteractionMatrix:
            sampled_mat = mat.users_in(users)
        else:
            sampled_mat = csr_matrix(mat.shape)
            sampled_mat[users, :] = mat[users, :]

        sampled_matrices.append(sampled_mat)

    return sampled_matrices


def union_csr_matrices(a: csr_matrix, b: csr_matrix) -> csr_matrix:
    """Combine entries of 2 binary csr_matrices.


    Inputs must have compatible shapes. Values are added and then binarized, so
    every coordinate that is nonzero in either matrix is one in the result.

    :param a: First binary CSR matrix.
    :type a: csr_matrix
    :param b: Second binary CSR matrix.
    :type b: csr_matrix
    :return: Binary union of ``a`` and ``b``.
    :rtype: csr_matrix
    """
    return to_binary(a + b)


def invert(x: Union[np.ndarray, csr_matrix]) -> Union[np.ndarray, csr_matrix]:
    """Invert the nonzero elements of an array or CSR matrix.

    Zero entries remain zero. The input is not modified.

    :param x: Dense array or sparse matrix to invert element-wise.
    :type x: numpy.ndarray or scipy.sparse.csr_matrix
    :raises TypeError: If ``x`` is neither an ndarray nor a CSR matrix.
    :return: Object of the same kind and shape as ``x`` containing reciprocal
        nonzero values.
    :rtype: numpy.ndarray or scipy.sparse.csr_matrix
    """
    if isinstance(x, np.ndarray):
        ret = np.zeros(x.shape)
    elif isinstance(x, csr_matrix):
        ret = csr_matrix(x.shape)
    else:
        raise TypeError("Unsupported type for argument x.")
    ret[x.nonzero()] = 1 / x[x.nonzero()]
    return ret
