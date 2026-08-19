# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from scipy.sparse import csr_array


def sparse_inverse_nonzero(a: csr_array) -> csr_array:
    """Invert nonzero elements of a `scipy.sparse.csr_array`.

    Zero entries remain zero, and the input is not modified. Integer matrices
    may need to be converted to floating point before use because reciprocal
    values are assigned using the input dtype.

    :param a: Matrix to invert.
    :type a: csr_array
    :return: Copy of ``a`` with stored values replaced by their reciprocals.
    :rtype: csr_array
    """
    inv_a = a.copy()
    inv_a.data = 1 / inv_a.data
    return inv_a


def sparse_divide_nonzero(a: csr_array, b: csr_array) -> csr_array:
    """Elementwise divide of nonzero elements of a by nonzero elements of b.

    Elements that are zero in either ``a`` or ``b`` are zero in the resulting
    matrix. The matrices must have compatible shapes.

    :param a: Numerator.
    :type a: csr_array
    :param b: Denominator.
    :type b: csr_array
    :return: Result of the elementwise division of matrix a by matrix b.
    :rtype: csr_array
    """
    return a.multiply(sparse_inverse_nonzero(b))
