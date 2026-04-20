# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Marouan El Marnissi


from recpack.algorithms import Algorithm
from scipy.sparse import csr_matrix

from sklearn.metrics.pairwise import cosine_similarity

from recpack.util import get_top_K_values
import warnings


def compute_cosine_similarity(X: csr_matrix) -> csr_matrix:
    """Compute the cosine similarity between the users in the matrix.

    Self similarity is removed.

    :param X: user x item matrix with scores per user, item pair.
    :type X: csr_matrix
    :return: similarity matrix
    :rtype: csr_matrix
    """
    user_cosine_similarities = cosine_similarity(X, dense_output=False)

    # Set diagonal to 0, because we don't want to support self similarity
    user_cosine_similarities.setdiag(0)

    return user_cosine_similarities


class UserKNN(Algorithm):

    def __init__(self, K):
        super().__init__()
        self.K = K

    def _fit(self, X: csr_matrix) -> None:
        """
        Fit a cosine similarity matrix from user to user
        :param X: user x item matrix with scores per user, item pair.
        """

        user_similarities = compute_cosine_similarity(X)

        user_similarities = get_top_K_values(user_similarities, K=self.K)

        self.similarity_matrix_ = user_similarities

    def _predict(self, X: csr_matrix) -> csr_matrix:
        """Predict scores for nonzero users in X

        Scores are computed by matrix multiplication of the stored similarity matrix with X.

        :param X: user x item matrix with scores per user, item pair.
        :type X: csr_matrix
        :return: csr_matrix with scores
        :rtype: csr_matrix
        """
        scores =  self.similarity_matrix_ @ X

        return scores

    def _check_fit_complete(self):
        """Helper function to check if model was correctly fitted

        Checks implemented:

        - Checks if the algorithm has been fitted, using sklearn's `check_is_fitted`
        - Checks if the fitted similarity matrix contains similar users for each user

        For failing checks a warning is printed.
        """
        # Use super to check is fitted
        super()._check_fit_complete()

        # Additional checks on the fitted matrix.
        # Check if actually exists!
        assert hasattr(self, "similarity_matrix_")

        # Check row wise, since that will determine the recommendation options.
        users_with_score = set(self.similarity_matrix_.nonzero()[0])

        missing = self.similarity_matrix_.shape[0] - len(users_with_score)
        if missing > 0:
            warnings.warn(f"{self.name} missing similar users for {missing} users.")
