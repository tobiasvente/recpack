# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import warnings
from typing import Optional

import numpy as np
from scipy.sparse import diags_array
from scipy.sparse import csr_array
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer

from recpack.algorithms.base import Algorithm, TopKItemSimilarityMatrixAlgorithm
from recpack.algorithms.util import invert, to_binary
from recpack.util import get_top_K_values


def compute_conditional_probability(X: csr_array, pop_discount: float = 0) -> csr_array:
    """Compute conditional probability like similarity.

    Computation using equation (3) from the original ItemKNN paper.
    'Item-based top-n recommendation algorithms.'
    Deshpande, Mukund, and George Karypis

    .. math ::
        sim(i,j) = \\frac{\\sum\\limits_{u \\in U} \\mathbb{I}_{u,i} X_{u,j}}{Freq(i) \\times Freq(j)^{\\alpha}}

    Where :math:`\\mathbb{I}_{ui}` is 1 if the user u has visited item i, and 0 otherwise.
    And alpha is the pop_discount parameter.
    Note that this is a non-symmetric similarity measure.
    Given that X is a binary matrix, and alpha is set to 0,
    this simplifies to pure conditional probability.

    .. math::
        sim(i,j) = \\frac{Freq(i \\land j)}{Freq(i)}

    :param X: user x item matrix with scores per user, item pair.
    :type X: csr_array
    :param pop_discount: Parameter defining popularity discount. Defaults to 0
    :type pop_discount: float, Optional.
    """
    # matrix with co_mat_i,j =  SUM(1_u,i * X_u,j for each user u)
    # If the input matrix is binary, this is the cooccurence count matrix.
    co_mat = to_binary(X).T @ X

    # Compute the inverse of the item frequencies
    A = invert(diags_array(to_binary(X).sum(axis=0)).tocsr())

    if pop_discount:
        # This has all item similarities
        # Co_mat is weighted by both the frequencies of item i
        # and the frequency of item j to the pop_discount power.
        # If pop_discount = 1, this similarity is symmetric again.
        item_cond_prob_similarities = A @ co_mat @ A.power(pop_discount)
    else:
        # Weight the co_mat with the amount of occurences of item i.
        item_cond_prob_similarities = A @ co_mat

    # Set diagonal to 0, because we don't support self similarity
    item_cond_prob_similarities.setdiag(0)

    return item_cond_prob_similarities


def compute_cosine_similarity(X: csr_array) -> csr_array:
    """Compute cosine similarity between the rows of a matrix.

    Self similarity is removed.

    :param X: Matrix whose rows represent the entities to compare.
    :type X: csr_array
    :return: similarity matrix
    :rtype: csr_array
    """
    cosine_similarities = cosine_similarity(X, dense_output=False)
    cosine_similarities.setdiag(0)

    return cosine_similarities


def compute_pearson_similarity(X: csr_array) -> csr_array:
    """Compute the pearson correlation as a similarity between each item in the matrix.

    Self similarity is removed.
    When computing similarity, the avg of nonzero entries per user is used.

    :param X: Rating or psuedo rating matrix.
    :type X: csr_array
    :return: similarity matrix.
    :rtype: csr_array
    """

    if (X == 1).sum() == X.nnz:
        raise ValueError("Pearson similarity can not be computed on a binary matrix.")

    count_per_item = (X > 0).sum(axis=0)

    avg_per_item = X.sum(axis=0).astype(float)

    avg_per_item[count_per_item > 0] = avg_per_item[count_per_item > 0] / count_per_item[count_per_item > 0]

    X = X - (X > 0).multiply(avg_per_item)

    # Given the rescaled matrix, the pearson correlation is just cosine similarity on this matrix.
    # X.T otherwise we are doing a user KNN
    return compute_cosine_similarity(X.T)


def compute_jaccard_similarity(X: csr_array) -> csr_array:
    """Compute the Jaccard similarity between items.

    Jaccard similarity between item i and j is computed as:

    .. math::
        sim(i,j) = \\frac{|U_i \\cap U_j|}{|U_i \\cup U_j|}

    Self similarity is removed.

    :param X: user x item matrix with scores per user, item pair.
    :type X: csr_array
    :return: item-item similarity matrix
    :rtype: csr_array
    """
    # convert score matrix to binary interaction matrix
    X = to_binary(X)

    intersection = X.T @ X
    intersection = intersection.tocoo()

    item_counts = X.sum(axis=0)

    unions = item_counts[intersection.row] + item_counts[intersection.col] - intersection.data

    item_jaccard_similarities = csr_array(
        (intersection.data / unions, (intersection.row, intersection.col)),
        shape=intersection.shape,
    )

    # remove self similarity
    item_jaccard_similarities.setdiag(0)

    # remove explicit 0 on the diagonal
    item_jaccard_similarities.eliminate_zeros()

    return item_jaccard_similarities


def compute_dice_similarity(X: csr_array) -> csr_array:
    """Compute the Sorensen-Dice similarity between items.

    Dice similarity between item i and j is computed as:

    .. math::
        sim(i,j) = \\frac{2 |U_i \\cap U_j|}{|U_i| + |U_j|}

    Self similarity is removed.

    :param X: user x item matrix with scores per user, item pair.
    :type X: csr_array
    :return: item-item similarity matrix
    :rtype: csr_array
    """
    # convert score matrix to binary interaction matrix
    X = to_binary(X)

    intersection = X.T @ X
    intersection = intersection.tocoo()

    item_counts = X.sum(axis=0)
    denominators = item_counts[intersection.row] + item_counts[intersection.col]

    item_dice_similarities = csr_array(
        (2 * intersection.data / denominators, (intersection.row, intersection.col)),
        shape=intersection.shape,
    )

    # remove self similarity
    item_dice_similarities.setdiag(0)

    # remove explicit 0 on the diagonal
    item_dice_similarities.eliminate_zeros()

    return item_dice_similarities


def compute_overlap_similarity(X: csr_array) -> csr_array:
    """Compute the overlap coefficient between items.

    Overlap similarity between item i and j is computed as:

    .. math::
        sim(i,j) = \\frac{|U_i \\cap U_j|}{min(|U_i|, |U_j|)}

    Self similarity is removed.

    :param X: user x item matrix with scores per user, item pair.
    :type X: csr_array
    :return: item-item similarity matrix
    :rtype: csr_array
    """
    # convert score matrix to binary interaction matrix
    X = to_binary(X)

    intersection = X.T @ X
    intersection = intersection.tocoo()

    item_counts = X.sum(axis=0)
    denominators = np.minimum(
        item_counts[intersection.row],
        item_counts[intersection.col],
    )

    item_overlap_similarities = csr_array((intersection.data / denominators, (intersection.row, intersection.col)),
        shape=intersection.shape,
    )

    # remove self similarity
    item_overlap_similarities.setdiag(0)

    # remove explicit 0 on the diagonal
    item_overlap_similarities.eliminate_zeros()

    return item_overlap_similarities


def compute_lift_similarity(X: csr_array) -> csr_array:
    """Compute lift between items.

    Lift between item i and j is computed as:

    .. math::
        sim(i,j) = \\frac{Support(i \\cap j)}{Support(i)Support(j)}

    Self similarity is removed.

    :param X: user x item matrix with scores per user, item pair.
    :type X: csr_array
    :return: item-item similarity matrix
    :rtype: csr_array
    """
    # convert score matrix to binary interaction matrix
    X = to_binary(X)

    intersection = X.T @ X
    intersection = intersection.tocoo()

    item_counts = X.sum(axis=0)
    denominators = item_counts[intersection.row] * item_counts[intersection.col]
    n_users = X.shape[0]

    # formula can be simplified to sim(i,j) = \\frac{Freq(i \\land j)*n_users}{Freq(i)*Freq(j)}
    item_lift_similarities = csr_array(
        (intersection.data * n_users / denominators, (intersection.row, intersection.col)),
        shape=intersection.shape,
    )

    # remove self similarity
    item_lift_similarities.setdiag(0)

    # remove explicit 0 on the diagonal
    item_lift_similarities.eliminate_zeros()

    return item_lift_similarities


def compute_pmi_similarity(X: csr_array) -> csr_array:
    """Compute pointwise mutual information (PMI) between items.

    PMI between item i and j is computed as:

    .. math::
        sim(i,j) = \\log \\frac{Support(i \\cap j)}{Support(i)Support(j)}

    This is equivalent to the natural logarithm of lift.
    Self similarity is removed.

    :param X: user x item matrix with scores per user, item pair.
    :type X: csr_array
    :return: item-item similarity matrix
    :rtype: csr_array
    """
    item_pmi_similarities = compute_lift_similarity(X)
    item_pmi_similarities.data = np.log(item_pmi_similarities.data)

    item_pmi_similarities.eliminate_zeros()

    return item_pmi_similarities


class ItemKNN(TopKItemSimilarityMatrixAlgorithm):
    """Item K Nearest Neighbours model.

    First described in 'Item-based top-n recommendation algorithms.'
    Deshpande, Mukund, and George Karypis,
    ACM Transactions on Information Systems (TOIS) 22.1 (2004): 143-177

    For each item the K most similar items are computed during fit.
    Similarity parameter decides how to compute the similarity between two items.
    Supported options are: ``"cosine"``, ``"conditional_probability"``, ``"jaccard"``, ``"dice"``, ``"overlap"``,
    ``"lift"``, and ``"pmi"``.

    Cosine similarity between item i and j is computed as

    .. math::
        sim(i,j) = \\frac{X_i X_j}{||X_i||_2 ||X_j||_2}

    The conditional probablity based similarity of item i with j is computed as

    .. math ::
        sim(i,j) = \\frac{\\sum\\limits_{u \\in U} \\mathbb{I}_{u,i} X_{u,j}}{Freq(i) \\times Freq(j)^{\\alpha}}

    Where I_ui is 1 if the user u has visited item i, and 0 otherwise.
    And alpha is the pop_discount parameter.
    Note that this is a non-symmetric similarity measure.
    Given that X is a binary matrix, and alpha is set to 0, this simplifies to pure conditional probability.

    .. math::
        sim(i,j) = \\frac{Freq(i \\land j)}{Freq(i)}

    For the following set-based metrics, U_i is the set of users that have interacted with item i.

    Jaccard similarity between item i and j is computed as:

    .. math::
        sim(i,j) = \\frac{|U_i \\cap U_j|}{|U_i \\cup U_j|}

    Dice similarity between item i and j is computed as:

    .. math::
        sim(i,j) = \\frac{2 |U_i \\cap U_j|}{|U_i| + |U_j|}

    Overlap similarity between item i and j is computed as:

    .. math::
        sim(i,j) = \\frac{|U_i \\cap U_j|}{min(|U_i|, |U_j|)}

    Lift similarity between item i and j is computed as:

    .. math::
        sim(i,j) = \\frac{Support(i \\cap j)}{Support(i)Support(j)}

    PMI similarity between item i and j is computed as:

    .. math::
        sim(i,j) = \\log \\frac{Support(i \\cap j)}{Support(i)Support(j)}

    If sim_normalize is True, the scores are normalized per predictive item,
    making sure the sum of each row in the similarity matrix is 1.

    :param K: How many neigbours to use per item,
        make sure to pick a value below the number of columns of the matrix to fit on.
        Defaults to 200
    :type K: int, optional
    :param similarity: Which similarity measure to use,
        can be one of ["cosine", "conditional_probability", "jaccard", "dice", "overlap", "lift", "pmi"],
        defaults to "cosine"
    :type similarity: str, optional
    :param pop_discount: Power applied to the comparing item in the denominator,
        to discount contributions of very popular items.
        Should be between 0 and 1. If None, apply no discounting.
        Defaults to None.
    :type pop_discount: float, optional
    :param normalize_X: Normalize rows in the interaction matrix so that
        the contribution of users who have viewed more items is smaller,
        defaults to False
    :type normalize_X: bool, optional
    :param normalize_sim: Normalize scores per row in the similarity matrix to
        counteract artificially large similarity scores when the predictive item is
        rare, defaults to False.
    :type normalize_sim: bool, optional
    :raises ValueError: If an unsupported similarity measure is passed.
    """

    SUPPORTED_SIMILARITIES = ["cosine", "conditional_probability", "jaccard", "dice", "overlap", "lift", "pmi"]
    """The supported similarity options"""

    def __init__(
        self,
        K=200,
        similarity: str = "cosine",
        pop_discount: Optional[float] = None,
        normalize_X: bool = False,
        normalize_sim: bool = False,
    ):
        super().__init__(K)

        if similarity not in self.SUPPORTED_SIMILARITIES:
            raise ValueError(f"similarity {similarity} not supported")
        self.similarity = similarity

        if self.similarity != "conditional_probability" and pop_discount:
            warnings.warn(
                "Argument pop_discount is incompatible with all similarity \
                functions except conditional probability. \
                This argument will be ignored, \
                popularity discounting won't be applied.",
                UserWarning,
            )

        if type(pop_discount) == float and (pop_discount < 0 or pop_discount > 1):
            raise ValueError("Invalid value for pop_discount. Value should be between 0 and 1.")

        self.pop_discount = pop_discount

        self.normalize_X = normalize_X
        # Sim_normalize takes precedence.
        self.normalize_sim = normalize_sim

    def _fit(self, X: csr_array) -> None:
        """Fit a cosine similarity matrix from item to item"""

        transformer = Normalizer(norm="l1", copy=False)

        if self.normalize_X:
            X = transformer.transform(X)

        if self.similarity == "cosine":
            # X.T otherwise we are doing a user KNN
            item_similarities = compute_cosine_similarity(X.T)
        elif self.similarity == "conditional_probability":
            item_similarities = compute_conditional_probability(X, self.pop_discount)
        elif self.similarity == "jaccard":
            item_similarities = compute_jaccard_similarity(X)
        elif self.similarity == "dice":
            item_similarities = compute_dice_similarity(X)
        elif self.similarity == "overlap":
            item_similarities = compute_overlap_similarity(X)
        elif self.similarity == "lift":
            item_similarities = compute_lift_similarity(X)
        elif self.similarity == "pmi":
            item_similarities = compute_pmi_similarity(X)

        item_similarities = get_top_K_values(item_similarities, K=self.K)

        # j, M (*, j) = 1
        if self.normalize_sim:
            # Normalize such that sum per row = 1
            item_similarities = transformer.transform(item_similarities)

        self.similarity_matrix_ = item_similarities


class ItemPNN(ItemKNN):
    """Item Probabilistic Nearest Neighbours model.

    First described in Panagiotis Adamopoulos and Alexander Tuzhilin. 2014.
    'On over-specialization and concentration bias of recommendations:
    probabilistic neighborhood selection in collaborative filtering systems'.
    In Proceedings of the 8th ACM Conference on Recommender systems (RecSys '14).
    Association for Computing Machinery, New York, NY, USA, 153–160.
    DOI:https://doi.org/10.1145/2645710.2645752

    For each item K neighbours are selected either uniformly or based on the empirical
    distribution of the items (or a softmax thereof).
    Similarity parameter decides how to compute the similarity between two items.
    Supported options are: ``"cosine"``, ``"conditional_probability"``, ``"jaccard"``, ``"dice"``, ``"overlap"``,
    ``"lift"``, and ``"pmi"``.

    - Cosine similarity between item i and j is computed as
      the ``count(i and j) / (count(i)*count(j))``.
    - Conditional probablity of item i with j is computed
      as ``count(i and j) / (count(i))``.
      Note that this is a non-symmetric similarity measure.
    - Jaccard similarity between item i and j is computed as
      ``count(i and j) / count(i or j)``.
    - Dice similarity between item i and j is computed as
      ``2 * count(i and j) / (count(i) + count(j))``.
    - Overlap similarity between item i and j is computed as
      ``count(i and j) / min(count(i), count(j))``.
    - Lift similarity between item i and j is computed as
      ``Support(i and j) / (Support(i) * Support(j))``.
    - PMI similarity between item i and j is computed as
      ``log(Support(i and j) / (Support(i) * Support(j)))``.

    If sim_normalize is True, the scores are normalized per predictive item,
    making sure the sum of each row in the similarity matrix is 1.

    :param K: How many neigbours to use per item,
        make sure to pick a value below the number of columns of the matrix to fit on.
        Defaults to 200
    :type K: int, optional
    :param similarity: Which similarity measure to use,
        can be one of ["cosine", "conditional_probability", "jaccard", "dice", "overlap", "lift", "pmi"],
        defaults to "cosine"
    :type similarity: str, optional
    :param pop_discount: Power applied to the comparing item in the denominator,
        to discount contributions of very popular items.
        Should be between 0 and 1. If None, apply no discounting.
        Defaults to None.
    :type pop_discount: float, optional
    :param normalize_X: Normalize rows in the interaction matrix so that
        the contribution of users who have viewed more items is smaller,
        defaults to False
    :type normalize_X: bool, optional
    :param normalize_sim: Normalize scores per row in the similarity matrix to
        counteract artificially large similarity scores
        when the predictive item is rare,
        defaults to False.
    :type normalize_sim: bool, optional
    :param pdf: Which probability distribution to use,
        can be one of ["empirical", "uniform", "softmax_empirical"],
        defaults to "empirical"
    :type pdf: str, optional
    :param seed: Seed to the randomizers, useful for reproducible results,
        defaults to None
    :type seed: int, optional
    :raises ValueError: If an unsupported similarity measure or
        probability distribution is passed.
    """

    SUPPORTED_SAMPLING_FUNCTIONS = ["empirical", "uniform", "softmax_empirical"]
    """The supported similarity options"""

    def __init__(
        self,
        K=200,
        similarity: str = "cosine",
        pop_discount: Optional[float] = None,
        normalize_X: bool = False,
        normalize_sim: bool = False,
        pdf: str = "empirical",
        seed: Optional[int] = None,
    ):
        super().__init__(
            K=K,
            similarity=similarity,
            pop_discount=pop_discount,
            normalize_X=normalize_X,
            normalize_sim=normalize_sim,
        )

        if pdf not in self.SUPPORTED_SAMPLING_FUNCTIONS:
            raise ValueError(f"Sampling function {pdf} not supported")

        self.pdf = pdf

        if seed is None:
            seed = np.random.get_state()[1][0]

        np.random.seed(seed)
        self.seed = seed

    def _compute_pdf(self, pdf: str, sim_matrix: csr_array) -> np.ndarray:
        # TODO Outside of the class maybe?
        sim_matrix = sim_matrix.toarray()
        if pdf == "empirical":
            # Add the None dimension at the end to do a row-wise division.
            # Otherwise the default is column-wise.
            p = sim_matrix / sim_matrix.sum(axis=1)[:, None]
        elif pdf == "uniform":
            p = np.ones(sim_matrix.shape) / sim_matrix.shape[1]
        elif pdf == "softmax_empirical":
            softmax_item_sims = np.exp(sim_matrix)
            p = softmax_item_sims / softmax_item_sims.sum(axis=1)[:, None]
        else:
            raise ValueError(f"Sampling function {pdf} not supported")

        return p

    def _fit(self, X: csr_array) -> None:
        """Fit a cosine similarity matrix from item to item"""

        transformer = Normalizer(norm="l1", copy=False)

        if self.normalize_X:
            X = transformer.transform(X)

        if self.similarity == "cosine":
            # X.T otherwise we are doing a user KNN
            item_similarities = compute_cosine_similarity(X.T)
        elif self.similarity == "conditional_probability":
            item_similarities = compute_conditional_probability(X, self.pop_discount)
        elif self.similarity == "jaccard":
            item_similarities = compute_jaccard_similarity(X)
        elif self.similarity == "dice":
            item_similarities = compute_dice_similarity(X)
        elif self.similarity == "overlap":
            item_similarities = compute_overlap_similarity(X)
        elif self.similarity == "lift":
            item_similarities = compute_lift_similarity(X)
        elif self.similarity == "pmi":
            item_similarities = compute_pmi_similarity(X)

        self.pdf_ = self._compute_pdf(self.pdf, item_similarities)

        item_similarities = get_K_values(item_similarities, self.K, self.pdf_)

        # j, M (*, j) = 1
        if self.normalize_sim:
            # Normalize such that sum per row = 1
            item_similarities = transformer.transform(item_similarities)

        self.similarity_matrix_ = item_similarities

    # def _predict(self, X: csr_array) -> csr_array:
    #     pass


def get_K_values(X: csr_array, K: int, pdf: np.ndarray) -> csr_array:
    """Select K values random values for every row in X,
    sampled according to the probabilities in pdf.
    All other values in the row are set to zero.

    :param X: Matrix from which we will select K values in every row.
    :type X: csr_array
    :param K: Amount of values to select.
    :type K: int
    :param pdf: np.ndarray of probabilities of items in X, given another item.
        Rows should sum to 1.
    :type pdf: np.ndarray
    :return: Matrix with K values per row.
    :rtype: csr_array
    """
    items = np.arange(0, X.shape[1], dtype=int)

    U, I, V = [], [], []

    for row_ix in range(0, X.shape[0]):
        # Select one more, so that we can eliminate the item itself.
        selected_K = np.random.choice(items, size=K + 1, p=pdf[row_ix, :], replace=False)

        try:
            # Eliminate the item itself if it was selected.
            mismatch = np.where(selected_K == row_ix)[0][0]
        except IndexError:
            # If it was not selected, just eliminate the last item.
            mismatch = -1

        selected_K = np.delete(selected_K, mismatch)

        U.extend([row_ix] * K)
        I.extend(selected_K)
        V.extend([1] * K)

    data_K = csr_array((V, (U, I)), shape=X.shape)
    return data_K.multiply(X)


class UserKNN(Algorithm):
    """User K Nearest Neighbours model.

    For each user, the K most similar users are computed during fit.
    The similarity parameter determines how similarity between two users is
    computed. Supported options are ``"cosine"``, ``"conditional_probability"``,
    ``"jaccard"``, ``"dice"``, ``"overlap"``, ``"lift"``, and ``"pmi"``.

    Cosine similarity between users u and v is computed as

    .. math::
        sim(u,v) = \\frac{X_u X_v}{||X_u||_2 ||X_v||_2}

    The conditional-probability similarity of user u with user v is computed as

    .. math::
        sim(u,v) = \\frac{\\sum\\limits_{i \\in I} \\mathbb{I}_{u,i} X_{v,i}}{Freq(u)}

    Where :math:`\\mathbb{I}_{u,i}` is 1 if user u has interacted with item i,
    and 0 otherwise. This is a non-symmetric similarity measure.

    Recommendation scores are computed by multiplying the fitted user
    similarity matrix with the user-item interaction matrix supplied during
    prediction.

    :param K: How many neighbours to use per user. This should be smaller than
        the number of rows in the matrix used for fitting.
    :type K: int
    :param similarity: Which similarity measure to use. Can be one of
        ``["cosine", "conditional_probability", "jaccard", "dice", "overlap",
        "lift", "pmi"]``. Defaults to ``"cosine"``.
    :type similarity: str, optional
    :raises ValueError: If an unsupported similarity measure is passed.
    """

    SUPPORTED_SIMILARITIES = ["cosine", "conditional_probability", "jaccard", "dice", "overlap", "lift", "pmi",]

    def __init__(self, K, similarity: str = "cosine"):
        super().__init__()
        self.K = K
        if similarity not in self.SUPPORTED_SIMILARITIES:
            raise ValueError(f"similarity {similarity} not supported")
        self.similarity = similarity

    def _fit(self, X: csr_array) -> None:
        """Fit a similarity matrix from user to user.

        :param X: user x item matrix with scores per user, item pair.
        """
        if self.similarity == "cosine":
            user_similarities = compute_cosine_similarity(X)
        elif self.similarity == "conditional_probability":
            user_similarities = compute_conditional_probability(X.T)
        elif self.similarity == "jaccard":
            user_similarities = compute_jaccard_similarity(X.T)
        elif self.similarity == "dice":
            user_similarities = compute_dice_similarity(X.T)
        elif self.similarity == "overlap":
            user_similarities = compute_overlap_similarity(X.T)
        elif self.similarity == "lift":
            user_similarities = compute_lift_similarity(X.T)
        elif self.similarity == "pmi":
            user_similarities = compute_pmi_similarity(X.T)

        user_similarities = get_top_K_values(user_similarities, K=self.K)

        self.similarity_matrix_ = user_similarities

    def _predict(self, X: csr_array) -> csr_array:
        """Predict scores for nonzero users in X

        Scores are computed by matrix multiplication of the stored similarity matrix with X.

        :param X: user x item matrix with scores per user, item pair.
        :type X: csr_array
        :return: csr_array with scores
        :rtype: csr_array
        """
        scores = self.similarity_matrix_ @ X

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
