# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging
from typing import List, Optional, Tuple

from scipy.sparse import csr_matrix, lil_matrix

from tqdm.auto import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from recpack.algorithms.base import TorchMLAlgorithm
from recpack.algorithms.loss_functions import bpr_loss
from recpack.algorithms.samplers import BootstrapSampler
from recpack.util import get_top_K_values

logger = logging.getLogger("recpack")

class BPRBase(TorchMLAlgorithm):
    """Base class for recommendation algorithms optimized with BPR-OPT.

    Implements the shared LearnBPR optimization procedure defined in Rendle,
    Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback."
    For every sampled interaction, the sampler selects a target item that the
    user interacted with and an MNAR item that the user did not interact with.
    The parameters are optimized so that the target item receives a higher
    score than the MNAR item.

    Subclasses provide the model-specific score calculation, regularization and
    prediction behavior. This class provides bootstrap triple sampling, BPR loss
    calculation, gradient updates and optimizer initialization.

    :param batch_size: Size of the batches to use during gradient descent.
    :type batch_size: int
    :param max_epochs: The maximum amount of epochs to train the model.
    :type max_epochs: int
    :param learning_rate: The learning rate of the optimization procedure.
    :type learning_rate: float
    :param stopping_criterion: Which criterion to use to optimize the parameters.
        Available criteria can be found at
        :attr:`recpack.algorithms.stopping_criterion.StoppingCriterion.FUNCTIONS`.
    :type stopping_criterion: str
    :param stop_early: If True, stop training when the improvement remains below
        ``min_improvement`` for ``max_iter_no_change`` evaluations.
        Defaults to False.
    :type stop_early: bool, optional
    :param max_iter_no_change: Amount of evaluations without sufficient
        improvement allowed when early stopping is enabled. Defaults to 5.
    :type max_iter_no_change: int, optional
    :param min_improvement: Minimum improvement required for an iteration to be
        considered an improvement. Defaults to 0.01.
    :type min_improvement: float, optional
    :param seed: Seed used to make random sampling reproducible. Defaults to None.
    :type seed: int, optional
    :param save_best_to_file: If True, save the best model to disk after fitting.
        Defaults to False.
    :type save_best_to_file: bool, optional
    :param sample_size: Number of triples sampled during each epoch. If None, one
        triple is sampled for every interaction. Sampling happens with replacement.
        Defaults to None.
    :type sample_size: int, optional
    :param keep_last: Retain the final model instead of the model with the best
        validation score. Defaults to False.
    :type keep_last: bool, optional
    :param predict_topK: Number of highest-scoring recommendations to retain per
        user. If None, all scores are retained. Defaults to None.
    :type predict_topK: int, optional
    :param validation_sample_size: Number of users sampled when calculating the
        validation loss and stopping criterion. If None, all nonzero users are
        used. Defaults to None.
    :type validation_sample_size: int, optional
    """

    def __init__(
        self,
        batch_size: int,
        max_epochs: int,
        learning_rate: float,
        stopping_criterion: str,
        stop_early: bool = False,
        max_iter_no_change: int = 5,
        min_improvement: float = 0.01,
        seed: int = None,
        save_best_to_file: bool = False,
        sample_size=None,
        keep_last: bool = False,
        predict_topK: int = None,
        validation_sample_size: int = None,
    ):
        super().__init__(
            batch_size,
            max_epochs,
            learning_rate,
            stopping_criterion,
            stop_early=stop_early,
            max_iter_no_change=max_iter_no_change,
            min_improvement=min_improvement,
            seed=seed,
            save_best_to_file=save_best_to_file,
            keep_last=keep_last,
            predict_topK=predict_topK,
            validation_sample_size=validation_sample_size,
        )
        self.sample_size = sample_size
        self.sampler = BootstrapSampler(
            num_negatives=1,
            batch_size=self.batch_size,
        )

    def _init_model(self, X: csr_matrix) -> None:
        self.model_ = self._create_model(X).to(self.device)
        self.optimizer = optim.Adagrad(self.model_.parameters(), lr=self.learning_rate)

    def _create_model(self, X: csr_matrix) -> nn.Module:
        raise NotImplementedError()

    def _get_batch_scores(
        self,
        train_data: csr_matrix,
        users: torch.Tensor,
        target_items: torch.Tensor,
        mnar_items: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, object]:
        raise NotImplementedError()

    def _regularization_loss(
        self,
        target_items: torch.Tensor,
        mnar_items: torch.Tensor,
        context,
    ) -> torch.Tensor:
        raise NotImplementedError()

    def _train_epoch(self, train_data: csr_matrix):
        """train a single epoch. Uses sampler to generate samples,
        and loop through them in batches of self.batch_size.
        After each batch, update the parameters according to gradients.

        :param train_data: interaction matrix.
        :type train_data: csr_matrix
        """
        losses = []

        # For each positive item sample a single negative item.
        for users, target_items, mnar_items in tqdm(
            self.sampler.sample(
                train_data,
                sample_size=self.sample_size,
            ),
            desc=f"train_epoch {self.name}",
        ):
            users = users.to(self.device)
            # Target items are items the user has interacted with,
            # and we expect to recommend high
            target_items = target_items.to(self.device)
            # Items the user has not seen, and assuming MNAR data
            # Is a batch_size x 1 matrix, squeeze into an array.
            mnar_items = mnar_items.squeeze(-1).to(self.device)

            self.optimizer.zero_grad()

            target_sim, mnar_sim, context = self._get_batch_scores(
                train_data,
                users,
                target_items,
                mnar_items,
            )

            # Checks to make sure the shapes are correct.
            if not ((mnar_sim.shape == target_sim.shape) and (target_sim.shape[0] == users.shape[0])):
                raise AssertionError("Shapes should match")

            loss = bpr_loss(target_sim, mnar_sim)
            loss += self._regularization_loss(target_items, mnar_items, context)
            loss.backward()
            losses.append(loss.item())
            self.optimizer.step()

        return losses


class BPRMF(BPRBase):
    """Implements Matrix Factorization by using the BPR-OPT objective
    and SGD optimization.

    MF implementation using the BPR criterion as defined in Rendle, Steffen, et al.
    "BPR: Bayesian personalized ranking from implicit feedback."

    The BPR optimization criterion aims to construct a factorization that optimally
    ranks interesting items (interacted with previously)
    above uninteresting or unknown items for all users.

    :param num_components: The size of the latent vectors for both users and items.
                            defaults to 100
    :type num_components: int, optional
    :param lambda_h: The regularization parameter for the item embedding,
        should be a value between 0 and 1.
        Defaults to 0.0
    :type lambda_h: float, optional
    :param lambda_w: The regularization parameter for the user embedding,
        defaults to 0.0
    :type lambda_w: float, optional
    :param batch_size: Size of the batches to use during gradient descent. Defaults to 1000.
    :type batch_size: int, optional
    :param max_epochs: The max amount of epochs to train the model, defaults to 20
    :type max_epochs: int, optional
    :param learning_rate: The learning rate of the optimization procedure,
                            defaults to 0.01
    :type learning_rate: float, optional
    :param seed: Seed to fix random numbers, to make results reproducible,
                    defaults to None
    :type seed: int, optional
    :param stopping_criterion: Which criterion to use optimise the parameters,
        a string which indicates the name of the stopping criterion.
        Which criterions are available can be found at
        :attr:`recpack.algorithms.stopping_criterion.StoppingCriterion.FUNCTIONS`.
        Defaults to 'recall'
    :type stopping_criterion: str, optional
    :param stop_early: If True, early stopping is enabled,
        and after ``max_iter_no_change`` iterations where improvement of loss function
        is below ``min_improvement`` the optimisation is stopped,
        even if max_epochs is not reached.
        Defaults to False
    :type stop_early: bool, optional
    :param max_iter_no_change: If early stopping is enabled,
        stop after this amount of iterations without change.
        Defaults to 5
    :type max_iter_no_change: int, optional
    :param min_improvement: If early stopping is enabled, no change is detected,
        if the improvement is below this value.
        Defaults to 0.01
    :type min_improvement: float, optional
    :param save_best_to_file: If True, the best model is saved to disk after fit.
    :type save_best_to_file: bool, optional
    :param sample_size: How many samples to take during training using bootstrap sampling.
        If None a sample is taken for each interaction,
        there is no guarantee that all interactions will be used though,
        since sampling happens with replacement.
        Defaults to None
    :type sample_size: int, optional
    :param keep_last: Retain last model, rather than best
        (according to stopping criterion value on validation data), defaults to False
    :type keep_last: bool, optional
    :param predict_topK: The topK recommendations to keep per row in the matrix.
        Use when the user x item output matrix would become too large for RAM.
        Defaults to None, which results in no filtering.
    :type predict_topK: int, optional
    :param validation_sample_size: Amount of users that will be sampled to calculate
        validation loss and stopping criterion value.
        This reduces computation time during validation, such that training times are strongly reduced.
        If None, all nonzero users are used. Defaults to None.
    :type validation_sample_size: int, optional
    """

    def __init__(
        self,
        num_components: int = 100,
        lambda_h: float = 0.0,
        lambda_w: float = 0.0,
        batch_size: int = 1_000,
        max_epochs: int = 20,
        learning_rate: float = 0.01,
        stopping_criterion: str = "bpr",
        stop_early: bool = False,
        max_iter_no_change: int = 5,
        min_improvement: float = 0.01,
        seed: int = None,
        save_best_to_file: bool = False,
        sample_size=None,
        keep_last: bool = False,
        predict_topK: int = None,
        validation_sample_size: int = None,
    ):
        super().__init__(
            batch_size,
            max_epochs,
            learning_rate,
            stopping_criterion,
            stop_early=stop_early,
            max_iter_no_change=max_iter_no_change,
            min_improvement=min_improvement,
            seed=seed,
            save_best_to_file=save_best_to_file,
            sample_size=sample_size,
            keep_last=keep_last,
            predict_topK=predict_topK,
            validation_sample_size=validation_sample_size,
        )
        self.num_components = num_components
        self.lambda_h = lambda_h
        self.lambda_w = lambda_w

    def _create_model(self, X: csr_matrix) -> nn.Module:
        return MFModule(X.shape[0], X.shape[1], num_components=self.num_components)

    def _get_batch_scores(self, train_data, users, target_items, mnar_items):
        target_sim = self.model_(users, target_items).diag()
        mnar_sim = self.model_(users, mnar_items).diag()
        return target_sim, mnar_sim, None

    def _regularization_loss(self, target_items, mnar_items, context):
        return (
            self.lambda_h * self.model_.item_embedding_.weight.norm()
            + self.lambda_w * self.model_.user_embedding_.weight.norm()
        )

    def _batch_predict(self, X: csr_matrix, users: List[int]) -> csr_matrix:
        """Predict scores for matrix X, given the selected users in this batch

        :param X: Matrix of user item interactions,
            expected to only contain interactions for those users that are in `users`
        :type X: csr_matrix
        :param users: users selected for recommendation
        :type users: List[int]
        :return: Sparse matrix of scores per user item pair.
        :rtype: csr_matrix
        """

        user_tensor = torch.LongTensor(users).to(self.device)
        item_tensor = torch.arange(X.shape[1]).to(self.device)

        result = lil_matrix(X.shape)
        result[users] = self.model_(user_tensor, item_tensor).detach().cpu().numpy()

        return result.tocsr()


class BPRKNN(BPRBase):
    """Implements adaptive item-based nearest neighbours by using the BPR-OPT
    objective and SGD optimization.

    KNN implementation using the BPR criterion as defined in Rendle, Steffen, et al.
    "BPR: Bayesian personalized ranking from implicit feedback."

    The BPR optimization criterion aims to construct an item similarity measure
    that optimally ranks interesting items (interacted with previously) above
    uninteresting or unknown items for all users.

    This algorithm learns an item-to-item similarity measure from implicit
    feedback rather than calculating it from a fixed measure such as cosine
    similarity. A candidate item is scored by summing its learned similarities
    to the items in the user's interaction history.

    ``similarity_mode="direct"`` learns the matrix :math:`C` used in the
    paper's experiments. This requires :math:`O(|I|^2)` model memory, but allows
    every item pair to have an independent learned similarity.

    ``similarity_mode="factorized"`` learns an item embedding :math:`H` such
    that :math:`C = HH^T`. This requires :math:`O(|I|k)` model memory and is
    intended for item sets where storing the full similarity matrix is too
    expensive. The factorized mode does not construct :math:`C` during fitting
    or prediction.

    :param K: Number of highest learned similarities to retain per item after
        fitting in direct mode. If None, all learned similarities are retained.
        This parameter does not affect factorized mode. Defaults to 200.
    :type K: int, optional
    :param similarity_mode: Representation used for item similarities. Must be
        either ``"direct"`` or ``"factorized"``. Defaults to ``"direct"``.
    :type similarity_mode: str, optional
    :param lambda_target: Regularization applied to similarities or factors
        involved in target-item scores. Defaults to 0.0025.
    :type lambda_target: float, optional
    :param lambda_mnar: Regularization applied to similarities or factors
        involved in MNAR-item scores. Defaults to 0.00025.
    :type lambda_mnar: float, optional
    :param num_components: Size of each item embedding in factorized mode.
        This parameter does not affect direct mode. Defaults to 100.
    :type num_components: int, optional
    :param batch_size: Size of the batches to use during gradient descent.
        Defaults to 1000.
    :type batch_size: int, optional
    :param max_epochs: Maximum amount of epochs to train the model. Defaults to 20.
    :type max_epochs: int, optional
    :param learning_rate: Learning rate of the Adagrad optimization procedure.
        Defaults to 0.01.
    :type learning_rate: float, optional
    :param stopping_criterion: Criterion used to select the best parameters.
        Available criteria can be found at
        :attr:`recpack.algorithms.stopping_criterion.StoppingCriterion.FUNCTIONS`.
        Defaults to ``"bpr"``.
    :type stopping_criterion: str, optional
    :param stop_early: If True, enable early stopping. Defaults to False.
    :type stop_early: bool, optional
    :param max_iter_no_change: Stop after this number of evaluations without
        sufficient improvement when early stopping is enabled. Defaults to 5.
    :type max_iter_no_change: int, optional
    :param min_improvement: Minimum improvement required to reset the early
        stopping counter. Defaults to 0.01.
    :type min_improvement: float, optional
    :param seed: Seed used to make random sampling reproducible. Defaults to None.
    :type seed: int, optional
    :param save_best_to_file: If True, save the best model to disk after fitting.
        Defaults to False.
    :type save_best_to_file: bool, optional
    :param sample_size: Number of triples sampled during each epoch. If None, one
        triple is sampled for every interaction. Defaults to None.
    :type sample_size: int, optional
    :param keep_last: Retain the final model instead of the model with the best
        validation score. Defaults to False.
    :type keep_last: bool, optional
    :param predict_topK: Number of highest-scoring recommendations to retain per
        user. If None, all scores are retained. Defaults to None.
    :type predict_topK: int, optional
    :param validation_sample_size: Number of users sampled for validation. If
        None, all nonzero users are used. Defaults to None.
    :type validation_sample_size: int, optional
    """

    def __init__(
        self,
        K: Optional[int] = 200,
        similarity_mode: str = "direct",
        lambda_target: float = 0.0025,
        lambda_mnar: float = 0.00025,
        num_components: int = 100,
        batch_size: int = 1_000,
        max_epochs: int = 20,
        learning_rate: float = 0.01,
        stopping_criterion: str = "bpr",
        stop_early: bool = False,
        max_iter_no_change: int = 5,
        min_improvement: float = 0.01,
        seed: Optional[int] = None,
        save_best_to_file: bool = False,
        sample_size: Optional[int] = None,
        keep_last: bool = False,
        predict_topK: Optional[int] = None,
        validation_sample_size: Optional[int] = None,
    ):
        super().__init__(
            batch_size,
            max_epochs,
            learning_rate,
            stopping_criterion,
            stop_early=stop_early,
            max_iter_no_change=max_iter_no_change,
            min_improvement=min_improvement,
            seed=seed,
            save_best_to_file=save_best_to_file,
            sample_size=sample_size,
            keep_last=keep_last,
            predict_topK=predict_topK,
            validation_sample_size=validation_sample_size,
        )

        if K is not None and K <= 0:
            raise ValueError("K must be positive or None")
        if similarity_mode not in ("direct", "factorized"):
            raise ValueError("similarity_mode must be either 'direct' or 'factorized'")
        if num_components <= 0:
            raise ValueError("num_components must be positive")
        if lambda_target < 0 or lambda_mnar < 0:
            raise ValueError("regularization parameters must be non-negative")
        if sample_size is not None and sample_size < 0:
            raise ValueError("sample_size must be non-negative or None")

        self.K = K
        self.similarity_mode = similarity_mode
        self.num_components = num_components
        self.lambda_target = lambda_target
        self.lambda_mnar = lambda_mnar

    def _create_model(self, X: csr_matrix) -> nn.Module:
        if self.similarity_mode == "direct":
            return BPRKNNModule(X.shape[1])
        return FactorizedBPRKNNModule(X.shape[1], self.num_components)

    def _get_batch_scores(self, train_data, users, target_items, mnar_items):
        history = torch.FloatTensor(train_data[users.cpu().numpy()].toarray()).to(self.device)
        target_sim = self.model_.score_history(history, target_items).diag()
        mnar_sim = self.model_.score_history(history, mnar_items).diag()
        return target_sim, mnar_sim, history

    def _regularization_loss(self, target_items, mnar_items, history):
        return self.model_.regularization_loss(
            target_items,
            mnar_items,
            history,
            self.lambda_target,
            self.lambda_mnar,
        )

    def _batch_predict(self, X: csr_matrix, users: List[int]) -> csr_matrix:
        history = torch.FloatTensor(X[users].toarray()).to(self.device)
        item_tensor = torch.arange(X.shape[1]).to(self.device)
        result = lil_matrix(X.shape)
        result[users] = self.model_.score_history(history, item_tensor).detach().cpu().numpy()
        return result.tocsr()

    def fit(self, X, validation_data):
        super().fit(X, validation_data)
        if self.similarity_mode == "direct":
            similarities = self.model_.similarity_matrix().detach().cpu().numpy()
            similarity_matrix = csr_matrix(similarities)
            if self.K is not None and self.K < similarity_matrix.shape[1]:
                similarity_matrix = get_top_K_values(similarity_matrix, K=self.K)
                self.model_.set_similarity_matrix(similarity_matrix.toarray())
            similarity_matrix.eliminate_zeros()
            self.similarity_matrix_ = similarity_matrix
        else:
            self.item_embedding_ = self.model_.item_embedding_.weight.detach().cpu().numpy()
        return self


class MFModule(nn.Module):
    """MF torch module, encodes the embeddings and the forward functionality.

    :param num_users: the amount of users
    :type num_users: int
    :param num_items: the amount of items
    :type num_items: int
    :param num_components: The size of the embedding per user and item, defaults to 100
    :type num_components: int, optional
    """

    def __init__(self, num_users: int, num_items: int, num_components: int = 100):
        super().__init__()

        self.num_components = num_components
        self.num_users = num_users
        self.num_items = num_items

        self.user_embedding_ = nn.Embedding(num_users, num_components)  # User embedding
        self.item_embedding_ = nn.Embedding(num_items, num_components)  # Item embedding

        # Keep variance low enough, to alow learning
        self.std = min(1 / num_components ** 0.5, 0.05)
        # Initialise embeddings to a random start
        nn.init.normal_(self.user_embedding_.weight, std=self.std)
        nn.init.normal_(self.item_embedding_.weight, std=self.std)

    def forward(self, user_tensor: torch.Tensor, item_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute dot-product of user embedding (w_u) and item embedding (h_i)
        for every user and item pair in user_tensor and item_tensor.

        :param user_tensor: [description]
        :type user_tensor: [type]
        :param item_tensor: [description]
        :type item_tensor: [type]
        """
        w_u = self.user_embedding_(user_tensor)
        h_i = self.item_embedding_(item_tensor)

        return w_u.matmul(h_i.T)


class BPRKNNModule(nn.Module):
    """PyTorch module for direct adaptive item similarities.

    Stores a trainable square item-to-item parameter matrix. The effective
    similarity matrix is made symmetric and its diagonal is set to zero, so an
    item cannot contribute to its own recommendation score. Scores are computed
    by multiplying a user's binary interaction history by this matrix.

    After training, :class:`BPRKNN` can replace the learned matrix with a
    top-K-pruned version used during prediction.

    :param num_items: Number of items in the interaction matrix. Determines both
        dimensions of the learned similarity matrix.
    :type num_items: int
    """

    def __init__(self, num_items: int):
        super().__init__()
        self.register_buffer("pruned_similarity_", None)
        self.item_similarity_ = nn.Parameter(torch.zeros((num_items, num_items)))

    def similarity_matrix(self) -> torch.Tensor:
        if self.pruned_similarity_ is not None:
            return self.pruned_similarity_
        similarities = (self.item_similarity_ + self.item_similarity_.T) / 2
        mask = 1 - torch.eye(similarities.shape[0], device=similarities.device)
        return similarities * mask

    def score_history(self, history: torch.Tensor, item_tensor: torch.Tensor) -> torch.Tensor:
        return history.matmul(self.similarity_matrix().T)[:, item_tensor]

    def regularization_loss(self, target_items, mnar_items, history, lambda_target, lambda_mnar):
        similarities = self.similarity_matrix()
        return (
            lambda_target * similarities[target_items].norm()
            + lambda_mnar * similarities[mnar_items].norm()
        )

    def set_similarity_matrix(self, similarities: np.ndarray) -> None:
        self.pruned_similarity_ = torch.as_tensor(
            similarities,
            dtype=self.item_similarity_.dtype,
            device=self.item_similarity_.device,
        )


class FactorizedBPRKNNModule(nn.Module):
    """PyTorch module for factorized adaptive item similarities.

    Represents the item similarity matrix as :math:`C = HH^T`, where each row
    of :math:`H` is a learned item embedding. Scores are calculated through the
    embeddings without materializing the dense :math:`|I| \times |I|` matrix.
    The contribution of an item's similarity with itself is subtracted from the
    score to match the zero diagonal used by the direct model.

    :param num_items: Number of items in the interaction matrix.
    :type num_items: int
    :param num_components: Size of each learned item embedding. The model stores
        ``num_items * num_components`` trainable values.
    :type num_components: int
    """

    def __init__(self, num_items: int, num_components: int):
        super().__init__()
        self.num_items = num_items
        self.num_components = num_components
        self.item_embedding_ = nn.Embedding(num_items, num_components)
        std = min(1 / num_components**0.5, 0.05)
        nn.init.normal_(self.item_embedding_.weight, std=std)

    def score_history(self, history: torch.Tensor, item_tensor: torch.Tensor) -> torch.Tensor:
        item_factors = self.item_embedding_(item_tensor)
        history_factors = history.matmul(self.item_embedding_.weight)
        scores = history_factors.matmul(item_factors.T)
        self_similarity = item_factors.pow(2).sum(dim=1)
        return scores - history[:, item_tensor] * self_similarity

    def regularization_loss(self, target_items, mnar_items, history, lambda_target, lambda_mnar):
        target_factors = self.item_embedding_(target_items)
        mnar_factors = self.item_embedding_(mnar_items)
        history_penalty = history.matmul(self.item_embedding_.weight.pow(2)).sum(dim=1).mean()
        return (
            lambda_target * (target_factors.norm() + history_penalty)
            + lambda_mnar * mnar_factors.norm()
        )
