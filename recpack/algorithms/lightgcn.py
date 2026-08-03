# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging
from typing import List

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, lil_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from recpack.algorithms.base import TorchMLAlgorithm
from recpack.algorithms.loss_functions import bpr_loss
from recpack.algorithms.samplers import BootstrapSampler
from recpack.util import to_binary

logger = logging.getLogger("recpack")


class LightGCN(TorchMLAlgorithm):
    """LightGCN graph convolution algorithm for collaborative filtering.

    LightGCN as presented in He, Xiangnan, et al.
    "LightGCN: Simplifying and powering graph convolution network for
    recommendation." SIGIR 2020.

    User and item embeddings are smoothed by propagating them over the
    normalized user-item interaction graph. Each propagation layer computes

    .. math::

        E^{(k+1)} = (D^{-1/2} A D^{-1/2}) E^{(k)}

    where :math:`A` is the adjacency matrix of the bipartite interaction
    graph and :math:`D` its degree matrix.
    The final representation of a user or item is the mean of its
    embeddings at every layer :math:`0..K`.
    Scores are the dot products of the final user and item embeddings.

    Training optimises the BPR criterion on sampled (user, positive item,
    negative item) triplets, with L2 regularization on the layer-0
    (ego) embeddings of the sampled entities.

    :param num_components: The size of the latent vectors for both users and items.
        Defaults to 64
    :type num_components: int, optional
    :param num_layers: The number of graph convolution (propagation) layers.
        Defaults to 3
    :type num_layers: int, optional
    :param lambda_reg: L2 regularization coefficient applied to the ego
        embeddings of the entities in each training batch. Defaults to 1e-4
    :type lambda_reg: float, optional
    :param batch_size: Size of the batches to use during gradient descent. Defaults to 1000.
    :type batch_size: int, optional
    :param max_epochs: The max amount of epochs to train the model, defaults to 20
    :type max_epochs: int, optional
    :param learning_rate: The learning rate of the optimization procedure,
        defaults to 0.001
    :type learning_rate: float, optional
    :param stopping_criterion: Which criterion to use to optimise the parameters,
        a string which indicates the name of the stopping criterion.
        Which criterions are available can be found at
        :attr:`recpack.algorithms.stopping_criterion.StoppingCriterion.FUNCTIONS`.
        Defaults to 'bpr'
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
    :param seed: Seed to fix random numbers, to make results reproducible,
        defaults to None
    :type seed: int, optional
    :param save_best_to_file: If True, the best model is saved to disk after fit.
    :type save_best_to_file: bool, optional
    :param sample_size: How many samples to take during a training epoch
        using bootstrap sampling.
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
        num_components: int = 64,
        num_layers: int = 3,
        lambda_reg: float = 1e-4,
        batch_size: int = 1_000,
        max_epochs: int = 20,
        learning_rate: float = 0.001,
        stopping_criterion: str = "bpr",
        stop_early: bool = False,
        max_iter_no_change: int = 5,
        min_improvement: float = 0.01,
        seed: int = None,
        save_best_to_file: bool = False,
        sample_size: int = None,
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

        self.num_components = num_components
        self.num_layers = num_layers
        self.lambda_reg = lambda_reg

        self.sample_size = sample_size

        self.sampler = BootstrapSampler(
            num_negatives=1,
            batch_size=self.batch_size,
        )

    def _init_model(self, X: csr_matrix):
        num_users, num_items = X.shape
        self.model_ = LightGCNModule(
            num_users,
            num_items,
            X,
            num_components=self.num_components,
            num_layers=self.num_layers,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

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

    def _train_epoch(self, train_data: csr_matrix):
        """Train a single epoch. Uses sampler to generate samples,
        and loop through them in batches of self.batch_size.
        After each batch, update the parameters according to gradients.

        :param train_data: interaction matrix.
        :type train_data: csr_matrix
        """
        losses = []

        for users, target_items, mnar_items in tqdm(
            self.sampler.sample(
                train_data,
                sample_size=self.sample_size,
            ),
            desc="train_epoch LightGCN",
        ):
            users = users.to(self.device)
            # Target items are items the user has interacted with,
            # and we expect to recommend high
            target_items = target_items.to(self.device)
            # Items the user has not seen, and assuming MNAR data
            # Is a batch_size x 1 matrix, squeeze into an array.
            mnar_items = mnar_items.squeeze(-1).to(self.device)

            self.optimizer.zero_grad()

            user_final, item_final = self.model_.propagate()

            positive_sim = (user_final[users] * item_final[target_items]).sum(dim=1)
            negative_sim = (user_final[users] * item_final[mnar_items]).sum(dim=1)

            loss = self._compute_loss(users, target_items, mnar_items, positive_sim, negative_sim)
            loss.backward()
            losses.append(loss.item())
            self.optimizer.step()

        return losses

    def _compute_loss(self, users, target_items, mnar_items, positive_sim, negative_sim):
        loss = bpr_loss(positive_sim, negative_sim)

        # L2 regularization on the ego (layer 0) embeddings of the batch,
        # as in the LightGCN paper.
        reg = (
            self.model_.user_embedding_(users).pow(2).sum()
            + self.model_.item_embedding_(target_items).pow(2).sum()
            + self.model_.item_embedding_(mnar_items).pow(2).sum()
        )
        loss = loss + self.lambda_reg * reg / users.shape[0]

        return loss


class LightGCNModule(nn.Module):
    """LightGCN torch module, encodes the embeddings
    and the graph propagation functionality.

    :param num_users: the amount of users
    :type num_users: int
    :param num_items: the amount of items
    :type num_items: int
    :param X: Binary user-item interaction matrix used to construct
        the normalized adjacency matrix of the propagation graph.
    :type X: csr_matrix
    :param num_components: The size of the embedding per user and item, defaults to 64
    :type num_components: int, optional
    :param num_layers: The number of propagation layers, defaults to 3
    :type num_layers: int, optional
    """

    def __init__(self, num_users: int, num_items: int, X: csr_matrix, num_components: int = 64, num_layers: int = 3):
        super().__init__()

        self.num_components = num_components
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers

        self.user_embedding_ = nn.Embedding(num_users, num_components)
        self.item_embedding_ = nn.Embedding(num_items, num_components)

        # Keep variance low enough, to allow learning
        self.std = min(1 / num_components ** 0.5, 0.05)
        nn.init.normal_(self.user_embedding_.weight, std=self.std)
        nn.init.normal_(self.item_embedding_.weight, std=self.std)

        # register_buffer makes the adjacency move along
        # with the module between devices, and be saved with it.
        self.register_buffer("norm_adjacency", self._normalized_adjacency(X))

        self._cached_propagation = None

    def _normalized_adjacency(self, X: csr_matrix) -> torch.Tensor:
        """Construct the symmetrically normalized adjacency matrix
        D^-1/2 A D^-1/2 of the bipartite interaction graph as a
        torch sparse tensor.
        """
        R = to_binary(csr_matrix(X))
        A = sp.bmat([[None, R], [R.T, None]], format="csr")

        degrees = np.asarray(A.sum(axis=1)).flatten()
        with np.errstate(divide="ignore"):
            d_inv_sqrt = np.power(degrees, -0.5)
        # Users or items without interactions have degree 0.
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0

        D_inv_sqrt = sp.diags(d_inv_sqrt)
        A_norm = (D_inv_sqrt @ A @ D_inv_sqrt).tocoo()

        indices = torch.LongTensor(np.vstack([A_norm.row, A_norm.col]))
        values = torch.FloatTensor(A_norm.data)

        return torch.sparse_coo_tensor(indices, values, A_norm.shape).coalesce()

    def train(self, mode: bool = True):
        # Embeddings change with every update,
        # so a cached propagation is no longer valid.
        self._cached_propagation = None
        return super().train(mode)

    def propagate(self):
        """Propagate the ego embeddings through the graph.

        :return: The final user and item embeddings,
            the mean over the embeddings at every layer.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        if not self.training and self._cached_propagation is not None:
            return self._cached_propagation

        embeddings = torch.cat([self.user_embedding_.weight, self.item_embedding_.weight], dim=0)
        all_layers = [embeddings]

        for _ in range(self.num_layers):
            embeddings = torch.sparse.mm(self.norm_adjacency, embeddings)
            all_layers.append(embeddings)

        final = torch.stack(all_layers, dim=0).mean(dim=0)
        user_final, item_final = torch.split(final, [self.num_users, self.num_items], dim=0)

        if not self.training:
            self._cached_propagation = (user_final, item_final)

        return user_final, item_final

    def forward(self, user_tensor: torch.Tensor, item_tensor: torch.Tensor) -> torch.Tensor:
        """Compute the dot product of the final (propagated) user and item
        embeddings for every user and item pair in user_tensor and item_tensor.

        :param user_tensor: 1D tensor with user ids.
        :type user_tensor: torch.Tensor
        :param item_tensor: 1D tensor with item ids.
        :type item_tensor: torch.Tensor
        """
        user_final, item_final = self.propagate()

        return user_final[user_tensor].matmul(item_final[item_tensor].T)
