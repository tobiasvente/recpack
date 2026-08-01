# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging
from math import sqrt
from typing import List, Optional, Tuple

from scipy.sparse import csr_matrix, lil_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from recpack.algorithms.base import TorchMLAlgorithm
from recpack.algorithms.samplers import (
    SequenceMiniBatchPositivesTargetsNegativesSampler,
    SequenceMiniBatchSampler,
)
from recpack.matrix import InteractionMatrix

logger = logging.getLogger("recpack")


class SASRec(TorchMLAlgorithm):
    """Self-Attentive Sequential Recommendation (SASRec).

    SASRec as presented in Kang, Wang-Cheng, and Julian McAuley.
    "Self-attentive sequential recommendation." ICDM 2018.

    The algorithm treats a user's interactions as an ordered sequence and
    trains a causal (left-to-right) Transformer to predict the next item at
    every position of the sequence::

                    iid_1_pred  iid_2_pred  iid_3_pred
                        |           |           |
                    [ causal self-attention blocks ]
                        |           |           |
                      iid_0       iid_1       iid_2

    Item and position embeddings are summed at the input; the output
    representation at each position is matched against the (shared) item
    embeddings to produce scores. Training uses the binary cross-entropy
    objective of the paper: at every position the observed next item should
    score high, and sampled negative items should score low.

    At prediction time the representation at the last position of a user's
    sequence scores all items.

    Sequences longer than ``max_len`` are truncated to their most recent
    ``max_len`` interactions, both during training and prediction.

    :param num_components: Size of the item and position embeddings.
        Defaults to 100
    :type num_components: int, optional
    :param num_blocks: Number of self-attention blocks. Defaults to 2
    :type num_blocks: int, optional
    :param num_heads: Number of attention heads per block. Defaults to 1
    :type num_heads: int, optional
    :param max_len: Maximum sequence length; longer histories are truncated
        to their most recent ``max_len`` interactions. Defaults to 50
    :type max_len: int, optional
    :param dropout: Dropout applied to the embeddings and the feed-forward
        layers. Defaults to 0.2
    :type dropout: float, optional
    :param num_negatives: Number of negative items sampled per position
        during training. Defaults to 1
    :type num_negatives: int, optional
    :param clipnorm: Clip the gradient's l2 norm, None for no clipping.
        Defaults to 1.0
    :type clipnorm: float, optional
    :param batch_size: Number of sequences per mini-batch. Defaults to 128
    :type batch_size: int, optional
    :param max_epochs: Max training runs through the entire dataset.
        Defaults to 10
    :type max_epochs: int, optional
    :param learning_rate: Adam learning rate. Defaults to 0.001
    :type learning_rate: float, optional
    :param stopping_criterion: Name of the stopping criterion to use for training.
        For available values,
        check :attr:`recpack.algorithms.stopping_criterion.StoppingCriterion.FUNCTIONS`
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
        Defaults to 0.0
    :type min_improvement: float, optional
    :param seed: Seed to the randomizers, useful for reproducible results,
        defaults to None
    :type seed: int, optional
    :param save_best_to_file: If true, the best model will be saved after training,
        defaults to False
    :type save_best_to_file: bool, optional
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
        num_blocks: int = 2,
        num_heads: int = 1,
        max_len: int = 50,
        dropout: float = 0.2,
        num_negatives: int = 1,
        clipnorm: float = 1.0,
        batch_size: int = 128,
        max_epochs: int = 10,
        learning_rate: float = 0.001,
        stopping_criterion: str = "recall",
        stop_early: bool = False,
        max_iter_no_change: int = 5,
        min_improvement: float = 0.0,
        seed: Optional[int] = None,
        save_best_to_file: bool = False,
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
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.max_len = max_len
        self.dropout = dropout
        self.num_negatives = num_negatives
        self.clipnorm = clipnorm

    def _init_model(self, X: InteractionMatrix) -> None:
        # Invalid item ID, used to pad sequences to equal length.
        self.num_items = X.shape[1]
        self.pad_token = self.num_items

        self.model_ = SASRecTorch(
            self.num_items,
            self.pad_token,
            num_components=self.num_components,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            max_len=self.max_len,
            dropout=self.dropout,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate, betas=(0.9, 0.98))

        self.predict_sampler = SequenceMiniBatchSampler(self.pad_token, batch_size=self.batch_size)

        self.fit_sampler = SequenceMiniBatchPositivesTargetsNegativesSampler(
            self.num_negatives, self.pad_token, batch_size=self.batch_size
        )

    def _transform_fit_input(
        self,
        X: InteractionMatrix,
        validation_data: Tuple[InteractionMatrix, InteractionMatrix],
    ):
        """Check the training and validation data are InteractionMatrices
        with timestamps, and pass them through unchanged.
        """
        self._assert_is_interaction_matrix(X, *validation_data)
        self._assert_has_timestamps(X, *validation_data)

        return X, validation_data

    def _transform_predict_input(self, X: InteractionMatrix) -> InteractionMatrix:
        self._assert_is_interaction_matrix(X)
        self._assert_has_timestamps(X)
        return X

    def _truncate_seq_batch(self, seq_batch: torch.LongTensor, *aligned_batches: torch.LongTensor):
        """Truncate right-padded sequence batches to the most recent ``max_len`` entries.

        All ``aligned_batches`` are truncated with the same per-row window
        as ``seq_batch``, so positives, targets and negatives stay aligned.
        """
        max_hist_len = seq_batch.shape[1]
        if max_hist_len <= self.max_len:
            return (seq_batch, *aligned_batches)

        lengths = (seq_batch != self.pad_token).sum(dim=1)
        starts = torch.clamp(lengths - self.max_len, min=0)
        window = starts.unsqueeze(1) + torch.arange(self.max_len, device=seq_batch.device)

        truncated = [seq_batch.gather(1, window)]
        for t in aligned_batches:
            if t.dim() == 2:
                truncated.append(t.gather(1, window))
            else:
                truncated.append(t.gather(1, window.unsqueeze(-1).expand(-1, -1, t.shape[2])))

        return tuple(truncated)

    def _train_epoch(self, X: InteractionMatrix) -> List[float]:
        losses = []

        for _, positives_batch, targets_batch, negatives_batch in self.fit_sampler.sample(X):
            positives_batch, targets_batch, negatives_batch = self._truncate_seq_batch(
                positives_batch, targets_batch, negatives_batch
            )
            # positives shape = (batch_size x max_hist_len)
            # targets shape = (batch_size x max_hist_len)
            # negatives shape = (batch_size x max_hist_len x num_negatives)
            positives_batch = positives_batch.to(self.device)
            targets_batch = targets_batch.to(self.device)
            negatives_batch = negatives_batch.to(self.device)

            self.optimizer.zero_grad()

            # Scores for every item at every position, (batch_size x max_hist_len x num_items + 1)
            output = self.model_(positives_batch)

            true_input_mask = targets_batch != self.pad_token
            loss = self._compute_loss(output, targets_batch, negatives_batch, true_input_mask)

            loss.backward()
            losses.append(loss.item())

            if self.clipnorm:
                nn.utils.clip_grad_norm_(self.model_.parameters(), self.clipnorm)

            self.optimizer.step()

        return losses

    def _compute_loss(
        self,
        output: torch.FloatTensor,
        targets_chunk: torch.LongTensor,
        negatives_chunk: torch.LongTensor,
        true_input_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Binary cross-entropy loss of the SASRec paper.

        At every non-padded position, the target item should score high
        and each sampled negative item should score low.
        """
        # positive_scores has shape (batch_size x max_hist_len)
        positive_scores = torch.gather(output, 2, targets_chunk.unsqueeze(-1)).squeeze(-1)
        # negative_scores has shape (batch_size x max_hist_len x num_negatives)
        negative_scores = torch.gather(output, 2, negatives_chunk)

        num_negatives = negative_scores.shape[2]
        negative_mask = true_input_mask.unsqueeze(-1).expand(-1, -1, num_negatives)

        positive_loss = -F.logsigmoid(positive_scores[true_input_mask]).sum()
        # log(1 - sigmoid(x)) == logsigmoid(-x)
        negative_loss = -F.logsigmoid(-negative_scores[negative_mask]).sum()

        num_scores = true_input_mask.sum() * (1 + num_negatives)

        return (positive_loss + negative_loss) / num_scores

    def _predict(self, X: InteractionMatrix) -> csr_matrix:
        X_pred = lil_matrix(X.shape)
        self.model_.eval()
        with torch.no_grad():
            for uid_batch, positives_batch in self.predict_sampler.sample(X):
                (positives_batch,) = self._truncate_seq_batch(positives_batch)
                positives_batch = positives_batch.to(self.device)

                # (batch_size x max_hist_len x num_items + 1)
                output = self.model_(positives_batch)

                # Score all items with the representation
                # at the last non-padded position of each sequence.
                last_item_ix = (positives_batch != self.pad_token).sum(dim=1) - 1
                item_scores = (
                    output[torch.arange(output.shape[0], dtype=int), last_item_ix].detach().cpu().numpy()
                )

                # Slice off the padding token scores.
                X_pred[uid_batch.detach().cpu().numpy()] = self._get_top_k_recommendations(
                    csr_matrix(item_scores[:, :-1])
                )

        return X_pred.tocsr()


class SASRecTorch(nn.Module):
    """SASRec torch module: item + position embeddings followed by
    causal self-attention blocks.

    Scores are computed against the shared item embeddings,
    as in the original paper.

    :param num_items: Number of items
    :type num_items: int
    :param pad_token: Index of the padding token
    :type pad_token: int
    :param num_components: Size of the item and position embeddings, defaults to 100
    :type num_components: int, optional
    :param num_blocks: Number of self-attention blocks, defaults to 2
    :type num_blocks: int, optional
    :param num_heads: Number of attention heads per block, defaults to 1
    :type num_heads: int, optional
    :param max_len: Maximum sequence length, defaults to 50
    :type max_len: int, optional
    :param dropout: Dropout applied to embeddings and feed-forward layers, defaults to 0.2
    :type dropout: float, optional
    """

    def __init__(
        self,
        num_items: int,
        pad_token: int,
        num_components: int = 100,
        num_blocks: int = 2,
        num_heads: int = 1,
        max_len: int = 50,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_items = num_items
        self.pad_token = pad_token
        self.num_components = num_components
        self.max_len = max_len

        # Padding token gets a zero-valued, non-updated embedding.
        self.emb = nn.Embedding(num_items + 1, num_components, padding_idx=pad_token)
        self.pos_emb = nn.Embedding(max_len, num_components)
        self.dropout = nn.Dropout(dropout)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(num_components))
            self.attention_layers.append(
                nn.MultiheadAttention(num_components, num_heads, dropout=dropout, batch_first=True)
            )
            self.forward_layernorms.append(nn.LayerNorm(num_components))
            self.forward_layers.append(
                nn.Sequential(
                    nn.Linear(num_components, num_components),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(num_components, num_components),
                    nn.Dropout(dropout),
                )
            )

        self.last_layernorm = nn.LayerNorm(num_components)

        nn.init.normal_(self.emb.weight, std=0.01)
        nn.init.normal_(self.pos_emb.weight, std=0.01)

        # Set the embedding of the padding token to 0.
        with torch.no_grad():
            self.emb.weight[pad_token] = torch.zeros(num_components)

    def forward(self, seq: torch.LongTensor) -> torch.Tensor:
        """Compute scores for every item at every position of the sequences.

        :param seq: Right-padded item id sequences, shape (batch_size, max_hist_len)
            with max_hist_len <= max_len.
        :type seq: torch.LongTensor
        :return: Scores for every item (including the padding token, as final column)
            at every position, shape (batch_size, max_hist_len, num_items + 1).
        :rtype: torch.Tensor
        """
        batch_size, max_hist_len = seq.shape
        pad_mask = seq == self.pad_token

        positions = torch.arange(max_hist_len, device=seq.device)
        x = self.emb(seq) * sqrt(self.num_components) + self.pos_emb(positions).unsqueeze(0)
        x = self.dropout(x)
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        # Causal mask: position t may only attend to positions <= t.
        causal_mask = torch.triu(
            torch.ones(max_hist_len, max_hist_len, dtype=torch.bool, device=seq.device), diagonal=1
        )

        for i in range(len(self.attention_layers)):
            q = self.attention_layernorms[i](x)
            attn_output, _ = self.attention_layers[i](
                q, q, q, attn_mask=causal_mask, key_padding_mask=pad_mask, need_weights=False
            )
            x = x + attn_output

            x = x + self.forward_layers[i](self.forward_layernorms[i](x))
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        x = self.last_layernorm(x)

        # Scores against the shared item embeddings.
        return x.matmul(self.emb.weight.T)
