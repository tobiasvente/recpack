# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from typing import Optional

from recpack.matrix import InteractionMatrix
from recpack.scenarios import Scenario
from recpack.scenarios.splitters import FractionInteractionSplitter


class WeakGeneralization(Scenario):
    """Predict (randomly) held-out interactions for all users, with remaining data used for training.

    For each user their events ``I_u`` are distributed over the datasets as follows:

    - :attr:`full_training_data` (``IT_u``) contains ``frac_data_in * |I_u|`` (rounded up)
      of the user's interactions in the full dataset.
    - :attr:`test_data_in` contains the same events as :attr:`full_training_data`.
    - :attr:`test_data_out` contains the remaining ``(1 - frac_data_in) * |I_u|``
      of the user's interactions in the full dataset.
    - :attr:`validation_training_data` contains ``frac_data_in * |IT_u|`` (rounded up)
      of the user's interactions in the full training dataset.
    - :attr:`validation_data_in` contains the same events as :attr:`validation_training_data`
    - :attr:`validation_data_out` contains the remaining ``(1 - frac_data_in) * |IT_u|``
      (rounded down) of the user's interactions in the full training dataset.

    **Example**

    As an example, splitting following data with ``data_in_frac = 0.5``,
    and ``validation = True``::

        item    0   1   2   3   4   5
        Alice   X   X   X
        Bob             X   X   X   X

    would yield full_training_data::

        item    0   1   2   3   4   5
        Alice   X   X
        Bob             X       X

    validation_training_data::

        item    0   1   2   3   4   5
        Alice       X
        Bob             X

    validation_data_in::

        item    0   1   2   3   4   5
        Alice       X
        Bob             X

    validation_data_out::

        item    0   1   2   3   4   5
        Alice   X
        Bob                     X

    test_data_in::

        item    0   1   2   3   4   5
        Alice   X   X
        Bob             X       X

    test_data_out::

        item    0   1   2   3   4   5
        Alice           X
        Bob                 X       X

    :param frac_data_in: Fraction of interactions per user used for
        training. The interactions are randomly chosen.
        Defaults to 0.8.
    :type frac_data_in: float, optional
    :param validation: Assign a portion of the training dataset to validation data
        if True,
        else split without validation data into only a training and test dataset.
    :type validation: boolean, optional
    :param seed: The seed to use for the random components of the splitter.
        Used as fallback for the per-step seeds below.
        If None, a random seed will be used. Defaults to None
    :type seed: int, optional
    :param train_test_seed: Seed for the split of the full dataset into
        the full training data and the test hold-out.
        In this scenario this single split determines both the training set
        and the test data.
        Defaults to None, which falls back to ``seed``.
    :type train_test_seed: int, optional
    :param validation_seed: Seed for the split of the full training data
        into validation training and validation hold-out data.
        Defaults to None, which falls back to ``seed``.
    :type validation_seed: int, optional
    :param test_split_seed: Accepted for API consistency with the other
        scenarios. The test fold-in data is a copy of the full training data
        in this scenario, so this seed drives no additional random step.
        Defaults to None, which falls back to ``seed``.
    :type test_split_seed: int, optional
    """

    def __init__(
        self,
        frac_data_in: float = 0.8,
        validation: bool = False,
        seed: int = None,
        train_test_seed: Optional[int] = None,
        validation_seed: Optional[int] = None,
        test_split_seed: Optional[int] = None,
    ):
        super().__init__(
            validation=validation,
            seed=seed,
            train_test_seed=train_test_seed,
            validation_seed=validation_seed,
            test_split_seed=test_split_seed,
        )

        self.frac_data_in = frac_data_in

        self.interaction_split = FractionInteractionSplitter(self.frac_data_in, seed=self.train_test_seed)

        if validation:
            self.validation_splitter = FractionInteractionSplitter(
                self.frac_data_in,
                seed=self.validation_seed,
            )

    def _split(self, data: InteractionMatrix):
        """Splits your data so that a user has interactions in all of
            training, validation and test set.

        :param data: Interaction matrix to be split.
        :type data: InteractionMatrix
        """
        self._full_train_X, self._test_data_out = self.interaction_split.split(data)

        if self.validation:
            (
                self._validation_train_X,
                self._validation_data_out,
            ) = self.validation_splitter.split(self._full_train_X)
            self._validation_data_in = self._validation_train_X.copy()

        self._test_data_in = self._full_train_X.copy()
