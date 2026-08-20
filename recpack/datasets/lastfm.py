# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert
import os
from typing import List
import zipfile

import numpy as np
import pandas as pd

from recpack.datasets.base import Dataset, _fetch_remote
from recpack.preprocessing.filters import (
    Filter,
    MinItemsPerUser,
    MinUsersPerItem,
)


class LastFM(Dataset):
    """Handles the HetRec 2011 Last.fm dataset.

    Full information on the dataset can be found at
    https://grouplens.org/datasets/hetrec-2011/. Uses the
    ``user_artists.dat`` file, which lists for every (user, artist) pair
    the number of times the user listened to that artist (the ``weight``
    column), to construct an implicit feedback interaction matrix. Unlike
    the MovieLens datasets, this dataset has no per-interaction timestamp.

    Default processing makes sure that:

    - Each remaining user has interacted with at least 5 items
    - Each remaining item has been interacted with by at least 5 users

    :param path: The path to the data directory.
        Defaults to `data`
    :type path: str, optional
    :param filename: Name of the file, if no name is provided the dataset default will be used if known.
    :type filename: str, optional
    :param use_default_filters: Should a default set of filters be initialised? Defaults to True
    :type use_default_filters: bool, optional
    """

    USER_IX = "userID"
    """Name of the column in the DataFrame that contains user identifiers."""
    ITEM_IX = "artistID"
    """Name of the column in the DataFrame that contains item identifiers."""
    WEIGHT_IX = "weight"
    """Name of the column in the DataFrame that contains the number of
    times a user listened to the artist."""

    DATASETURL = "https://files.grouplens.org/datasets/hetrec2011"
    """URL of the directory serving the HetRec 2011 datasets."""

    REMOTE_ZIPNAME = "hetrec2011-lastfm-2k"
    """Name of the zip-file on the GroupLens server."""

    REMOTE_FILENAME = "user_artists.dat"
    """Name of the file containing user-artist listening counts."""

    DEFAULT_FILENAME = "user_artists.dat"
    """Default filename that will be used if it is not specified by the user."""

    @property
    def _default_filters(self) -> List[Filter]:
        """The default filters for the Last.fm dataset.

        Filters users and items that do not have enough interactions.

        :return: List of filters to use as default preprocessing.
        :rtype: List[Filter]
        """
        return [
            MinItemsPerUser(5, self.ITEM_IX, self.USER_IX),
            MinUsersPerItem(5, self.ITEM_IX, self.USER_IX),
        ]

    def _download_dataset(self):
        """Downloads the dataset.

        Downloads the zipfile, and extracts the ``user_artists.dat`` file
        to ``self.file_path``. Unlike the MovieLens zip archives, the
        HetRec 2011 archive stores its files at the root of the zip,
        without a containing subdirectory.
        """
        _fetch_remote(
            f"{self.DATASETURL}/{self.REMOTE_ZIPNAME}.zip",
            os.path.join(self.path, f"{self.REMOTE_ZIPNAME}.zip"),
        )

        with zipfile.ZipFile(os.path.join(self.path, f"{self.REMOTE_ZIPNAME}.zip"), "r") as zip_ref:
            zip_ref.extract(self.REMOTE_FILENAME, self.path)

        extracted_path = os.path.join(self.path, self.REMOTE_FILENAME)
        if extracted_path != self.file_path:
            os.rename(extracted_path, self.file_path)

    def _load_dataframe(self) -> pd.DataFrame:
        """Load the raw dataset from file, and return it as a pandas DataFrame.

        .. warning::

            This does not apply any preprocessing, and returns the raw dataset.

        :return: The interaction data as a DataFrame with a row per interaction.
        :rtype: pd.DataFrame
        """
        self.fetch_dataset()

        df = pd.read_csv(
            self.file_path,
            sep="\t",
            dtype={
                self.USER_IX: np.int64,
                self.ITEM_IX: np.int64,
                self.WEIGHT_IX: np.int64,
            },
        )

        return df
