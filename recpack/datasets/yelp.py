# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Marouan El Marnissi

import os
from typing import List
import gzip
import zipfile
import tarfile
import shutil
import urllib.request

import numpy as np
import pandas as pd

from recpack.datasets.base import Dataset, _fetch_remote
from recpack.preprocessing.filters import (
    Filter,
    MinUsersPerItem,
    MinRating,
)


class YelpOpenDataset(Dataset):
    """
    Handles the Yelp Open Dataset
    """

    USER_IX = "user_id"
    """Name of the column in the DataFrame that contains user identifiers."""
    ITEM_IX = "business_id"
    """Name of the column in the DataFrame that contains item identifiers."""
    RATING_IX = "stars"
    """Name of the column in the DataFrame that contains the rating a user gave to the item."""
    TIMESTAMP_IX = "timestamp"
    """Name of the column in the DataFrame that contains time of interaction in seconds since epoch."""

    DATASETURL = "https://business.yelp.com/external-assets/files/Yelp-JSON.zip"

    REMOTE_ZIPNAME = "Yelp-JSON.zip"
    """Name of the zip-file on the Yelp server."""

    REMOTE_FOLDERNAME = "Yelp JSON"
    """Name of the folder in the zip-file containing the rating tar-file."""

    REMOTE_TARNAME = "yelp_dataset.tar"
    """Name of the tar-file containing the user ratings."""

    REMOTE_FILENAME = "yelp_academic_dataset_review.json"
    """Name of the file containing user ratings in the tar file."""

    def __init__(self, path: str = "data", filename: str = None, use_default_filters=True):
        if filename and not filename.endswith(".json"):
            filename = filename + ".json"
        Dataset.__init__(self, path=path, filename=filename, use_default_filters=use_default_filters)

    @property
    def DEFAULT_FILENAME(self) -> str:
        """Default filename that will be used if it is not specified by the user."""
        return f"{self.REMOTE_FILENAME}"

    @property
    def _default_filters(self) -> List[Filter]:
        """The default filters for the Yelp Open Dataset.

        Default processing  is done as in "Variational autoencoders for collaborative filtering." Liang, Dawen, et al.:

        - Ratings above or equal to 4 are interpreted as implicit feedback
        - Each remaining item has been interacted with by at least 5 users

        :return: List of filters to use as default preprocessing.
        :rtype: List[Filter]
        """
        return [
            MinRating(4, self.RATING_IX),
            MinUsersPerItem(5, self.ITEM_IX, self.USER_IX),
        ]

    def _download_dataset(self):
        """Downloads the dataset.

        Downloads the zipfile, and extracts the ratings file to `self.file_path`
        """
        os.makedirs(self.path, exist_ok=True)
        archive_path = os.path.join(self.path, self.REMOTE_ZIPNAME)

        # Yelp may reject automated downloads with HTTP 403. In that case users
        # can download the archive in a browser and place it at archive_path.
        if not os.path.exists(archive_path):
            partial_path = archive_path + ".part"
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "application/zip,application/octet-stream,*/*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://business.yelp.com/data/resources/open-dataset/",
            }
            request = urllib.request.Request(self.DATASETURL, headers=headers)

            try:
                with urllib.request.urlopen(request) as response, open(partial_path, "wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
                os.replace(partial_path, archive_path)
            except Exception as error:
                if os.path.exists(partial_path):
                    os.remove(partial_path)
                raise RuntimeError(
                    "Automatic download of the Yelp Open Dataset failed. "
                    "Open https://business.yelp.com/data/resources/open-dataset/ "
                    f"in a browser, download the JSON archive, save it as '{archive_path}', "
                    "and call fetch_dataset() again."
                ) from error

        # Extract the tarfile which contains the rating file
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extract(f"{self.REMOTE_FOLDERNAME}/{self.REMOTE_TARNAME}", self.path)

        # Extract the ratings file which we will use
        with tarfile.open(os.path.join(self.path, f"{self.REMOTE_FOLDERNAME}/{self.REMOTE_TARNAME}"), "r") as tar_ref:
            tar_ref.extractall(path=self.path, members=[tar_ref.getmember(self.REMOTE_FILENAME)])

        # rename the ratings file
        os.rename(os.path.join(self.path, self.REMOTE_FILENAME), self.file_path)

        # delete the zip file and folder containing the tar file
        os.remove(archive_path)
        shutil.rmtree(os.path.join(self.path, self.REMOTE_FOLDERNAME))

    def _load_dataframe(self) -> pd.DataFrame:
        """Load the raw dataset from file, and return it as a pandas DataFrame.

        .. warning::

            This does not apply any preprocessing, and returns the raw dataset.

        :return: The interaction data as a DataFrame with a row per interaction.
        :rtype: pd.DataFrame
        """

        # load the json in chunks
        chunks = pd.read_json(
            self.file_path,
            lines=True,
            chunksize=100_000,
            dtype={self.ITEM_IX: "string", self.USER_IX: "string", self.RATING_IX: np.int64},
        )

        # convert datetime into seconds since epoch and keep relevant columns
        processed = []
        for chunk in chunks:
            dt = pd.to_datetime(chunk["date"], utc=True)
            chunk[self.TIMESTAMP_IX] = dt.astype("int64") // 10**9
            chunk = chunk[[self.ITEM_IX, self.USER_IX, self.RATING_IX, self.TIMESTAMP_IX]]
            processed.append(chunk)

        # concatenate the chunks to form a pandas dataframe
        df = pd.concat(processed, ignore_index=True)

        return df
