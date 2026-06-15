# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Marouan El Marnissi
import os
from typing import List
import gzip
import shutil

import numpy as np
import pandas as pd
from enum import Enum

from recpack.datasets.base import Dataset, _fetch_remote
from recpack.preprocessing.filters import (
    Filter,
    MinUsersPerItem,
    MinRating,
)


class AmazonDataset(Dataset):
    """
    Handles a collection of different Amazon datasets
    """

    USER_IX = "user_id"
    """Name of the column in the DataFrame that contains user identifiers."""
    ITEM_IX = "parent_asin"
    """Name of the column in the DataFrame that contains the parent Amazon Standard Identification Number of the product."""
    RATING_IX = "rating"
    """Name of the column in the DataFrame that contains the rating a user gave to the item."""
    TIMESTAMP_IX = "timestamp"
    """Name of the column in the DataFrame that contains time of interaction in seconds since epoch."""

    DATASETURL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/0core/rating_only/"

    REMOTE_FILENAME = ""
    """Name of the file containing user reviews on the Amazon server."""

    class Category(Enum):
        ALL_BEAUTY = "All_Beauty.csv"
        AMAZON_FASHION = "Amazon_Fashion.csv"
        APPLIANCES = "Appliances.csv"
        ARTS_CRAFTS_AND_SEWING = "Arts_Crafts_and_Sewing.csv"
        AUTOMOTIVE = "Automotive.csv"
        BABY_PRODUCTS = "Baby_Products.csv"
        BEAUTY_AND_PERSONAL_CARE = "Beauty_and_Personal_Care.csv"
        BOOKS = "Books.csv"
        CDS_AND_VINYL = "CDs_and_Vinyl.csv"
        CELL_PHONES_AND_ACCESSORIES = "Cell_Phones_and_Accessories.csv"
        CLOTHING_SHOES_AND_JEWELRY = "Clothing_Shoes_and_Jewelry.csv"
        DIGITAL_MUSIC = "Digital_Music.csv"
        ELECTRONICS = "Electronics.csv"
        GIFT_CARDS = "Gift_Cards.csv"
        GROCERY_AND_GOURMET_FOOD = "Grocery_and_Gourmet_Food.csv"
        HANDMADE_PRODUCTS = "Handmade_Products.csv"
        HEALTH_AND_HOUSEHOLD = "Health_and_Household.csv"
        HEALTH_AND_PERSONAL_CARE = "Health_and_Personal_Care.csv"
        HOME_AND_KITCHEN = "Home_and_Kitchen.csv"
        INDUSTRIAL_AND_SCIENTIFIC = "Industrial_and_Scientific.csv"
        KINDLE_STORE = "Kindle_Store.csv"
        MAGAZINE_SUBSCRIPTIONS = "Magazine_Subscriptions.csv"
        MOVIES_AND_TV = "Movies_and_TV.csv"
        MUSICAL_INSTRUMENTS = "Musical_Instruments.csv"
        OFFICE_PRODUCTS = "Office_Products.csv"
        PATIO_LAWN_AND_GARDEN = "Patio_Lawn_and_Garden.csv"
        PET_SUPPLIES = "Pet_Supplies.csv"
        SOFTWARE = "Software.csv"
        SPORTS_AND_OUTDOORS = "Sports_and_Outdoors.csv"
        SUBSCRIPTION_BOXES = "Subscription_Boxes.csv"
        TOOLS_AND_HOME_IMPROVEMENT = "Tools_and_Home_Improvement.csv"
        TOYS_AND_GAMES = "Toys_and_Games.csv"
        VIDEO_GAMES = "Video_Games.csv"
        UNKNOWN = "Unknown.csv"

    def __init__(self, category: Category, path: str = "data", filename: str = None, use_default_filters=True):
        self.REMOTE_FILENAME = category.value
        if filename and not filename.endswith(".csv"):
            filename = filename + ".csv"
        Dataset.__init__(self, path=path, filename=filename, use_default_filters=use_default_filters)

    @property
    def DEFAULT_FILENAME(self) -> str:
        """Default filename that will be used if it is not specified by the user."""
        return f"{self.REMOTE_FILENAME}"

    @property
    def _default_filters(self) -> List[Filter]:
        """The default filters for the Amazon datasets.

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
        # Download the gzip into the data directory
        try:
            _fetch_remote(
                f"{self.DATASETURL}/{self.REMOTE_FILENAME}.gz", os.path.join(self.path, f"{self.REMOTE_FILENAME}.gz")
            )
        except Exception as e:
            raise RuntimeError(f"Failed to remotely fetch dataset: {e}")

        # Extract the ratings file which we will use
        with gzip.open(os.path.join(self.path, f"{self.REMOTE_FILENAME}.gz"), "rb") as f_in:
            with open(f"{self.path}/{self.REMOTE_FILENAME}", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # rename the ratings file
        os.rename(os.path.join(self.path, self.REMOTE_FILENAME), self.file_path)

        # delete the gzip file
        os.remove(os.path.join(self.path, f"{self.REMOTE_FILENAME}.gz"))

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
            dtype={
                self.ITEM_IX: "string",
                self.USER_IX: "string",
                self.RATING_IX: np.int64,
                self.TIMESTAMP_IX: np.int64,
            },
        )

        return df
