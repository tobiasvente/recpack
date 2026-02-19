# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Marouan El Marnissi

import numpy as np
import pandas as pd
import pytest
import socket
import os
import shutil


from pandas import DataFrame, StringDtype

from recpack.datasets.yelp import YelpOpenDataset
from recpack.matrix import InteractionMatrix


@pytest.fixture
def require_network():
    "tests if there is a network connection"
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)
    except OSError:
        pytest.skip("No network available")

@pytest.fixture(scope="session")
def dataset_path():
    path = "recpack/tests/test_datasets/downloaded_during_test"
    yield path
    shutil.rmtree(path, ignore_errors=True)

@pytest.fixture
def dataset_filename():
    filename = "yelpOpenDataset"
    return filename

@pytest.mark.network
@pytest.mark.slow
def test_fetch_dataset(require_network, dataset_path, dataset_filename):
    dataset = YelpOpenDataset(dataset_path, dataset_filename)
    dataset._download_dataset()
    assert os.path.exists(f"{dataset_path}/{dataset_filename}.json") # check if the dataset json file was properly downloaded and extracted

def test_create_dataframe(require_network, dataset_path, dataset_filename):
    assert os.path.exists(f"{dataset_path}/{dataset_filename}.json"), "dataset was not properly fetched in a previous test"
    dataset = YelpOpenDataset(dataset_path, dataset_filename)
    df = dataset._load_dataframe()
    # assert that a dataframe object is being created/returned
    assert isinstance(df, pd.DataFrame)

    # assert that the dataframe is not empty
    assert not df.empty

    #assert that the columns match the expected column names
    assert set(df.columns) == {"user_id", "business_id", "stars", "timestamp"}

    #assert that no null values are present in any of the columns
    assert df["user_id"].isnull().sum() == 0
    assert df["business_id"].isnull().sum() == 0
    assert df["stars"].isnull().sum() == 0
    assert df["timestamp"].isnull().sum() == 0

    #assert that the types for the columns match the expected types
    assert isinstance(df["user_id"].dtype,StringDtype)
    assert isinstance(df["user_id"].dtype,StringDtype)
    assert df["stars"].dtype == np.int64
    assert df["timestamp"].dtype == np.int64

def test_create_interaction_matrix(require_network, dataset_path, dataset_filename):
    assert os.path.exists(f"{dataset_path}/{dataset_filename}.json"), "dataset was not properly fetched in a previous test"
    dataset = YelpOpenDataset(dataset_path, dataset_filename)
    interaction_matrix = dataset.load()

    # assert that an InteractionMatrix object is being created/returned
    assert isinstance(interaction_matrix, InteractionMatrix)

    # assert that the InteractionMatrix is not empty
    assert interaction_matrix.shape[0] > 0
    assert interaction_matrix.shape[1] > 0








