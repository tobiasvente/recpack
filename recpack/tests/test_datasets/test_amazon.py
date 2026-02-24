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

from recpack.datasets.amazon import AmazonDataset
from recpack.matrix import InteractionMatrix


@pytest.fixture
def require_network():
    "tests if there is a network connection"
    try:
        with socket.create_connection(("8.8.8.8", 53), timeout=2):
            pass
    except OSError:
        pytest.skip("No network available")

@pytest.fixture(scope="session")
def dataset_path():
    path = "recpack/tests/test_datasets/downloaded_during_test"
    yield path

    shutil.rmtree(path, ignore_errors=True) # cleanup downloaded datasets after running all tests

@pytest.fixture
def dataset_filename():
    filename = "AmazonDataset_"
    return filename


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize("category", AmazonDataset.Category, ids=lambda x: x.value)
def test_fetch_dataset(require_network,  category, dataset_path, dataset_filename):
    dataset_filename = dataset_filename + category.value
    dataset = AmazonDataset(category, dataset_path, dataset_filename)
    dataset._download_dataset()
    assert os.path.exists(f"{dataset_path}/{dataset_filename}") # check if the dataset csv file was properly downloaded and extracted


@pytest.mark.network
@pytest.mark.parametrize("category", AmazonDataset.Category, ids=lambda x: x.value)
def test_create_dataframe(require_network, category, dataset_path, dataset_filename):
    dataset_filename = dataset_filename + category.value
    assert os.path.exists(f"{dataset_path}/{dataset_filename}"), "dataset was not properly fetched in a previous test"
    dataset = AmazonDataset(category, dataset_path, dataset_filename)
    df = dataset._load_dataframe()
    # assert that a dataframe object is being created/returned
    assert isinstance(df, pd.DataFrame)

    # assert that the dataframe is not empty
    assert not df.empty

    #assert that the columns match the expected column names
    assert set(df.columns) == {dataset.USER_IX, dataset.ITEM_IX, dataset.RATING_IX, dataset.TIMESTAMP_IX}

    #assert that no null values are present in any of the columns
    assert df[dataset.USER_IX].isnull().sum() == 0
    assert df[dataset.ITEM_IX].isnull().sum() == 0
    assert df[dataset.RATING_IX].isnull().sum() == 0
    assert df[dataset.TIMESTAMP_IX].isnull().sum() == 0

    # #assert that the types for the columns match the expected types
    assert isinstance(df[dataset.USER_IX].dtype,StringDtype)
    assert isinstance(df[dataset.ITEM_IX].dtype,StringDtype)
    assert df[dataset.RATING_IX].dtype == np.int64
    assert df[dataset.TIMESTAMP_IX].dtype == np.float64

@pytest.mark.network
@pytest.mark.parametrize("category", AmazonDataset.Category, ids=lambda x: x.value)
def test_create_interaction_matrix(require_network, category, dataset_path, dataset_filename):
    dataset_filename = dataset_filename + category.value
    assert os.path.exists(f"{dataset_path}/{dataset_filename}"), "dataset was not properly fetched in a previous test"
    dataset = AmazonDataset(category, dataset_path, dataset_filename)
    interaction_matrix = dataset.load()

    # assert that an InteractionMatrix object is being created/returned
    assert isinstance(interaction_matrix, InteractionMatrix)

    # assert that the InteractionMatrix is not empty
    assert interaction_matrix.shape[0] > 0
    assert interaction_matrix.shape[1] > 0



