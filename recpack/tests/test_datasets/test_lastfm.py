# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import os
import shutil
import socket

import numpy as np
import pandas as pd
import pytest

from recpack.datasets import LastFM
from recpack.matrix import InteractionMatrix
from recpack.preprocessing.filters import NMostPopular


@pytest.fixture
def require_network():
    "tests if there is a network connection"
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)
    except OSError:
        pytest.skip("No network available")


@pytest.fixture(scope="session")
def download_path():
    path = "recpack/tests/test_datasets/downloaded_during_test"
    yield path
    shutil.rmtree(path, ignore_errors=True)  # cleanup downloaded datasets after running all tests


@pytest.fixture
def dataset_filename():
    return "user_artists.dat"


@pytest.mark.network
@pytest.mark.slow
def test_fetch_dataset(require_network, download_path, dataset_filename):
    dataset = LastFM(download_path, dataset_filename)
    dataset._download_dataset()
    assert os.path.exists(f"{download_path}/{dataset_filename}")


@pytest.mark.network
def test_create_dataframe(require_network, download_path, dataset_filename):
    dataset = LastFM(download_path, dataset_filename)
    df = dataset._load_dataframe()

    # assert that a dataframe object is being created/returned
    assert isinstance(df, pd.DataFrame)

    # assert that the dataframe is not empty
    assert not df.empty

    # assert that the columns match the expected column names
    assert set(df.columns) == {dataset.USER_IX, dataset.ITEM_IX, dataset.WEIGHT_IX}

    # assert that no null values are present in any of the columns
    assert df[dataset.USER_IX].isnull().sum() == 0
    assert df[dataset.ITEM_IX].isnull().sum() == 0
    assert df[dataset.WEIGHT_IX].isnull().sum() == 0

    # assert that the types for the columns match the expected types
    assert df[dataset.USER_IX].dtype == np.int64
    assert df[dataset.ITEM_IX].dtype == np.int64
    assert df[dataset.WEIGHT_IX].dtype == np.int64


@pytest.mark.network
def test_create_interaction_matrix(require_network, download_path, dataset_filename):
    dataset = LastFM(download_path, dataset_filename)
    interaction_matrix = dataset.load()

    # assert that an InteractionMatrix object is being created/returned
    assert isinstance(interaction_matrix, InteractionMatrix)

    # assert that the InteractionMatrix is not empty
    assert interaction_matrix.shape[0] > 0
    assert interaction_matrix.shape[1] > 0


# Fast, network-independent tests using a small local sample (via the
# shared `dataset_path` fixture from conftest.py), following the same
# pattern as the other pre-existing per-dataset filter tests.
def test_add_filter(dataset_path):
    d = LastFM(path=dataset_path, filename="lastfm_sample.dat")

    d.add_filter(NMostPopular(3, d.ITEM_IX))

    data = d.load()

    assert data.shape[1] == 3


def test_add_filter_w_index(dataset_path):
    d = LastFM(path=dataset_path, filename="lastfm_sample.dat")

    d.add_filter(NMostPopular(3, d.ITEM_IX), index=0)

    assert type(d.preprocessor.filters[0]) == NMostPopular
    assert len(d.preprocessor.filters) == 3
