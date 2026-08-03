# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import json
import logging
import os
import time
from unittest.mock import MagicMock, patch, call

from hyperopt import hp
import numpy as np
import pytest

from hyperopt import hp
import numpy as np
import pytest

from recpack.postprocessing.filters import PostFilter
from recpack.pipelines import GridSearchInfo, HyperoptInfo
from recpack.pipelines.pipeline_builder import AlgorithmEntry


class MockFilter(PostFilter):
    def __init__(self):
        self.apply_calls = 0

    def apply(self, X):
        self.apply_calls += 1
        return X


def test_pipeline(pipeline_builder):
    pipeline = pipeline_builder.build()

    pipeline.run()

    metrics = pipeline.get_metrics()
    assert len(metrics) == len(pipeline.algorithm_entries)

    assert len(metrics[list(metrics.keys())[0]]) == len(pipeline.metric_entries)


def test_pipeline_save_metrics(pipeline_builder):
    pipeline = pipeline_builder.build()

    pipeline.run()

    mocker = MagicMock()
    with patch("recpack.pipelines.pipeline.pd.DataFrame.to_json", mocker):
        pipeline.save_metrics()

        mocker.assert_called_once_with(f"{pipeline_builder.results_directory}/results.json")


def test_pipeline_save_metrics_w_optimisation(pipeline_builder_optimisation):
    pipeline = pipeline_builder_optimisation.build()

    pipeline.run()

    mocker = MagicMock()
    with patch("recpack.pipelines.pipeline.pd.DataFrame.to_json", mocker):
        pipeline.save_metrics()

        mocker.assert_has_calls(
            [
                call(f"{pipeline_builder_optimisation.results_directory}/results.json"),
                call(f"{pipeline_builder_optimisation.results_directory}/optimisation_results.json"),
            ]
        )


@pytest.mark.parametrize(
    "algorithm_name, optimisation_info, gridsize",
    [("ItemKNN", GridSearchInfo({"K": [1, 2, 3]}), 3), ("EASE", GridSearchInfo({"l2": [2, 10]}), 2)],
)
def test_pipeline_optimisation_gridsearch(
    pipeline_builder_optimisation_no_algos, algorithm_name, optimisation_info, gridsize
):
    pipeline_builder_optimisation_no_algos.add_algorithm(algorithm_name, optimisation_info=optimisation_info)
    pipeline = pipeline_builder_optimisation_no_algos.build()

    pipeline.run()

    metrics = pipeline.get_metrics()

    assert metrics.shape[0] == len(pipeline.algorithm_entries)
    assert metrics.shape[1] == len(pipeline.metric_entries)
    assert pipeline.optimisation_results.shape[0] == gridsize


def test_pipeline_optimisation_results_default_single_metric(pipeline_builder_optimisation):
    """By default only the optimisation metric is recorded for validation runs."""
    pipeline = pipeline_builder_optimisation.build()
    pipeline.run()

    opt_results = pipeline.optimisation_results
    # Optimisation metric is CalibratedRecallK_2; CalibratedRecallK_3 is also configured
    # but should NOT appear in the optimisation results when the flag is off.
    assert "CalibratedRecallK_2" in opt_results.columns
    assert "CalibratedRecallK_3" not in opt_results.columns


def test_pipeline_optimisation_results_all_metrics(pipeline_builder_optimisation):
    """When optimisation_all_metrics is enabled, every configured metric is
    calculated and stored for each validation run."""
    pipeline_builder_optimisation.set_optimisation_all_metrics(True)
    pipeline = pipeline_builder_optimisation.build()
    pipeline.run()

    opt_results = pipeline.optimisation_results
    assert "CalibratedRecallK_2" in opt_results.columns
    assert "CalibratedRecallK_3" in opt_results.columns
    # All values should be populated (no NaN)
    assert opt_results["CalibratedRecallK_2"].notna().all()
    assert opt_results["CalibratedRecallK_3"].notna().all()
    # Best-parameter selection still uses only the optimisation metric, so the
    # final test-set evaluation should still produce a regular metrics table.
    metrics = pipeline.get_metrics()
    assert metrics.shape[0] == len(pipeline.algorithm_entries)


def test_pipeline_with_filters_applied(pipeline_builder):

    mock_filter = MockFilter()

    pipeline_builder.add_post_filter(mock_filter)
    pipe = pipeline_builder.build()
    pipe.run()
    assert mock_filter.apply_calls == len(pipe.algorithm_entries)


def test_pipeline_with_filters_applied_optimisation(pipeline_builder_optimisation_no_algos):
    pb = pipeline_builder_optimisation_no_algos
    pb.add_algorithm("ItemKNN", optimisation_info=GridSearchInfo({"K": [1, 2, 3]}))
    pb.add_algorithm("EASE", optimisation_info=GridSearchInfo({"l2": [2, 10]}))

    mock_filter = MockFilter()

    pb.add_post_filter(mock_filter)
    pipe = pb.build()
    pipe.run()
    assert mock_filter.apply_calls == 7  # 3 ItemKNN, 2 EASE, 2 final eval


@pytest.mark.parametrize(
    "info",
    [
        HyperoptInfo(space={"K": hp.uniformint("K", 50, 1000)}, max_evals=10),
        HyperoptInfo(
            space={
                "similarity": hp.choice(
                    "similarity",
                    [
                        "conditional_probability",
                        "cosine",
                    ],
                ),
                "pop_discount": hp.uniform("pop_discount", 0, 1),
                "K": hp.uniformint("K", 50, 1000),
            },
            max_evals=10,
        ),
    ],
)
def test_pipeline_with_hyperopt_max_evals(pipeline_builder_optimisation_no_algos, info):
    pb = pipeline_builder_optimisation_no_algos

    pb.add_algorithm("ItemKNN", optimisation_info=info)

    pipe = pb.build()
    pipe.run()
    assert pipe.optimisation_results.shape[0] == 10


@pytest.mark.parametrize(
    "info",
    [
        HyperoptInfo(space={"K": hp.uniformint("K", 50, 1000)}, timeout=2),
    ],
)
def test_pipeline_with_hyperopt_timeout(pipeline_builder_optimisation_no_algos, info):
    pb = pipeline_builder_optimisation_no_algos

    pb.add_algorithm("ItemKNN", optimisation_info=info)

    pipe = pb.build()
    start = time.time()
    pipe.run()
    end = time.time()
    # Because of the small matrix, there should be only very little overhead.
    np.testing.assert_almost_equal(end - start, info.timeout, 0)


def test_pipeline_with_both_hyperopt_and_grid(pipeline_builder_optimisation_no_algos):
    pb = pipeline_builder_optimisation_no_algos

    pb.add_algorithm(
        "ItemKNN", optimisation_info=HyperoptInfo(space={"K": hp.uniformint("K", 50, 1000)}, max_evals=10)
    )
    pb.add_algorithm("EASE", optimisation_info=GridSearchInfo({"l2": [1, 10, 100, 1000]}))

    pipe = pb.build()
    pipe.run()
    assert pipe.optimisation_results.shape[0] == 10 + 4


def test_pipeline_optimisation_results_output(pipeline_builder_optimisation_no_algos):
    pb = pipeline_builder_optimisation_no_algos

    pb.add_algorithm(
        "ItemKNN",
        optimisation_info=HyperoptInfo(space={"K": hp.uniformint("K", 50, 1000)}, max_evals=10),
        params={"similarity": "conditional_probability"},
    )
    pb.add_algorithm("EASE", optimisation_info=GridSearchInfo({"l2": [1, 10, 100, 1000]}))

    pipe = pb.build()
    pipe.run()
    assert pipe.optimisation_results.shape[1] == 4

    assert "algorithm" in pipe.optimisation_results.columns
    assert "identifier" in pipe.optimisation_results.columns
    assert "params" in pipe.optimisation_results.columns
    assert (
        pipe.optimisation_metric_entry.name + "_" + str(pipe.optimisation_metric_entry.K)
        in pipe.optimisation_results.columns
    )

    print(pipe.optimisation_results)

    print(pipe.optimisation_results["params"][0])

    # ItemKNN optimal hyperparameter asserts
    assert (
        "similarity" in pipe.optimisation_results["params"][0]
        and pipe.optimisation_results["params"][0]["similarity"] == "conditional_probability"
    )
    assert "K" in pipe.optimisation_results["params"][0]

    # EASE optimial hyperparameter asserts
    assert "l2" in pipe.optimisation_results["params"].iloc[-1]


def test_pipeline_logs_validation_metric(pipeline_builder_optimisation, caplog):
    """The validation metric should be logged for every trial so progress is
    visible even if the pipeline is terminated before it completes."""
    pipeline = pipeline_builder_optimisation.build()
    with caplog.at_level(logging.INFO, logger="recpack"):
        pipeline.run()

    validation_log_messages = [
        rec.message for rec in caplog.records if "Validation" in rec.message
    ]
    # 3 ItemKNN trials + 2 EASE trials = 5 validation log lines
    assert len(validation_log_messages) == 5
    for msg in validation_log_messages:
        assert "CalibratedRecallK_2" in msg


def test_pipeline_logs_test_metric(pipeline_builder, caplog):
    """Final test metrics should also be logged so they appear in the run log."""
    pipeline = pipeline_builder.build()
    with caplog.at_level(logging.INFO, logger="recpack"):
        pipeline.run()

    test_log_messages = [
        rec.message for rec in caplog.records if "Test metric" in rec.message
    ]
    # 2 algorithms x 2 metrics = 4 test metric log lines
    assert len(test_log_messages) == 4


def test_pipeline_incremental_save_writes_test_metrics(pipeline_builder, tmp_path):
    """When incremental_save is enabled, each test metric should be appended
    to results.jsonl as soon as it is computed."""
    pipeline_builder.base_path = str(tmp_path)
    pipeline_builder.results_directory = f"{pipeline_builder.base_path}/{pipeline_builder.folder_name}"
    pipeline_builder.set_incremental_save(True)

    pipeline = pipeline_builder.build()
    pipeline.run()

    results_file = os.path.join(pipeline.results_directory, "results.jsonl")
    assert os.path.exists(results_file)

    with open(results_file, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]

    # 2 algorithms x 2 metrics = 4 records
    assert len(lines) == 4
    for record in lines:
        assert "algorithm" in record
        assert "identifier" in record
        assert "metric" in record
        assert "value" in record


def test_pipeline_incremental_save_writes_optimisation_results(pipeline_builder_optimisation, tmp_path):
    """When incremental_save is enabled, each validation trial should be
    appended to optimisation_results.jsonl as it runs."""
    pipeline_builder_optimisation.base_path = str(tmp_path)
    pipeline_builder_optimisation.results_directory = (
        f"{pipeline_builder_optimisation.base_path}/{pipeline_builder_optimisation.folder_name}"
    )
    pipeline_builder_optimisation.set_incremental_save(True)

    pipeline = pipeline_builder_optimisation.build()
    pipeline.run()

    opt_file = os.path.join(pipeline.results_directory, "optimisation_results.jsonl")
    assert os.path.exists(opt_file)

    with open(opt_file, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]

    # 3 ItemKNN trials + 2 EASE trials = 5 records
    assert len(lines) == 5
    for record in lines:
        assert "algorithm" in record
        assert "identifier" in record
        assert "params" in record
        assert "loss" in record


def test_pipeline_incremental_save_disabled_writes_nothing(pipeline_builder, tmp_path):
    """When incremental_save is disabled (default), no jsonl files should be created."""
    pipeline_builder.base_path = str(tmp_path)
    pipeline_builder.results_directory = f"{pipeline_builder.base_path}/{pipeline_builder.folder_name}"

    pipeline = pipeline_builder.build()
    pipeline.run()

    # Files should not exist; results_directory may not even have been created.
    if os.path.isdir(pipeline.results_directory):
        assert not os.path.exists(os.path.join(pipeline.results_directory, "results.jsonl"))
        assert not os.path.exists(os.path.join(pipeline.results_directory, "optimisation_results.jsonl"))
