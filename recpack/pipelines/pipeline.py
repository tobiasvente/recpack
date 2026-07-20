# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from collections import defaultdict
import logging
import os
from typing import Tuple, Union, Dict, List, Any, Optional, Callable

from hyperopt import Trials, fmin, tpe, space_eval, STATUS_OK
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from recpack.algorithms.base import Algorithm, TorchMLAlgorithm
from recpack.matrix import InteractionMatrix
from recpack.pipelines.registries import (
    ALGORITHM_REGISTRY,
    METRIC_REGISTRY,
    AlgorithmEntry,
    MetricEntry,
    OptimisationMetricEntry,
)
from recpack.pipelines.hyperparameter_optimisation import HyperoptInfo, GridSearchInfo
from recpack.postprocessing.postprocessors import Postprocessor
from recpack.util import get_top_K_ranks


logger = logging.getLogger("recpack")


class MetricAccumulator:
    """Accumulates metrics and
    provides methods to aggregate results into usable formats.
    """

    def __init__(self):
        self.acc = defaultdict(dict)

    def __getitem__(self, key):
        return self.acc[key]

    def add(self, metric, algorithm_name, metric_name):
        logger.debug(f"Metric {metric_name} created for algorithm {algorithm_name}")
        self.acc[algorithm_name][metric_name] = metric

    @property
    def metrics(self):
        results = defaultdict(dict)
        for key in self.acc:
            for k in self.acc[key]:
                results[key][k] = self.acc[key][k].value
        return results

    @property
    def num_users(self):
        results = defaultdict(dict)
        for key in self.acc:
            for k in self.acc[key]:
                results[key][k] = self.acc[key][k].num_users
        return results

    @property
    def results(self):
        results = []
        for algorithm_name, algorithm_metrics in self.acc.items():
            for metric_name, metric in algorithm_metrics.items():
                metric_results = metric.results.copy()
                metric_results.insert(0, "metric", metric_name)
                metric_results.insert(0, "algorithm", algorithm_name)
                results.append(metric_results)

        if not results:
            return pd.DataFrame(columns=["algorithm", "metric", "score"])

        return pd.concat(results).reset_index(drop=True)


class RecommendationAccumulator:
    """Accumulates final recommendation scores per algorithm."""

    def __init__(self):
        self.acc = {}

    def add(self, algorithm_name: str, X_pred: csr_matrix) -> None:
        logger.debug(f"Recommendations stored for algorithm {algorithm_name}")
        self.acc[algorithm_name] = X_pred.copy()

    def get_recommendations(self, K: Optional[int] = None) -> pd.DataFrame:
        """Return recommendations in long-form tabular format.

        :param K: The maximum number of recommendations to return per user.
            If ``None``, all nonzero predictions are returned.
        :type K: int, optional
        :return: DataFrame with columns ``algorithm``, ``user_id``,
            ``item_id``, ``rank`` and ``score``.
        :rtype: pd.DataFrame
        """
        if K is not None and K <= 0:
            raise ValueError("K should be a positive integer or None.")

        results = []
        for algorithm_name, X_pred in self.acc.items():
            ranks = get_top_K_ranks(X_pred, K)
            users, items = ranks.nonzero()
            scores = X_pred[users, items].A1
            recommendations = pd.DataFrame(
                {
                    "algorithm": algorithm_name,
                    "user_id": users,
                    "item_id": items,
                    "rank": ranks.data,
                    "score": scores,
                }
            )
            results.append(recommendations)

        if not results:
            return pd.DataFrame(columns=["algorithm", "user_id", "item_id", "rank", "score"])

        return (
            pd.concat(results)
            .sort_values(["algorithm", "user_id", "rank"])
            .reset_index(drop=True)
        )


class Pipeline(object):
    """Performs hyperparameter optimisation, training, prediction and evaluation.

    Pipeline is run per algorithm.
    First, if an `optimisation_metric` is specified, hyperparameters are optimised by training
    on `validation_training_data` and evaluating using `validation_data`.
    Next, unless the model is based on :class:`recpack.algorithms.TorchMLAlgorithm`,
    the model with optimised parameters is retrained on `full_training_data`.
    The final evaluation happens using the `test_data`.

    Results can be accessed via the :meth:`get_metrics` method.

    :param results_directory: Path to a directory in which to save results of the pipeline
        when save_metrics() is called.
    :type results_directory: string
    :param algorithm_entries: List of AlgorithmEntry objects to evaluate in this pipeline.
        An AlgorithmEntry defines which algorithm to train, with which fixed parameters
        (params) and which parameters to optimize (grid).
    :type algorithm_entries: List[AlgorithmEntry]
    :param metric_entries: List of MetricEntry objects to evaluate each algorithm on.
        A MetricEntry defines which metric and value of the parameter K
        (number of recommendations).
    :type metric_entries: List[MetricEntry]
    :param full_training_data: The data to train models on, in the final evaluation.
    :type full_training_data: InteractionMatrix
    :param validation_training_data: The data to train models on when optimising parameters.
    :type validation_training_data: InteractionMatrix
    :param validation_data: The data to use for optimising parameters,
        can be None only if none of the algorithms require optimisation.
    :type validation_data: Union[Tuple[InteractionMatrix, InteractionMatrix], None]
    :param test_data: The data to perform evaluation, as (`test_in`, `test_out`) tuple.
    :type: Tuple[InteractionMatrix, InteractionMatrix]
    :param optimisation_metric: The metric to optimise each algorithm on.
    :type optimisation_metric: Union[OptimisationMetricEntry, None]
    :param post_processor: A postprocessor instance to apply filters
        on the recommendation scores.
    :type post_processor: Postprocessor
    :param remove_history: Boolean to configure if the recommendations can include items that were previously interacted with.
    :type remove_history: Boolean
    """

    def __init__(
        self,
        results_directory: str,
        algorithm_entries: List[AlgorithmEntry],
        metric_entries: List[MetricEntry],
        full_training_data: InteractionMatrix,
        validation_training_data: Union[InteractionMatrix, None],
        validation_data: Union[Tuple[InteractionMatrix, InteractionMatrix], None],
        test_data: Tuple[InteractionMatrix, InteractionMatrix],
        optimisation_metric_entry: Union[OptimisationMetricEntry, None],
        post_processor: Postprocessor,
        remove_history: bool,
    ):
        self.results_directory = results_directory
        self.algorithm_entries = algorithm_entries
        self.metric_entries = metric_entries
        self.full_training_data = full_training_data
        self.validation_training_data = validation_training_data
        self.validation_data = validation_data
        self.test_data_in, self.test_data_out = test_data
        self.optimisation_metric_entry = optimisation_metric_entry
        self.post_processor = post_processor
        self.remove_history = remove_history

        self._metric_acc = MetricAccumulator()
        self._recommendation_acc = RecommendationAccumulator()
        # Hyperparameter optimisation results are accumulated
        self._optimisation_results = []

    def run(self):
        """Runs the pipeline."""
        for algorithm_entry in tqdm(self.algorithm_entries):
            # Check whether we need to optimize hyperparameters
            if algorithm_entry.optimise:
                params = self._optimise_hyperparameters(algorithm_entry)
            else:
                params = algorithm_entry.params

            algorithm = ALGORITHM_REGISTRY.get(algorithm_entry.name)(**params)
            # Train the final version of the algorithm
            if isinstance(algorithm, TorchMLAlgorithm):
                # TODO In theory, if early stopping is not used, this could be the full training dataset.
                self._train(algorithm, self.validation_training_data)
            else:
                self._train(algorithm, self.full_training_data)
            # Make predictions
            X_pred = self._predict_and_postprocess(algorithm, self.test_data_in)
            self._recommendation_acc.add(algorithm.identifier, X_pred)

            for metric_entry in self.metric_entries:
                metric_cls = METRIC_REGISTRY.get(metric_entry.name)
                if metric_entry.K is not None:
                    metric = metric_cls(K=metric_entry.K)
                else:
                    metric = metric_cls()
                metric.calculate(self.test_data_out.binary_values, X_pred)
                self._metric_acc.add(metric, algorithm.identifier, metric.name)

    def _train(self, algorithm: Algorithm, training_data: InteractionMatrix) -> Algorithm:
        if isinstance(algorithm, TorchMLAlgorithm):
            algorithm.fit(training_data, self.validation_data)
        else:
            algorithm.fit(training_data)
        return algorithm

    def _predict_and_postprocess(self, algorithm: Algorithm, data_in: InteractionMatrix) -> csr_matrix:
        X_pred = algorithm.predict(data_in)

        # QUESTION: This removes only the test_data_in/validation_data_in, I think in general more is removed. Was this intentional?
        if self.remove_history:
            X_pred = X_pred - X_pred.multiply(data_in.binary_values)

        X_pred = self.post_processor.process(X_pred)

        return X_pred

    def _optimise_hyperparameters(self, algorithm_entry: AlgorithmEntry) -> Dict[str, Any]:
        def optimise(args: Dict[str, Any]) -> Dict[str, Any]:
            algorithm = ALGORITHM_REGISTRY.get(algorithm_entry.name)(**args, **algorithm_entry.params)
            # Train with given hyperparameter instantiations
            self._train(algorithm, self.validation_training_data)
            # Make predictions and postprocess
            validation_data_in, validation_data_out = self.validation_data
            X_pred_val = self._predict_and_postprocess(algorithm, validation_data_in)
            # Compute optimisation metric value
            optimisation_metric = METRIC_REGISTRY.get(self.optimisation_metric_entry.name)(
                K=self.optimisation_metric_entry.K
            )
            optimisation_metric.calculate(validation_data_out.binary_values, X_pred_val)

            # We need to return STATUS_OK for hyperopt to work

            result = {
                "loss": optimisation_metric.value,
                "status": STATUS_OK,
                "algorithm": algorithm_entry.name,
                "identifier": algorithm.identifier,
                "params": {**args, **algorithm_entry.params},
                optimisation_metric.name: optimisation_metric.value,
            }

            # Hyperopt always minimises, so to maximize a metric we just turn it negative.
            if not self.optimisation_metric_entry.minimise:
                result["loss"] *= -1
            return result

        if isinstance(algorithm_entry.optimisation_info, HyperoptInfo):
            results = self._optimise_w_hyperopt(optimise, algorithm_entry.optimisation_info)
        else:
            results = self._optimise_w_grid(optimise, algorithm_entry.optimisation_info)

        # Sort by metric value
        optimal_params = sorted(results, key=lambda x: x["loss"], reverse=False)[0]["params"]

        self._optimisation_results.append(pd.DataFrame.from_records(results).drop(columns=["loss", "status"]))

        return optimal_params

    def _optimise_w_grid(self, objective: Callable, optimisation_info: GridSearchInfo) -> List:
        results = []
        for p in optimisation_info.grid:
            res = objective(p)
            results.append(res)

        return results

    def _optimise_w_hyperopt(self, objective: Callable, optimisation_info: HyperoptInfo) -> List:
        tpe_trials = Trials()

        best = fmin(
            objective,
            optimisation_info.space,
            max_evals=optimisation_info.max_evals,
            timeout=optimisation_info.timeout,
            algo=tpe.suggest,
            trials=tpe_trials,
        )

        return tpe_trials.results

    def get_metrics(self, short: Optional[bool] = False) -> pd.DataFrame:
        """Get the metrics for the pipeline.

        :param short: If short is True, only the algorithm names are returned, and not the parameters.
            Defaults to False
        :type short: bool, optional
        :return: Algorithms and their respective performance.
        :rtype: pd.DataFrame
        """
        df = pd.DataFrame.from_dict(self._metric_acc.metrics).T
        if short:
            # Parameters are between (), so if we split on the (,
            # we can get the algorithm name by taking the first of the splits.
            df.index = df.index.map(lambda x: x.split("(")[0])
        return df

    def save_metrics(self) -> None:
        """Save the metrics in a json file

        The file will be saved in the experiment directory.
        """
        if not os.path.exists(self.results_directory):
            os.mkdir(self.results_directory)

        df = self.get_metrics()
        df.to_json(f"{self.results_directory}/results.json")

        try:
            self.optimisation_results.to_json(f"{self.results_directory}/optimisation_results.json")
        except AttributeError:
            pass

    def get_metric_results(self) -> pd.DataFrame:
        return self._metric_acc.results

    def _ensure_visualization_output_directory(self, output_directory: Optional[str] = None) -> str:
        output_directory = output_directory or self.results_directory
        os.makedirs(output_directory, exist_ok=True)
        return output_directory

    def _save_visualization_axis(self, ax, output_path: str) -> None:
        from matplotlib import pyplot as plt

        ax.figure.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(ax.figure)

    def get_recommendations(self, K: Optional[int] = None) -> pd.DataFrame:
        if K is None:
            K_values = [metric_entry.K for metric_entry in self.metric_entries if metric_entry.K is not None]
            K = max(K_values) if K_values else 10

        return self._recommendation_acc.get_recommendations(K)

    def save_visualization_data(self, K: Optional[int] = None) -> None:
        if not os.path.exists(self.results_directory):
            os.mkdir(self.results_directory)

        self.get_metric_results().to_csv(f"{self.results_directory}/metric_details.csv", index=False)
        self.get_recommendations(K).to_csv(f"{self.results_directory}/recommendations.csv", index=False)

    def save_metric_comparison_plot(
        self,
        output_directory: Optional[str] = None,
        short_names: bool = True,
        file_format: str = "png",
    ) -> str:
        from recpack.visualization import plot_metric_comparison

        output_directory = self._ensure_visualization_output_directory(output_directory)
        output_path = os.path.join(output_directory, f"metric_comparison.{file_format}")
        ax = plot_metric_comparison(self.get_metrics(short=short_names), figsize=(8, 4))
        self._save_visualization_axis(ax, output_path)
        return output_path

    def save_metric_distribution_plot(
        self,
        output_directory: Optional[str] = None,
        metric: Optional[str] = None,
        short_names: bool = True,
        file_format: str = "png",
    ) -> str:
        """Save a plot with detailed score distribution for one metric."""
        from recpack.visualization import plot_metric_distribution

        output_directory = self._ensure_visualization_output_directory(output_directory)
        output_path = os.path.join(output_directory, f"metric_distribution.{file_format}")

        if metric is None:
            metric_entry = self.metric_entries[-1]
            metric = f"{metric_entry.name}_{metric_entry.K}" if metric_entry.K is not None else metric_entry.name

        metric_results = self.get_metric_results()
        if short_names:
            metric_results = metric_results.copy()
            metric_results["algorithm"] = metric_results["algorithm"].map(lambda x: x.split("(")[0])

        ax = plot_metric_distribution(metric_results, metric=metric, figsize=(6, 4))
        self._save_visualization_axis(ax, output_path)
        return output_path

    def save_recommendation_popularity_plot(
        self,
        output_directory: Optional[str] = None,
        K: Optional[int] = None,
        top_n: int = 20,
        file_format: str = "png",
    ) -> str:
        """Save a plot showing the most frequently recommended items."""
        from recpack.visualization import plot_recommendation_popularity

        output_directory = self._ensure_visualization_output_directory(output_directory)
        output_path = os.path.join(output_directory, f"recommendation_popularity.{file_format}")
        ax = plot_recommendation_popularity(self.get_recommendations(K), top_n=top_n, figsize=(6, 4))
        self._save_visualization_axis(ax, output_path)
        return output_path



    def get_num_users(self) -> int:
        """Get the amount of users used in the evaluation.

        :return: The number of users used in the evaluation.
        :rtype: int
        """
        return self._metric_acc.num_users

    @property
    def optimisation_results(self):
        """Contains a result for each of the hyperparameter combinations tried out,
        for each of the algorithms evaluated.
        """
        if not self._optimisation_results:
            raise AttributeError("No hyperparameter optimisation was performed.")
        return pd.concat(self._optimisation_results).reset_index(drop=True)
