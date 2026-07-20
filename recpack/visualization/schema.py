# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE

RESULTS_FILENAME = "results.json"
OPTIMISATION_RESULTS_FILENAME = "optimisation_results.json"
METRIC_DETAILS_FILENAME = "metric_details.csv"
RECOMMENDATIONS_FILENAME = "recommendations.csv"


def output_schema():
    return {
        RESULTS_FILENAME: {
            "writer": "Pipeline.save_metrics()",
            "format": "json",
            "description": "Aggregated metric values with algorithms as rows and metrics as columns.",
        },
        OPTIMISATION_RESULTS_FILENAME: {
            "writer": "Pipeline.save_metrics()",
            "format": "json",
            "description": "Hyperparameter optimisation trials, when optimisation was performed.",
        },
        METRIC_DETAILS_FILENAME: {
            "writer": "Pipeline.save_visualization_data()",
            "format": "csv",
            "columns": ["algorithm", "metric", "user_id", "item_id", "score"],
            "description": "Long-form detailed metric scores. Some metric types omit user_id or item_id.",
        },
        RECOMMENDATIONS_FILENAME: {
            "writer": "Pipeline.save_visualization_data()",
            "format": "csv",
            "columns": ["algorithm", "user_id", "item_id", "rank", "score"],
            "description": "Top-K recommendation rows after history removal and postprocessing.",
        },
    }
