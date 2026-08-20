"""Run the new BPRKNN implementations through a lightweight CPU pipeline."""

from pathlib import Path

import torch

from recpack.pipelines import PipelineBuilder
from recpack.datasets import MovieLens100K
from recpack.scenarios import WeakGeneralization

def result_label(identifier: str) -> str:
    if "similarity_mode=direct" in identifier:
        return "BPRKNN (direct)"
    if "similarity_mode=factorized" in identifier:
        return "BPRKNN (factorized)"
    return identifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def main():
    dataset = MovieLens100K(path=str(DATA_DIR))
    interactions = dataset.load()

    scenario = WeakGeneralization(frac_data_in=0.8, seed=42, validation=True)
    scenario.split(interactions)

    builder = PipelineBuilder(
        folder_name="presentation_bpr",
        base_path=str(Path(__file__).parent),
    )
    builder.set_data_from_scenario(scenario)
    builder.add_metric("NDCGK", 10)

    shared_params = {
        "max_epochs": 5,
        "batch_size": 256,
        "sample_size": 1000,
        "keep_last": True,
        "seed": 42,
    }
    builder.add_algorithm(
        "BPRKNN",
        params={"K": 100, "similarity_mode": "direct", **shared_params},
    )
    builder.add_algorithm(
        "BPRKNN",
        params={
            "K": None,
            "similarity_mode": "factorized",
            "num_components": 16,
            **shared_params,
        },
    )

    pipeline = builder.build()
    pipeline.run()

    results = pipeline.get_metrics()
    results.index = results.index.map(result_label)
    results.index.name = "algorithm"

    print("\nBPR recommendation quality and timing")
    print(results[["NDCGK_10", "fitting_time", "inference_time"]].to_string())


if __name__ == "__main__":
    main()
