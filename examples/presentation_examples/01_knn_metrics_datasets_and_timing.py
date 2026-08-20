"""Show the new similarities on ItemKNN and UserKNN, including timings."""

from pathlib import Path

from recpack.datasets import AmazonDataset, MovieLens100K, YelpOpenDataset
from recpack.pipelines import PipelineBuilder
from recpack.scenarios import WeakGeneralization

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SIMILARITIES = ("cosine", "jaccard", "dice", "overlap", "lift", "pmi")


def main():
    dataset = MovieLens100K(path=str(DATA_DIR))
    # dataset = AmazonDataset(AmazonDataset.Category.DIGITAL_MUSIC, path=str(DATA_DIR))
    # dataset = YelpOpenDataset(path=str(DATA_DIR))

    interactions = dataset.load()

    scenario = WeakGeneralization(frac_data_in=0.8, seed=42)
    scenario.split(interactions)

    builder = PipelineBuilder(
        folder_name="presentation_knn",
        base_path=str(Path(__file__).parent),
    )
    builder.set_data_from_scenario(scenario)
    builder.add_metric("NDCGK", 10)

    for similarity in SIMILARITIES:
        builder.add_algorithm("ItemKNN", params={"K": 200, "similarity": similarity})
        builder.add_algorithm("UserKNN", params={"K": 200, "similarity": similarity})

    pipeline = builder.build()
    pipeline.run()

    results = pipeline.get_metrics()
    results.index.name = "algorithm"

    print(f"\n{type(dataset).__name__}: ItemKNN and UserKNN results")
    print(results[["NDCGK_10", "fitting_time", "inference_time"]].to_string())


if __name__ == "__main__":
    main()
