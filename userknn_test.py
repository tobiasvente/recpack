from recpack.datasets import MovieLens1M
from recpack.scenarios import StrongGeneralization
from recpack.pipelines import PipelineBuilder
import numpy as np

seed = 42
np.random.seed(seed)

# Dataset
dataset = MovieLens1M(path='dataset/ml-1m/')
dataset.fetch_dataset()
interaction_matrix = dataset.load()

# Scenario
scenario = StrongGeneralization(validation=True, seed=42)
scenario.split(interaction_matrix)

# Pipeline
builder = PipelineBuilder()
builder.set_data_from_scenario(scenario)

# Algorithm (UserKNN)
builder.add_algorithm('UserKNN', grid={
    'K': [50, 100, 250, 500],
    'similarity': ['cosine']  # , 'pearson_similarity'],
})
# Optimisation metric
builder.set_optimisation_metric('NDCGK', K=10)
# Metrics
builder.add_metric('NDCGK', K=10)

pipeline = builder.build()
pipeline.run()

print(pipeline.get_metrics())
