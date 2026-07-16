# Attenshape: Discovering the Shape of Public Attention

Attenshape is a framework for discovering recurring patterns in collective public attention and evaluating whether those patterns contain useful information for financial market volatility.

Rather than grouping search terms by what people search for, Attenshape groups them by how attention evolves over time. Using Google Trends, Symbolic Aggregate approXimation (SAX), and hierarchical clustering, the framework identifies recurring attention trajectories, such as intermittent surges, gradual climbs, and seasonal pulses—and summarizes them as interpretable attention signatures.

## Research workflow

```text
Google Top and Rising Queries
            |
            v
Google Trends collection and stitching
            |
            v
Preprocessing and robust normalization
            |
            v
SAX representation learning
            |
            v
Hierarchical clustering
            |
            v
Robustness test (consensus, stability, silhouette)
            |
            v
Market-volatility forecasting and evaluation
```

The repository contains two connected stages:

1. **Attention-shape discovery:** transforms individual search-volume series into SAX representations, clusters them by shape, and validates the resulting attention signatures.
2. **Downstream prediction:** merges cluster-level attention indices with market data and compares a benchmark volatility model against models augmented with attention information.


## Pipeline structure

The numbered sections below align the code organization with the research narrative in the abstract: broad query discovery, temporal representation learning, attention-signature identification, validation, and downstream forecasting.

| Section | Stage | What it does | Main output |
|---:|---|---|---|
| 0 | Configuration | Defines paths, date coverage, preprocessing settings, SAX parameters, clustering choices, random seeds, and robustness settings | `00_provenance/` |
| 1 | Data loading and preprocessing | Loads the stitched Google Trends files, filters terms, aligns dates, interpolates small gaps, denoises, detrends, and robust-normalizes each series | `01_preprocessing/` |
| 2 | SAX representation learning | Applies PAA and symbolic breakpoints to encode temporal trajectories as SAX features | `02_sax/` |
| 3 | Shape-distance construction | Computes the SAX MINDIST matrix and resolves symbolic ties using the underlying normalized trajectories | `03_distance/` |
| 4 | Hierarchical clustering | Builds the linkage tree, assigns final cluster labels, records cluster sizes, and exports the dendrogram | `04_clustering/` |
| 5 | Validation and interpretation | Evaluates candidate cluster counts, subsample stability, silhouette, representative terms, cluster summaries, and residual structure | `05_validation/` |
| 6 | Robustness analysis | Tests preprocessing choices, filtering rules, SAX specifications, and one-factor-at-a-time parameter perturbations | `06_robustness/` |
| 7 | Attention-signature visualization | Produces cluster archetypes, representative series, SAX illustrations, phase portraits, and the shape dictionary | `07_visualization/` |
| 8 | Volatility forecasting | Compares a market-only forecast with models that add all attention indices or one cluster at a time | `08_prediction/` |

## Evaluation metrics

### Clustering validation

- **Consensus matrix:** visualizes how consistently pairs of terms remain clustered across repeated subsampling.
- **Subsample stability:** measures how reproducible the overall clustering is under repeated resampling.
- **Adjusted Rand Index (ARI):** quantifies agreement between cluster partitions while correcting for chance.
- **Silhouette score:** measures within-cluster cohesion relative to the nearest neighboring cluster.

### Forecasting evaluation

- **Market-only benchmark:** baseline volatility model used for comparison.
- **Cluster-by-cluster comparison:** evaluates each attention signature independently against the market-only benchmark.
- **QLIKE loss:** primary volatility forecasting metric that compares predicted and realized variance; lower values are better.
- **RMSE:** measures the root mean squared forecast error, placing greater weight on large errors; lower values are better.
- **MAE:** measures the average absolute forecast error; lower values are better.
- **Out-of-sample \(R^2\):** measures improvement in squared forecast error relative to the baseline model; positive values indicate better predictive performance.

## Interpretation

Attenshape is designed to discover **how attention evolves**, not merely **what people search for**. A cluster can therefore contain semantically unrelated terms that share a common temporal pattern. The resulting indices represent empirical attention signatures—such as burst-like, gradually increasing, or seasonally recurring behavior—rather than conventional topic categories.
