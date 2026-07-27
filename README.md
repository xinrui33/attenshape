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


## Project Architecture & Pipeline Stages

The pipeline is organized into modular sequential stages. The active subdirectories currently configured are:

| Section | Stage | What it does | Main output |
|---:|---|---|---|
| 0 | Configuration | Defines paths, date coverage, preprocessing settings, SAX parameters, clustering choices, random seeds, and robustness settings | `00_provenance/` |
| 1 | Data loading and preprocessing | Loads the stitched Google Trends files, filters terms, aligns dates, interpolates small gaps, denoises, detrends, and robust-normalizes each series | `01_preprocessing/` |
| 2 | SAX representation learning | Applies PAA and symbolic breakpoints to encode temporal trajectories as SAX features | `02_sax/` |
| 3 | Shape-distance construction | Computes the SAX MINDIST matrix and resolves symbolic ties using the underlying normalized trajectories | `03_distance/` |
| 4 | Hierarchical clustering | Builds the linkage tree, assigns final cluster labels, records cluster sizes, and exports the dendrogram | `04_clustering/` |
| 5 | Validation and interpretation | Evaluates candidate cluster counts, subsample stability, silhouette, representative terms, cluster summaries, and residual structure | `05_validation/` |
| 6 | Robustness analysis | Tests preprocessing choices, filtering rules, and SAX specifications | `06_robustness/` |
| 7 | Attention-signature visualization | Produces figures from the paper | `07_visualization/` |

## Interpretation

Attenshape is designed to discover **how attention evolves**, not merely **what people search for**. A cluster can therefore contain semantically unrelated terms that share a common temporal pattern. The resulting indices represent empirical attention signatures—such as burst-like, gradually increasing, or seasonally recurring behavior—rather than conventional topic categories.
