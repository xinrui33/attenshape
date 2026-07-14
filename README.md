# SAX / MINDIST Hierarchical Clustering of Google Trends Search Terms

Clusters ~1203 Google Trends search-volume time series (Jan 2022 – May 2026)
by shape, using Symbolic Aggregate approXimation (SAX) and the SAX MINDIST
distance, agglomerative hierarchical clustering, and a set of validity checks
that justify every parameter choice against evidence rather than convention.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Place input data under `./data/` (see layout below), or point the notebook
at a different location via environment variables:

```bash
export SAX_DATA_DIR=/path/to/data_events
export SAX_OUTPUT_DIR=/path/to/output/run_01
```

All paths default to relative locations under `./data` and `./output` if the
environment variables aren't set, so the notebook runs out of the box for
any collaborator who clones the repo and drops data in the expected place.

### Expected data layout

```
data/
  data_events/
    <term_name>/
      stitched/
        gt_fixed02_*.csv        # columns: date/time/week, and one numeric value column
  shock_detection/
    SP500_data.csv
    DOWJONES_data.csv
    NASDAQ100_data.csv
    RUSSELL2000_data.csv
```

## Running

Open `events_SAX.ipynb` and run top to bottom. Section 0 (Configuration) is
the only cell you should need to edit -- every downstream section reads its
parameters from there.

## Pipeline structure
The pipeline progresses from preprocessing and SAX representation learning to distance construction, hierarchical clustering, validation, robustness analysis, and cluster visualization.

| Section | What it does | Output |
|---|---|---|
| 0 | Configuration -- paths, preprocessing, SAX, clustering, and robustness parameters (all documented inline) | -- |
| 1 | Load, filter, interpolate, denoise, detrend, and robust-normalize each search term | `01_preprocessing/` |
| 2 | SAX representation learning (PAA + Gaussian or empirical breakpoints) | `02_sax/` |
| 3 | Construct the SAX MINDIST distance matrix with Euclidean tie-breaking | `03_distance/` |
| 4 | Final hierarchical clustering: cluster assignments, sizes, and dendrogram | `04_clustering/` |
| 5 | Cluster validation and interpretation: candidate-*k* evaluation, stability, silhouette, representative terms, and cluster summaries | `05_validation/` |
| 6 | Robustness and sensitivity analysis: preprocessing, filtering, representation ablations, and OFAAT parameter perturbations | `06_robustness/` |
| 7 | Visualizations: cluster archetypes, phase portraits, SAX illustrations, and shape dictionary | `07_visualization/` |
