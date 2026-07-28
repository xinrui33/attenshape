# %%
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from arch import arch_model

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.style.use("ggplot")

# %%
# =============================================================================
# Repository paths
# =============================================================================

ROOT = Path("../..")

OUTPUT_DIR = ROOT / "output"

PREPROCESS_DIR = OUTPUT_DIR / "01_preprocessing"
DISTANCE_DIR   = OUTPUT_DIR / "03_distance"
CLUSTER_DIR    = OUTPUT_DIR / "04_clustering"

# Market prediction data
DATA_DIR = ROOT / "data" / "application_market_prediction"

# %%
# =============================================================================
# Prepare panel dataframe
# =============================================================================

# Rename first column to "date"
panel = panel.rename(columns={panel.columns[0]: "date"}).copy()

# Convert to datetime
panel["date"] = pd.to_datetime(panel["date"])

print(panel.columns[:10].tolist())

# %%
# =============================================================================
# Load attention datasets
# =============================================================================

panel = pd.read_csv(
    PREPROCESS_DIR / "panel_normalized.csv"
)

clusters = pd.read_csv(
    CLUSTER_DIR / "final_cluster_assignments.csv"
)

# -------------------------------------------------------------------------
# Clean cluster assignments
# -------------------------------------------------------------------------

# Remove rows with missing search terms
clusters = clusters.dropna(subset=["term"]).copy()

# Standardise formatting
clusters["term"] = clusters["term"].astype(str).str.strip()
clusters["cluster"] = clusters["cluster"].astype(int)

mindist = pd.read_csv(
    DISTANCE_DIR / "sax_mindist_matrix_tiebroken.csv",
    index_col=0
)

print("Panel shape:      ", panel.shape)
print("Clusters shape:   ", clusters.shape)
print("MINDIST shape:    ", mindist.shape)

# %%
# =============================================================================
# Load Yahoo Finance csv
# =============================================================================

def load_market_csv(path):

    df = pd.read_csv(path)

    # Remove the ticker row
    df = df.iloc[2:].reset_index(drop=True)

    # Rename first column
    df.rename(columns={df.columns[0]: "date"}, inplace=True)

    # Convert types
    df["date"] = pd.to_datetime(df["date"])

    numeric_cols = [c for c in df.columns if c != "date"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# %%
# =============================================================================
# Load market data
# =============================================================================

indices = {
    "S&P500": load_market_csv(DATA_DIR / "SP500 (1).csv"),
    "NASDAQ100": load_market_csv(DATA_DIR / "NASDAQ100.csv"),
    "DowJones": load_market_csv(DATA_DIR / "DOW.csv"),
    "Russell2000": load_market_csv(DATA_DIR / "RUSSELL2000.csv"),
}

for name, df in indices.items():
    print(name)
    print(df.head())
    print()

# %%
# =============================================================================
# Verify consistency
# =============================================================================

panel_terms = set(panel.columns[1:])
cluster_terms = set(clusters["term"])
mindist_terms = set(mindist.index)

print(f"Terms in panel:      {len(panel_terms)}")
print(f"Terms in clusters:   {len(cluster_terms)}")
print(f"Terms in MINDIST:    {len(mindist_terms)}")

print()

print("Panel == Clusters :", panel_terms == cluster_terms)
print("Panel == MINDIST  :", panel_terms == mindist_terms)
print("Clusters == MINDIST:", cluster_terms == mindist_terms)

# %%
# =============================================================================
# Cluster summary
# =============================================================================

cluster_summary = (
    clusters["cluster"]
    .value_counts()
    .sort_index()
    .rename_axis("cluster")
    .reset_index(name="n_terms")
)

display(cluster_summary)

print()
print(f"Total clusters : {cluster_summary.shape[0]}")
print(f"Total terms    : {cluster_summary.n_terms.sum()}")

# %%
# =============================================================================
# Verify MINDIST matrix
# =============================================================================

print("Shape:", mindist.shape)

print()

print("Rows equal columns:",
      list(mindist.index) == list(mindist.columns))

print()

print("Symmetric:",
      np.allclose(mindist.values, mindist.values.T))

# %%
# =============================================================================
# Terms belonging to each cluster
# =============================================================================

cluster_lookup = {}

for c in sorted(clusters.cluster.unique()):

    cluster_lookup[c] = (
        clusters.loc[
            clusters.cluster == c,
            "term"
        ].tolist()
    )

for c in cluster_lookup:

    print(f"Cluster {c}: {len(cluster_lookup[c])} terms")

# %%
# Missing terms
print(clusters["term"].isna().sum())

clusters[clusters["term"].isna()]

# %%
# =============================================================================
# Rank terms inside each cluster
# =============================================================================

representative_rankings = {}

for cluster, terms in cluster_lookup.items():

    # MINDIST submatrix
    sub = mindist.loc[terms, terms]

    # Average distance to all other members
    mean_distance = sub.mean(axis=1)

    representative_rankings[cluster] = (
        mean_distance
        .sort_values()
        .reset_index()
        .rename(columns={
            "index": "term",
            0: "mean_distance"
        })
    )

# %%
# =============================================================================
# Construct cluster indices
# =============================================================================

cluster_indices = {}

fractions = {
    "100%": 1.00,
    "75%": 0.75,
    "50%": 0.50,
    "25%": 0.25,
    "10%": 0.10,
}

for label, fraction in fractions.items():

    print(f"Building {label}...")

    cluster_indices[label] = build_cluster_indices(
        panel=panel,
        representative_rankings=representative_rankings,
        fraction=fraction
    )

print("\nFinished!\n")

print(cluster_indices.keys())

# %%
# =============================================================================
# Prepare market datasets
# =============================================================================

market_data = {}

for name, df in indices.items():

    data = df.copy()

    # Yahoo column names
    close_col = "Close"

    data = data.sort_values("date")

    data["return"] = np.log(data[close_col]).diff()

    # Squared return as realised volatility proxy
    data["realized_volatility"] = data["return"] ** 2

    data = data.dropna().reset_index(drop=True)

    market_data[name] = data

    print(
        f"{name}:",
        len(data),
        "observations"
    )

# %%
# =============================================================================
# Merge attention indices with market data
# =============================================================================

merged_data = {}

for fraction, attention in cluster_indices.items():

    merged_data[fraction] = {}

    for market, prices in market_data.items():

        merged = (
            prices.merge(
                attention,
                on="date",
                how="inner"
            )
            .sort_values("date")
            .reset_index(drop=True)
        )

        merged_data[fraction][market] = merged

        print(
            f"{market} ({fraction:.2f}) ->",
            merged.shape
        )


