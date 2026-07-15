from __future__ import annotations

import json
import warnings
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# CONFIG -- edit for your local machine
# -----------------------------------------------------------------------------

DATA_DIR = Path(r"C:\Python\CSUREMM\data_events")
OUTPUT_DIR = Path(r"C:\Python\CSUREMM\output\sax_tests_july_07")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Current folder structure:
# DATA_DIR/
#     query_metadata.csv
#     weather/
#         chunks/
#         diagnostics/
#         stitched/
#             gt_stitched_weather_2022-01-01_2026-05-31.csv

# -----------------------------------------------------------------------------
# Export all query folder names to a text file
# -----------------------------------------------------------------------------

from pathlib import Path

folder_names = sorted(
    folder.name
    for folder in DATA_DIR.iterdir()
    if folder.is_dir()
)

output_file = DATA_DIR / "1203words.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for name in folder_names:
        f.write(name + "\n")

print(f"Saved {len(folder_names)} folder names to:")
print(output_file)