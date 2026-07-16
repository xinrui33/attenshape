import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------
EMBEDDINGS_PATH = "C:/Python/CSUREMM/output/bert/mpnet_embeddings.npy"
TERMS_PATH = "C:/Python/CSUREMM/output/bert/terms_index.csv"
SAX_CLUSTERS_PATH = "C:/Python/CSUREMM/output/archive_output_sax_tests/sax_tests_july_14/04_clustering/final_cluster_assignments.csv"
OUTPUT_DIR = "C:/Python/CSUREMM/output/cluster_comparison"

os.makedirs(OUTPUT_DIR, exist_ok=True)

embeddings = np.load(EMBEDDINGS_PATH)                 # (1203, 768), row-aligned to terms_index
terms_df = pd.read_csv(TERMS_PATH)                     # raw_term, processed_term
sax_df = pd.read_csv(SAX_CLUSTERS_PATH)                # term, cluster (SAX/temporal cluster labels)

print(f"Embeddings: {embeddings.shape}")
print(f"Terms index: {terms_df.shape}")
print(f"SAX cluster assignments: {sax_df.shape}, {sax_df['cluster'].nunique()} clusters")

# ---------------------------------------------------------------
# 2. Align the two term universes
# ---------------------------------------------------------------
# The SAX clustering was run on a subset of terms (847 of 1203) that had
# sufficient Google Trends time-series coverage. We match on a normalized
# (lowercased, whitespace-stripped) version of the processed term string.
terms_df = terms_df.reset_index().rename(columns={"index": "embedding_row"})
terms_df["match_key"] = terms_df["processed_term"].astype(str).str.strip().str.lower()

sax_df = sax_df.dropna(subset=["term"]).copy()
sax_df["match_key"] = sax_df["term"].astype(str).str.strip().str.lower()

merged = terms_df.merge(sax_df[["match_key", "cluster"]], on="match_key", how="inner")

n_unmatched = sax_df["match_key"].nunique() - merged["match_key"].nunique()
print(f"Matched {len(merged)} / {len(sax_df)} SAX-clustered terms to embeddings "
      f"({n_unmatched} unmatched).")

# Subset embeddings to only the rows that have a SAX cluster label,
# preserving row order alignment via embedding_row.
aligned_embeddings = embeddings[merged["embedding_row"].values]
sax_labels = merged["cluster"].values
terms = merged["processed_term"].values

n_sax_clusters = len(np.unique(sax_labels))
print(f"Working set: {aligned_embeddings.shape[0]} terms, {n_sax_clusters} SAX clusters")

# ---------------------------------------------------------------
# 3. Hierarchical clustering on BERT embeddings
# ---------------------------------------------------------------
# PRIMARY: Ward linkage, to match the linkage algorithm used for the SAX
# clustering. Ward is only mathematically valid under Euclidean distance
# (scipy enforces this), so we run it on the raw embedding vectors rather
# than a precomputed distance matrix. Because the mpnet embeddings are
# L2-normalized (normalize_embeddings=True at encoding time), Euclidean
# distance and cosine distance produce IDENTICAL rankings on this data
# (d_euclidean^2 = 2 - 2*cos_sim for unit vectors) -- so Ward here still
# respects cosine similarity structure while remaining a valid operation.
#
# Note: MINDIST itself (the SAX-specific lower-bound distance) has no
# equivalent for dense continuous embeddings, so it is not reused here --
# only the linkage algorithm (Ward) is matched, not the distance metric.
n_norm_check = np.linalg.norm(aligned_embeddings, axis=1)
assert np.allclose(n_norm_check, 1.0, atol=1e-3), (
    "Embeddings are not L2-normalized -- Ward+Euclidean will NOT match cosine "
    "ranking in this case. Re-encode with normalize_embeddings=True, or "
    "normalize aligned_embeddings here before proceeding."
)

Z_ward = linkage(aligned_embeddings, method="ward")  # metric is implicitly Euclidean
bert_labels = fcluster(Z_ward, t=n_sax_clusters, criterion="maxclust")

# SECONDARY (robustness check): average-linkage + cosine distance, the
# standard choice for embeddings when not trying to match a Ward pipeline.
Z_avg = linkage(aligned_embeddings, method="average", metric="cosine")
bert_labels_avg = fcluster(Z_avg, t=n_sax_clusters, criterion="maxclust")

print(f"Ward+Euclidean produced {len(np.unique(bert_labels))} clusters "
      f"(target was {n_sax_clusters}).")
print(f"Average+Cosine (robustness check) produced {len(np.unique(bert_labels_avg))} clusters.")

# ---------------------------------------------------------------
# 4. Cluster agreement metrics
# ---------------------------------------------------------------
def report_agreement(labels_a, labels_b, name):
    ari = adjusted_rand_score(labels_a, labels_b)
    nmi = normalized_mutual_info_score(labels_a, labels_b)
    fmi = fowlkes_mallows_score(labels_a, labels_b)
    print(f"\n--- {name} ---")
    print(f"Adjusted Rand Index (ARI):        {ari:.4f}  (0 = random, 1 = identical)")
    print(f"Normalized Mutual Info (NMI):     {nmi:.4f}  (0 = independent, 1 = identical)")
    print(f"Fowlkes-Mallows Index (FMI):      {fmi:.4f}  (0 = random, 1 = identical)")
    return ari, nmi, fmi

ari, nmi, fmi = report_agreement(sax_labels, bert_labels, "SAX (temporal) vs. BERT Ward+Euclidean [PRIMARY]")
report_agreement(sax_labels, bert_labels_avg, "SAX (temporal) vs. BERT Average+Cosine [robustness check]")

# ---------------------------------------------------------------
# 5. Permutation test for significance of the agreement score
# ---------------------------------------------------------------
# Low ARI alone doesn't tell you whether the overlap is "surprisingly low" —
# compare against a null distribution built by shuffling one label set.
rng = np.random.default_rng(seed=42)
N_PERMUTATIONS = 1000
null_aris = np.empty(N_PERMUTATIONS)
for i in range(N_PERMUTATIONS):
    shuffled = rng.permutation(bert_labels)
    null_aris[i] = adjusted_rand_score(sax_labels, shuffled)

p_value = (np.sum(null_aris >= ari) + 1) / (N_PERMUTATIONS + 1)
print(f"\nPermutation test: observed ARI={ari:.4f} vs. null mean={null_aris.mean():.4f} "
      f"(std={null_aris.std():.4f}), p={p_value:.4f}")

# ---------------------------------------------------------------
# 6. Overlay 1 — contingency heatmap (SAX cluster x BERT cluster)
# ---------------------------------------------------------------
contingency = pd.crosstab(
    pd.Series(sax_labels, name="SAX (temporal) cluster"),
    pd.Series(bert_labels, name="BERT Ward (semantic) cluster"),
)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(contingency, annot=True, fmt="d", cmap="viridis", ax=ax, cbar_kws={"label": "# terms"})
ax.set_title("Term overlap: SAX temporal clusters vs. BERT semantic clusters")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "contingency_heatmap.png"), dpi=150)
plt.close()
print("Saved contingency heatmap.")

# ---------------------------------------------------------------
# 7. Overlay 2 — side-by-side 2D projection colored by each labeling
# ---------------------------------------------------------------
# PCA on the embedding space gives a shared 2D layout; coloring the same
# points by SAX label vs. BERT label makes semantic/temporal disagreement
# visually inspectable (points that scatter across colors in one panel but
# not the other = terms grouped one way but not the other).
coords = PCA(n_components=2, random_state=42).fit_transform(aligned_embeddings)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
for ax, labels, title in zip(
    axes, [sax_labels, bert_labels], ["Colored by SAX (temporal) cluster", "Colored by BERT Ward (semantic) cluster"]
):
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=18, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("PC1")
axes[0].set_ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_overlay.png"), dpi=150)
plt.close()
print("Saved PCA overlay scatter.")

# ---------------------------------------------------------------
# 8. Save the merged, per-term comparison table for manual inspection
# ---------------------------------------------------------------
comparison_df = pd.DataFrame({
    "term": terms,
    "sax_cluster": sax_labels,
    "bert_cluster": bert_labels,
})
comparison_df.to_csv(os.path.join(OUTPUT_DIR, "sax_vs_bert.csv"), index=False)
print("Saved per-term comparison table: sax_vs_bert.csv")

# Preliminary analysis:
# (1) agreement between the SAX clusters and BERT semantic clusters is low
# (2)alignment is uneven across topics:
# Not every semantic category disperses across temporal clusters equally. Two clear counter-examples in the data:
# Elections: BERT cluster 5 (election 2024, 2020 election results, did trump win, etc.) is 89% concentrated in a single SAX cluster.
# Finance/markets: BERT cluster 9 (bitcoin, dow jones, apple stock, etc.) is 83% concentrated in one SAX cluster.
# Sports is semantically coherent (BERT cluster 1) but scatters across six different SAX clusters — 
# because college football season, election cycles, 
# and hurricane season all have genuinely different temporal shapes, sports terms don't share one shape despite sharing one topic.
# some semantic categories do have an inherently periodic or event-driven real-world rhythm (elections, markets) 
# that happens to align with a temporal cluster, while others (sports, entertainment) don't.
# The direction of the finding (low semantic agreement) is stable across linkage choices, even though the magnitude isn't

# --- SAX (temporal) vs. BERT Average+Cosine [robustness check] ---
# Adjusted Rand Index (ARI):        0.0292  (0 = random, 1 = identical)
# Normalized Mutual Info (NMI):     0.1328  (0 = independent, 1 = identical)
# Fowlkes-Mallows Index (FMI):      0.4122  (0 = random, 1 = identical)