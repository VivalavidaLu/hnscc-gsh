"""Fit cNMF models for K=4..8 using the submitted 1,508-gene feature set.

The exact ordered feature manifest used for the submitted analysis is versioned
with the repository. Up to 500 cells per patient are sampled for cNMF from the
6,165 frozen malignant-cell universe.
"""
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata
anndata.settings.allow_write_nullable_strings = True
import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("HNSCC_DATA_DIR", REPOSITORY / "data")).resolve()
RESULTS_ROOT = Path(os.environ.get("HNSCC_RESULTS_DIR", REPOSITORY / "results")).resolve()
from cnmf import cNMF

FINAL_IDS     = DATA_ROOT / "derived" / "malignant_DEFINITIVE_FINAL_cellids.csv"
FEATURE_GENES = DATA_ROOT / "derived" / "cnmf_feature_genes_1508.txt"
OUT_DIR       = str(RESULTS_ROOT / "cnmf")
RUN_DIR       = f"{OUT_DIR}/cnmf_runs"
RUN_NAME      = "malignant_final_v2"
K_RANGE       = list(range(4, 9))
N_ITER        = 200
SEED          = 42
N_PER_PATIENT = 500
N_FEATURES    = 1508

os.makedirs(RUN_DIR, exist_ok=True)
np.random.seed(SEED)

# ──────────────────────────────────────────────────────────────────────────────
print("="*64)
print("Loading all 6,165 frozen malignant cells")
print("="*64)
final_ids = pd.read_csv(FINAL_IDS)["cell_id"]
raw       = sc.read_h5ad(DATA_ROOT / "raw" / "single_cell_counts.h5ad")
common    = final_ids[final_ids.isin(raw.obs_names)].values
adata_all = raw[common].copy()
excl      = adata_all.obs["Patient"].value_counts()
excl      = excl[excl < 50].index.tolist()
if excl:
    adata_all = adata_all[~adata_all.obs["Patient"].isin(excl)].copy()
patients = sorted(adata_all.obs["Patient"].unique())
print(f"  {adata_all.n_obs:,} cells | {len(patients)} patients")

# ══════════════════════════════ STEP A ════════════════════════════════════════
print(f"\n{'='*64}")
print("STEP A: Load the submitted ordered cNMF feature manifest")
print(f"{'='*64}")
hvg_genes = [line.strip() for line in FEATURE_GENES.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(hvg_genes) != N_FEATURES:
    raise RuntimeError(f"Expected {N_FEATURES} cNMF feature genes, found {len(hvg_genes)}")
if len(set(hvg_genes)) != N_FEATURES:
    raise RuntimeError("The cNMF feature manifest contains duplicate genes")
missing_features = [gene for gene in hvg_genes if gene not in adata_all.var_names]
if missing_features:
    raise RuntimeError(f"Raw counts are missing {len(missing_features)} frozen cNMF features: {missing_features[:10]}")
print(f"  [STEP A RESULT] Loaded and validated {len(hvg_genes):,} ordered features")

# ══════════════════════════════ STEP B ════════════════════════════════════════
print(f"\n{'='*64}")
print(f"STEP B: N={N_PER_PATIENT} per-patient downsample (INDEPENDENT of A)")
print("  Downsampling controls patient contribution to NMF; the frozen feature set is unchanged")
print(f"{'='*64}")
sampled_ids = []
for pat in patients:
    pat_cells = adata_all.obs_names[adata_all.obs["Patient"] == pat].tolist()
    n = min(N_PER_PATIENT, len(pat_cells))
    chosen = np.random.choice(pat_cells, size=n, replace=False).tolist()
    sampled_ids.extend(chosen)
    print(f"  {pat:<12}: {n:>3}/{len(pat_cells):>4}")

total_cells = len(sampled_ids)
sub_pct     = adata_all[sampled_ids].obs["Patient"].value_counts() / total_cells * 100
print(f"\n  [STEP B RESULT] {total_cells:,} cells")
print(f"  Max patient: {sub_pct.index[0]} = {sub_pct.iloc[0]:.1f}%")
print(f"  Min patient: {sub_pct.index[-1]} = {sub_pct.iloc[-1]:.1f}%")
assert sub_pct.iloc[0] < 25, "Largest patient exceeds 25%; reduce N_PER_PATIENT."

# ══════════════════════════════ STEP C ════════════════════════════════════════
print(f"\n{'='*64}")
print(f"STEP C: Matrix = Step B cells × Step A HVGs, raw counts")
print(f"{'='*64}")
hvg_in_data = [g for g in hvg_genes if g in adata_all.var_names]
adata_cnmf  = adata_all[sampled_ids, hvg_in_data].copy()

Xdata   = adata_cnmf.X.data if sp.issparse(adata_cnmf.X) else adata_cnmf.X.flatten()
sample  = Xdata[:3000] if len(Xdata) > 3000 else Xdata
assert float(adata_cnmf.X.min()) >= 0.0, "Counts matrix contains negative values."
assert np.all(np.round(sample) == sample), "Counts matrix contains non-integer values."
spars = 1.0 - np.count_nonzero(Xdata) / (adata_cnmf.n_obs * adata_cnmf.n_vars)
print(f"  Shape:    {adata_cnmf.shape}")
print(f"  Range:    {float(adata_cnmf.X.min()):.0f} .. {float(adata_cnmf.X.max()):.0f}")
print(f"  Sparsity: {spars*100:.1f}%")
print(f"  [STEP C RESULT] Raw integer matrix confirmed")

hvg_path  = f"{OUT_DIR}/cnmf_v2_hvg.txt"
tsv_path  = f"{OUT_DIR}/malignant_v2_cnmf.tsv"
h5ad_path = f"{OUT_DIR}/malignant_v2_cnmf.h5ad"
with open(hvg_path, "w") as f:
    f.write("\n".join(hvg_in_data))
df = pd.DataFrame.sparse.from_spmatrix(
    adata_cnmf.X, index=adata_cnmf.obs_names, columns=adata_cnmf.var_names)
df.to_csv(tsv_path, sep="\t")
for col in adata_cnmf.obs.columns:
    if str(adata_cnmf.obs[col].dtype) in ["category", "string", "object"]:
        adata_cnmf.obs[col] = adata_cnmf.obs[col].astype(str)
adata_cnmf.write_h5ad(h5ad_path)
print("Saved cNMF input matrix under the configured results directory.")

# ══════════════════════════════ STEP D ════════════════════════════════════════
print(f"\n{'='*64}")
print(f"STEP D: cNMF {RUN_NAME} K={K_RANGE}, n_iter={N_ITER}")
print("  Fit candidate solutions and inspect stability/error for model selection")
print(f"{'='*64}")
cnmf_obj = cNMF(output_dir=RUN_DIR, name=RUN_NAME)
cnmf_obj.prepare(
    counts_fn=tsv_path, components=K_RANGE,
    n_iter=N_ITER, seed=SEED,
    num_highvar_genes=len(hvg_in_data),
    genes_file=hvg_path,
)
print("  Preparation complete. Factorizing...")
cnmf_obj.factorize(worker_i=0, total_workers=1)
print("  Factorization complete. Combining replicate spectra...")
cnmf_obj.combine(components=K_RANGE, skip_missing_files=False)

# `factorize` alone only writes replicate spectra.  Generate the consensus
# spectra/usages that all downstream K-selection and projection scripts read.
for k in K_RANGE:
    print(f"  Building consensus for K={k} (density threshold = 2.0)...")
    cnmf_obj.consensus(
        k=k,
        density_threshold=2.0,
        show_clustering=False,
        close_clustergram_fig=True,
    )

print("  [STEP D RESULT] Factorization, combination, and consensus complete.")
