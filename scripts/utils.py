# utils.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# -----------------------
# FS helpers
# -----------------------
def mkdir_p(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def savefig_tight(path: str | Path, dpi: int = 200, transparent: bool = False) -> None:
    path = Path(path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight", transparent=transparent)
    plt.close()


# -----------------------
# Small utilities
# -----------------------
def infer_gene_col(df: pd.DataFrame) -> str:
    """
    Try to infer the gene column name from common candidates,
    otherwise assume the first column is gene id/symbol.
    """
    cands = [c for c in df.columns if c.lower() in
             ["gene", "genes", "geneid", "symbol", "gene_name"]]
    return cands[0] if cands else df.columns[0]

def log2p(x: np.ndarray | pd.Series | pd.DataFrame) -> np.ndarray:
    """log2(x + 1) with numpy; returns np.ndarray."""
    return np.log2(np.asarray(x) + 1.0)


# -----------------------
# Plotting
# -----------------------
def plot_pca(
    log_norm_counts: pd.DataFrame,
    meta: pd.DataFrame,
    out_png: str | Path,
    sample_col: str = "sample",
    hue: str = "group",
    n_components: int = 2,
    random_state: int = 42,
    with_standardize: bool = True,
) -> pd.DataFrame:
    """
    PCA on (genes x samples) matrix (columns are samples).
    Returns the PC dataframe (for downstream use) and saves a plot.
    """
    # samples x genes
    X = log_norm_counts.T.values

    if with_standardize:
        X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    pca = PCA(n_components=n_components, random_state=random_state)
    PCs = pca.fit_transform(X)
    pc_cols = [f"PC{i+1}" for i in range(n_components)]

    pc_df = pd.DataFrame(PCs, columns=pc_cols, index=log_norm_counts.columns).reset_index()
    pc_df.rename(columns={"index": sample_col}, inplace=True)

    if sample_col not in meta.columns:
        raise ValueError(f"`meta` must contain '{sample_col}' column.")
    if sample_col not in pc_df.columns:
        pc_df = pc_df.reset_index().rename(columns={"index": sample_col})
    pc_df = pc_df.merge(meta, on=sample_col, how="left")

    plt.figure(figsize=(6, 5))
    ax = sns.scatterplot(data=pc_df, x="PC1", y="PC2", hue=hue, s=80, edgecolor="k")
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100 if n_components >= 2 else 0.0
    plt.title(f"PCA (PC1 {var1:.1f}%, PC2 {var2:.1f}%)")
    savefig_tight(out_png)
    return pc_df


def plot_volcano(
    res_df: pd.DataFrame,
    out_png: str | Path,
    padj_col: str = "padj",
    l2fc_col: str = "log2FoldChange",
    padj_th: float = 0.05,
    l2fc_th: float = 0.5,
    title: str = "Volcano Plot",
) -> pd.DataFrame:
    """
    Volcano plot for DE results. Returns a copy of dataframe with 'neglog10_padj' and 'sig' columns.
    """
    df = res_df.copy()
    # Avoid -log10(0)
    eps = np.nextafter(0, 1)
    df["neglog10_padj"] = -np.log10(df[padj_col].replace(0, eps))

    df["sig"] = "ns"
    df.loc[(df[padj_col] < padj_th) & (df[l2fc_col] >  l2fc_th), "sig"] = "Up"
    df.loc[(df[padj_col] < padj_th) & (df[l2fc_col] < -l2fc_th), "sig"] = "Down"

    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=df, x=l2fc_col, y="neglog10_padj", hue="sig",
        palette={"ns": "#aaaaaa", "Up": "#d62728", "Down": "#1f77b4"},
        s=12, edgecolor=None
    )
    plt.axvline(x=l2fc_th,  ls="--", c="k", lw=0.8)
    plt.axvline(x=-l2fc_th, ls="--", c="k", lw=0.8)
    plt.axhline(y=-np.log10(padj_th), ls="--", c="k", lw=0.8)
    plt.xlabel("log2 Fold Change")
    plt.ylabel("-log10(padj)")
    plt.title(title)
    savefig_tight(out_png)
    return df


def plot_heatmap(
    log_norm_counts: pd.DataFrame,
    sig_genes: Sequence[str],
    meta: pd.DataFrame,
    out_png: str | Path,
    top_n: int = 50,
    group_col: str = "group",
    sample_col: str = "sample",
    cmap: str = "vlag",
) -> None:
    """
    Heatmap of top-N genes (z-score by gene across samples).
    """
    genes = [g for g in sig_genes[:top_n] if g in log_norm_counts.index]
    if not genes:
        raise ValueError("No genes to plot (none found in expression matrix).")

    plot_mat = log_norm_counts.loc[genes]

    if group_col not in meta.columns or sample_col not in meta.columns:
        raise ValueError(f"`meta` must contain '{group_col}' and '{sample_col}' columns.")

    order = meta.sort_values(group_col)[sample_col].tolist()
    order = [s for s in order if s in plot_mat.columns]
    plot_mat = plot_mat[order]

    # row-wise z-score
    mu = plot_mat.mean(axis=1)
    sd = plot_mat.std(axis=1).replace(0, np.nan)  # avoid div by zero
    z = (plot_mat.sub(mu, axis=0)).div(sd, axis=0).fillna(0.0)

    plt.figure(figsize=(min(14, 0.4 * z.shape[1] + 4), 0.25 * len(genes) + 4))
    sns.heatmap(z, cmap=cmap, center=0, cbar_kws={"label": "Z-score"})
    plt.title("Top DEGs (Z-score by gene)")
    plt.ylabel("Genes")
    plt.xlabel("Samples")
    savefig_tight(out_png, dpi=250)


def plot_violin_lpl(
    log_norm_counts: pd.DataFrame,
    meta: pd.DataFrame,
    lpl_list: Sequence[str],
    outdir: str | Path,
    group_col: str = "group",
    sample_col: str = "sample",
    title: str = "LPL-related genes (log2 normalized counts)",
) -> Optional[pd.DataFrame]:
    """
    Violin plot for a list of LPL-related genes. Saves one figure and returns a melted dataframe (or None).
    """
    gset = [g for g in lpl_list if g in log_norm_counts.index]
    if not gset:
        # nothing to plot
        return None

    df = log_norm_counts.loc[gset].T
    if sample_col not in meta.columns or group_col not in meta.columns:
        raise ValueError(f"`meta` must contain '{sample_col}' and '{group_col}' columns.")
    df = df.join(meta.set_index(sample_col))

    melted = df.melt(id_vars=[group_col], var_name="gene", value_name="logExpr")

    plt.figure(figsize=(max(8, 0.4 * len(gset) + 4), 5))
    sns.violinplot(
        data=melted, x="gene", y="logExpr", hue=group_col,
        split=True, inner="quartile", cut=0
    )
    plt.xticks(rotation=90)
    plt.title(title)

    outdir = mkdir_p(outdir)
    savefig_tight(outdir / "LPL_genes_violin.png")
    return melted
