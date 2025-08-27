# utils.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ===================================================================
#   1. 파일 시스템 및 경로 처리 (File System & Path Helpers)
# ===================================================================

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


# ===================================================================
#   2. 데이터 처리 유틸리티 (Data Utilities)
# ===================================================================

def infer_gene_col(df: pd.DataFrame) -> str:
    """데이터프레임에서 유전자 컬럼 이름을 추정합니다."""
    cands = [c for c in df.columns if c.lower() in
             ["gene", "genes", "geneid", "symbol", "gene_name"]]
    return cands[0] if cands else df.columns[0]


def log2p(x: np.ndarray | pd.Series | pd.DataFrame) -> np.ndarray:
    """log2(x + 1) 변환을 수행합니다."""
    return np.log2(np.asarray(x) + 1.0)


# ===================================================================
#   3. 시각화 함수 (Plotting Functions)
# ===================================================================

#   3.1. 핵심 분석 플롯 (Core Analysis Plots)

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
    """PCA를 수행, 결과를 scatter plot으로 저장합니다."""
    X = log_norm_counts.T.values
    if with_standardize:
        X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    pca = PCA(n_components=n_components, random_state=random_state)
    PCs = pca.fit_transform(X)
    pc_cols = [f"PC{i + 1}" for i in range(n_components)]

    pc_df = pd.DataFrame(PCs, columns=pc_cols, index=log_norm_counts.columns)
    pc_df = pc_df.reset_index().rename(columns={"index": sample_col})

    if sample_col not in meta.columns:
        raise ValueError(f"`meta` must contain '{sample_col}' column.")

    pc_df = pc_df.merge(meta, on=sample_col, how="left")

    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=pc_df, x="PC1", y="PC2", hue=hue, s=80, edgecolor="k")
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
    """DEG 분석 결과를 Volcano plot으로 시각화."""
    df = res_df.copy()
    eps = np.nextafter(0, 1)  # log10(0) 방지
    df["neglog10_padj"] = -np.log10(df[padj_col].replace(0, eps))

    df["sig"] = "ns"
    df.loc[(df[padj_col] < padj_th) & (df[l2fc_col] > l2fc_th), "sig"] = "Up"
    df.loc[(df[padj_col] < padj_th) & (df[l2fc_col] < -l2fc_th), "sig"] = "Down"

    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=df, x=l2fc_col, y="neglog10_padj", hue="sig",
        palette={"ns": "#aaaaaa", "Up": "#d62728", "Down": "#1f77b4"},
        s=12, edgecolor=None
    )
    plt.axvline(x=l2fc_th, ls="--", c="k", lw=0.8)
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
    """상위 유전자들의 발현량을 Heatmap으로 시각화합니다 (유전자 기준 Z-score)."""
    genes = [g for g in sig_genes[:top_n] if g in log_norm_counts.index]
    if not genes:
        raise ValueError("No genes to plot (none found in expression matrix).")

    plot_mat = log_norm_counts.loc[genes]

    if group_col not in meta.columns or sample_col not in meta.columns:
        raise ValueError(f"`meta` must contain '{group_col}' and '{sample_col}' columns.")

    order = meta.sort_values(group_col)[sample_col].tolist()
    order = [s for s in order if s in plot_mat.columns]
    plot_mat = plot_mat[order]

    mu = plot_mat.mean(axis=1)
    sd = plot_mat.std(axis=1).replace(0, np.nan)
    z = (plot_mat.sub(mu, axis=0)).div(sd, axis=0).fillna(0.0)

    plt.figure(figsize=(min(14, 0.4 * z.shape[1] + 4), 0.25 * len(genes) + 4))
    sns.heatmap(z, cmap=cmap, center=0, cbar_kws={"label": "Z-score"})
    plt.title("Top DEGs (Z-score by gene)")
    plt.ylabel("Genes")
    plt.xlabel("Samples")
    savefig_tight(out_png, dpi=250)

# -------------------------------------------------------------------
#   3.2. 기능 분석 결과 시각화 (Functional Analysis Plots)
# -------------------------------------------------------------------

def plot_enrichment_results(
        enrichment_results: pd.DataFrame,
        out_prefix: str,
        top_n: int = 20,
        padj_cutoff: float = 0.05
):
    """유의 유전자 기반 Enrichment 분석 결과를 Bar plot과 Dot plot으로 시각화합니다."""
    if enrichment_results is None or enrichment_results.empty:
        print(f"{os.path.basename(out_prefix)}: 시각화할 유의미한 경로가 없습니다.")
        return

    sig_res = enrichment_results[enrichment_results["Adjusted P-value"] < padj_cutoff]
    sig_res_sorted = sig_res.sort_values(by="Adjusted P-value", ascending=True).head(top_n)

    if sig_res_sorted.empty:
        print(f"{os.path.basename(out_prefix)}: 시각화할 유의미한 경로가 없습니다 (p < {padj_cutoff}).")
        return

    try:
        gp.barplot(sig_res_sorted, title=f"Top {top_n} Enriched Pathways", ofname=f"{out_prefix}_barplot.png",
                   top_term=top_n)
        print(f"Enrichment Bar Plot 저장 완료: {os.path.basename(out_prefix)}_barplot.png")
    except Exception as e:
        print(f"Bar Plot 생성 중 오류 발생: {e}")

    try:
        gp.dotplot(sig_res_sorted, title=f"Top {top_n} Enriched Pathways", ofname=f"{out_prefix}_dotplot.png",
                   top_term=top_n)
        print(f"Enrichment Dot Plot 저장 완료: {os.path.basename(out_prefix)}_dotplot.png")
    except Exception as e:
        print(f"Dot Plot 생성 중 오류 발생: {e}")


def plot_filtered_enrichment_results(
        enrichment_results: pd.DataFrame,
        keywords: list,
        out_prefix: str,
        top_n: int = 20,
        padj_cutoff: float = 0.05
):
    """키워드로 필터링된 Enrichment 분석 결과를 Dot plot으로 시각화합니다."""
    if enrichment_results is None or enrichment_results.empty:
        return

    pattern = '|'.join(keywords)
    filtered_df = enrichment_results[enrichment_results['Term'].str.contains(pattern, case=False)]

    sig_res = filtered_df[filtered_df["Adjusted P-value"] < padj_cutoff]
    sig_res_sorted = sig_res.sort_values(by="Adjusted P-value", ascending=True).head(top_n)

    if sig_res_sorted.empty:
        print(f"{os.path.basename(out_prefix)}: 키워드 필터링 후 시각화할 유의미한 경로가 없습니다.")
        return

    try:
        gp.dotplot(sig_res_sorted, title=f"Top {top_n} Keyword-filtered Pathways",
                   ofname=f"{out_prefix}_filtered_dotplot.png", top_term=top_n)
        print(f"Filtered Enrichment Dot Plot 저장 완료: {os.path.basename(out_prefix)}_filtered_dotplot.png")
    except Exception as e:
        print(f"Filtered Dot Plot 생성 중 오류 발생: {e}")


# -------------------------------------------------------------------
#   3.3. 유전자 그룹 발현 플롯 (Gene Set Expression Plots)
# -------------------------------------------------------------------

def plot_violin(norm_counts, gene_list, meta, path, title="Gene Expression"):
    """주어진 유전자 목록의 발현량을 Violin plot으로 시각화합니다."""
    plot_genes = [gene for gene in gene_list if gene in norm_counts.index]
    if not plot_genes:
        print(f"경고: 유전자 목록의 유전자를 count matrix에서 찾을 수 없습니다. Violin plot을 건너뜁니다.")
        return

    plot_data = norm_counts.loc[plot_genes].T
    plot_data = plot_data.merge(meta[['sample', 'group']], left_index=True, right_on='sample')
    plot_data_melted = plot_data.melt(id_vars=['sample', 'group'], var_name='gene', value_name='Expression Level')

    plt.figure(figsize=(max(8, len(plot_genes) * 0.5), 6))
    sns.violinplot(data=plot_data_melted, x='gene', y='Expression Level', hue='group', split=True, inner="quart",
                   palette="muted")
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Violin Plot 저장 완료: {os.path.basename(path)}")


def plot_dot_expression(norm_counts, gene_list, meta, path, title="Gene Expression"):
    """주어진 유전자 목록의 발현량을 Dot plot으로 시각화합니다."""
    plot_genes = [gene for gene in gene_list if gene in norm_counts.index]
    if not plot_genes:
        print(f"경고: Dot plot을 위한 유전자를 찾을 수 없습니다.")
        return

    plot_data = norm_counts.loc[plot_genes].T
    plot_data = plot_data.merge(meta[['sample', 'group']], left_index=True, right_on='sample')
    plot_data_melted = plot_data.melt(id_vars=['sample', 'group'], var_name='gene',
                                      value_name='Expression Level')

    gene_order = plot_data_melted.groupby('gene')['Expression Level'].median().sort_values(ascending=True).index

    plt.figure(figsize=(8, max(4, len(plot_genes) * 0.3)))
    sns.stripplot(data=plot_data_melted, y='gene', x='Expression Level', hue='group',
                  order=gene_order, jitter=0.2, dodge=True, palette="muted", alpha=0.8)

    sns.pointplot(data=plot_data_melted, y='gene', x='Expression Level', hue='group',
                  order=gene_order, estimator=np.median, errorbar=None,
                  markersize=0, scale=0, linestyles='-', color='gray', dodge=0.4)

    plt.title(title)
    plt.ylabel("Gene")
    plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc='upper left')
    savefig_tight(path)
    print(f"Dot Plot 저장 완료: {os.path.basename(path)}")


# -------------------------------------------------------------------
#   3.4. 점수 분석 플롯 (Score Analysis Plots)
# -------------------------------------------------------------------

def plot_ssgsea_scores(scores_df, meta_df, outdir, prefix="", plot_type='grid'):
    """ssGSEA 점수를 Box plot 또는 Grid plot으로 시각화합니다."""
    plot_data = scores_df.T.reset_index().rename(columns={'index': 'sample'})
    plot_data = plot_data.merge(meta_df[['sample', 'condition']], on='sample')
    mkdir_p(outdir)
    signatures = scores_df.index.tolist()

    if plot_type == 'individual':
        for signature in signatures:
            plt.figure(figsize=(6, 5))
            sns.boxplot(data=plot_data, x='condition', y=signature)
            sns.stripplot(data=plot_data, x='condition', y=signature, color=".25", jitter=True)
            plt.title(f"{signature} Score by Condition")
            plt.ylabel("ssGSEA Score")
            filename = f"{prefix}{signature}_ssGSEA_scores.png".replace(" ", "_").replace("/", "_")
            savefig_tight(os.path.join(outdir, filename))

    elif plot_type == 'grid' and len(signatures) > 1:
        n_plots = len(signatures)
        n_cols = 4  # 한 줄에 4개씩
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
        axes = axes.flatten()

        for i, signature in enumerate(signatures):
            sns.boxplot(data=plot_data, x='condition', y=signature, ax=axes[i])
            sns.stripplot(data=plot_data, x='condition', y=signature, color=".25", jitter=True, ax=axes[i])
            axes[i].set_title(signature, fontsize=10)
            axes[i].set_xlabel("")
            axes[i].set_ylabel("Score")

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        fig.suptitle(f"{prefix}ssGSEA Scores by Condition", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = f"{prefix}ssGSEA_scores_grid.png".replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(outdir, filename), dpi=200)
        plt.close()
    else:
        plot_ssgsea_scores(scores_df, meta_df, outdir, prefix, plot_type='individual')
