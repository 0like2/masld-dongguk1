# 실행 : python dongguk1.py --config config.yaml

import os
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gseapy as gp
import yaml

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# utils.py (이름 정확히 'utils.py')에서 불러옵니다
from utils import (
    mkdir_p, savefig_tight, infer_gene_col, log2p,
    plot_pca, plot_volcano, plot_heatmap, plot_violin_lpl
)


# (Enrichment helpers, Config 로딩 부분은 수정 없음)
# ... (이전과 동일)
# -----------------------
# Enrichment helpers
# -----------------------
def do_enrichr(gene_list, out_csv_prefix, libraries):
    if len(gene_list) == 0:
        return []
    out_files = []
    for lib in libraries:
        enr = gp.enrichr(gene_list=gene_list, description=f"enrichr_{lib}",
                         gene_sets=[lib], outdir=None)
        df = enr.results.copy()
        path = f"{out_csv_prefix}_{lib}.csv"
        df.to_csv(path, index=False)
        out_files.append(path)
    return out_files


def filter_lipid_terms(enrichr_csvs, out_csv, keywords):
    frames = []
    for f in enrichr_csvs:
        if os.path.exists(f):
            df = pd.read_csv(f)
            mask = df["Term"].str.lower().fillna("").str.contains("|".join(keywords))
            frames.append(df.loc[mask].assign(Source=os.path.basename(f)))
    if frames:
        pd.concat(frames, axis=0).to_csv(out_csv, index=False)


def do_preranked_gsea(rnk_df, outdir, libraries, processes=4):
    mkdir_p(outdir)
    rnk_path = os.path.join(outdir, "ranked_list.rnk")
    rnk_df.to_csv(rnk_path, sep="\t", index=False, header=False)
    try:
        gp.prerank(
            rnk=rnk_path,
            gene_sets=libraries,
            processes=processes,
            max_size=5000, min_size=10,
            outdir=outdir, seed=42
        )
        return True
    except Exception as e:
        print(f"[WARN] preranked GSEA 실패: {e}")
        return False


# -----------------------
# Config 로딩 & 검증
# -----------------------
def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # 기본값 채우기
    cfg.setdefault("padj", 0.05)
    cfg.setdefault("l2fc", 0.5)
    cfg.setdefault("gsea", {})
    cfg["gsea"].setdefault("processes", 4)
    cfg["gsea"].setdefault("libraries", [
        "GO_Biological_Process_2023",
        "Reactome_2022",
        "KEGG_2021_Human",
        "MSigDB_Hallmark_2020",
    ])
    cfg["gsea"].setdefault("lipid_keywords", [
        "lipid", "fatty", "sterol", "cholesterol", "lipoprotein",
        "glycerolipid", "phospholipid", "triglyceride"
    ])
    cfg.setdefault("plots", {})
    cfg["plots"].setdefault("pca_filename", "PCA_log2norm.png")
    cfg["plots"].setdefault("volcano_filename", "Volcano.png")
    cfg["plots"].setdefault("heatmap_filename", "Heatmap_TopDEGs.png")
    cfg["plots"].setdefault("violin_filename", "LPL_genes_violin.png")
    cfg.setdefault("lpl_genes", [])

    # 필수 키 체크
    for k in ["counts", "metadata", "outdir"]:
        if k not in cfg or not cfg[k]:
            raise ValueError(f"config에 '{k}' 경로가 필요합니다.")
    return cfg


# -----------------------
# 실행(메인 함수 없이 top-level)
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="YAML/JSON 설정 파일 경로")
args = parser.parse_args()

cfg = load_config(args.config)

counts_path = cfg["counts"]
meta_path = cfg["metadata"]
outdir = mkdir_p(cfg["outdir"])
padj_th = float(cfg["padj"])
l2fc_th = float(cfg["l2fc"])
gsea_libs = cfg["gsea"]["libraries"]
gsea_cpus = int(cfg["gsea"]["processes"])
lip_keywords = [str(x).lower() for x in cfg["gsea"]["lipid_keywords"]]
plots_cfg = cfg["plots"]
lpl_genes = cfg["lpl_genes"]

# 1) 데이터 로드 (전처리된 파일을 불러오므로 매우 간단해짐)
if counts_path.lower().endswith((".tsv", ".txt")):
    counts_df = pd.read_csv(counts_path, sep="\t", index_col=0)
else:
    counts_df = pd.read_csv(counts_path, index_col=0)

meta = pd.read_csv(meta_path)
assert {"sample", "group"}.issubset(meta.columns), "metadata는 sample, group 컬럼 필요"

# 샘플 교집합
common = [s for s in meta["sample"] if s in counts_df.columns]
counts_df = counts_df[common]
meta = meta[meta["sample"].isin(common)].copy()
if counts_df.shape[1] == 0:
    raise ValueError("metadata와 counts의 sample 교집합이 없습니다.")

# 0-count gene 제거 및 NaN 처리
counts_df = counts_df.loc[counts_df.sum(axis=1) > 0]
counts_df = counts_df.fillna(0)

# group 체크
if set(meta["group"].unique()) - set(["Young", "Old"]):
    print("[WARN] group이 Young/Old가 아닙니다. 현재 값:", meta["group"].unique())

# 2) DESeq2
metadata = meta.set_index("sample")
dds = DeseqDataSet(
    counts=counts_df.T,
    metadata=metadata,
    design="~ group",
    refit_cooks=True
)
dds.deseq2()
stat_res = DeseqStats(dds, contrast=["group", "Palbo", "Control"], n_cpus=1)
stat_res.summary()
res = stat_res.results_df.copy()
res.index.name = "gene"
res = res.rename(columns={"pvalue": "pvalue", "padj": "padj", "log2FoldChange": "log2FoldChange"})
res.to_csv(os.path.join(outdir, "DEG_results_all.csv"))

# 정규화 & log
if hasattr(dds, "norm_counts"):
    # 구버전 호환 속성 (samples x genes 가정)
    norm_counts = dds.norm_counts
else:
    # 최신 버전: AnnData layers에 저장됨 (samples x genes)
    norm_counts = pd.DataFrame(
        dds.layers["normed_counts"],
        index=dds.obs_names,  # samples
        columns=dds.var_names  # genes
    )

log_norm_counts = pd.DataFrame(
    log2p(norm_counts.values), index=norm_counts.index, columns=norm_counts.columns
)

# 3) PCA
plot_pca(log_norm_counts, meta, os.path.join(outdir, plots_cfg["pca_filename"]))

# 4) Volcano & Heatmap
# (이하 내용은 수정 없음)
# ...
sig_mask = (res["padj"] < padj_th) & (res["log2FoldChange"].abs() > l2fc_th)
res_sig = res.loc[sig_mask].sort_values("padj")
res_sig.to_csv(os.path.join(outdir, "DEG_results_significant.csv"))
plot_volcano(res, os.path.join(outdir, plots_cfg["volcano_filename"]), padj_th, l2fc_th)

if len(res_sig) > 2:
    top_genes = res_sig.index.tolist()
    plot_heatmap(
        log_norm_counts, top_genes, meta,
        os.path.join(outdir, plots_cfg["heatmap_filename"]),
        top_n=min(50, len(top_genes))
    )

# 5) Enrichr (Up/Down)
up_genes = res_sig[res_sig["log2FoldChange"] > 0].index.tolist()
dn_genes = res_sig[res_sig["log2FoldChange"] < 0].index.tolist()

up_files = do_enrichr(up_genes, os.path.join(outdir, "GSEA_DEG_UP"), gsea_libs)
dn_files = do_enrichr(dn_genes, os.path.join(outdir, "GSEA_DEG_DN"), gsea_libs)

filter_lipid_terms(up_files, os.path.join(outdir, "GSEA_DEG_UP_lipid_terms.csv"), lip_keywords)
filter_lipid_terms(dn_files, os.path.join(outdir, "GSEA_DEG_DN_lipid_terms.csv"), lip_keywords)

# 6) Preranked GSEA
rnk = res[["log2FoldChange"]].replace([np.inf, -np.inf], np.nan).dropna().reset_index()
rnk.columns = ["gene", "score"]
do_preranked_gsea(rnk, os.path.join(outdir, "gsea_preranked"), gsea_libs, processes=gsea_cpus)

# 7) LPL genes
if lpl_genes:
    plot_violin_lpl(log_norm_counts, meta, lpl_genes, outdir)

print("\n[Done] 결과가 저장되었습니다:", outdir)
print("- DEG_results_all.csv / DEG_results_significant.csv")
print(f"- {plots_cfg['pca_filename']} / {plots_cfg['volcano_filename']} / {plots_cfg['heatmap_filename']}")
print("- GSEA_DEG_UP/DN_* .csv, *_lipid_terms.csv")
print("- gsea_preranked/ (가능한 경우)")
print(f"- {plots_cfg['violin_filename']}")
