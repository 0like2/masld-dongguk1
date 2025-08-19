# 실행 방법: 터미널에서 scripts 폴더로 이동 후 -> python dongguk1.py --config config.yaml
import os
import argparse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import yaml
import gseapy as gp
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# utils.py의 시각화 함수들을 사용합니다.
from utils import mkdir_p, plot_pca, plot_volcano, plot_heatmap


def load_config(path):
    """YAML 설정 파일을 로드하고 기본값을 설정합니다."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # 필수 키 체크
    for k in ["counts", "metadata", "outdir"]:
        if k not in cfg or not cfg[k]:
            raise ValueError(f"config 파일에 '{k}' 경로가 반드시 필요합니다.")
    return cfg


def run_gsea_enrichr(gene_list, out_prefix, libraries):
    """주어진 유전자 목록으로 Enrichr 분석을 수행합니다."""
    if not gene_list:
        print(f"{os.path.basename(out_prefix)}: 분석할 유전자가 없어 GSEA/Enrichr를 건너뜁니다.")
        return
    print(f"{os.path.basename(out_prefix)}: Enrichr 분석 중 ({len(gene_list)}개 유전자)...")
    gp.enrichr(
        gene_list=gene_list,
        gene_sets=libraries,
        outdir=out_prefix,
        cutoff=0.05
    ).results.to_csv(f"{out_prefix}_summary.csv")


# --- 메인 분석 로직 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DESeq2, GSEA, 시각화를 수행하는 RNA-seq 분석 파이프라인")
    parser.add_argument("--config", required=True, help="YAML 설정 파일 경로")
    args = parser.parse_args()

    # 1. 설정 및 데이터 로드
    print("--- 1. 설정 및 데이터 로딩 ---")
    cfg = load_config(args.config)
    outdir = mkdir_p(cfg["outdir"])

    counts_df = pd.read_csv(cfg["counts"], index_col=0)
    meta_df = pd.read_csv(cfg["metadata"])
    print(f"Count Matrix 로드 완료: {counts_df.shape} (유전자 수 x 샘플 수)")
    print(f"Metadata 로드 완료: {meta_df.shape} (샘플 수 x 정보)")

    # 2. DESeq2를 이용한 DEG 분석
    print("\n--- 2. DEG 분석 시작 ---")
    meta_df_indexed = meta_df.set_index("sample")

    dds = DeseqDataSet(
        counts=counts_df.T,
        metadata=meta_df_indexed,
        design_factors="condition"
    )
    dds.deseq2()

    stat_res = DeseqStats(dds, contrast=["condition", "Experimental", "Control"])

    # <<< 수정된 부분 >>>
    # 1. 먼저 .summary()를 실행하여 통계 계산을 수행합니다. (이때 결과가 화면에 출력됩니다)
    stat_res.summary()
    # 2. 계산이 완료된 후, .results_df 속성을 통해 결과를 데이터프레임으로 가져옵니다.
    res_df = stat_res.results_df

    res_df.to_csv(os.path.join(outdir, "DEG_results_all.csv"))
    print("DEG 분석 완료. 전체 결과 저장: DEG_results_all.csv")

    # 3. 유의미한 DEG 필터링 및 시각화
    print("\n--- 3. 유의미한 DEG 필터링 및 시각화 ---")
    padj_th = float(cfg["padj"])
    l2fc_th = float(cfg["l2fc"])

    plot_volcano(
        res_df, os.path.join(outdir, cfg["plots"]["volcano_filename"]),
        padj_th=padj_th, l2fc_th=l2fc_th
    )
    print(f"Volcano Plot 저장 완료: {cfg['plots']['volcano_filename']}")

    sig_mask = (res_df["padj"] < padj_th) & (res_df["log2FoldChange"].abs() > l2fc_th)
    res_sig = res_df.loc[sig_mask].sort_values("padj")
    res_sig.to_csv(os.path.join(outdir, "DEG_results_significant.csv"))
    print(f"유의미한 DEG {len(res_sig)}개 필터링 완료 및 저장: DEG_results_significant.csv")

    dds.layers['log1p_normed_counts'] = np.log1p(dds.layers['normed_counts'])
    log_norm_counts = pd.DataFrame(dds.layers['log1p_normed_counts'], index=dds.obs_names, columns=dds.var_names)

    plot_pca(
        log_norm_counts.T, meta_df.rename(columns={"condition": "group"}),
        os.path.join(outdir, cfg["plots"]["pca_filename"]),
        hue="group"
    )
    print(f"PCA Plot 저장 완료: {cfg['plots']['pca_filename']}")

    if len(res_sig) > 1:
        plot_heatmap(
            log_norm_counts.T, res_sig.index, meta_df.rename(columns={"condition": "group"}),
            os.path.join(outdir, cfg["plots"]["heatmap_filename"]),
            top_n=min(50, len(res_sig))
        )
        print(f"Heatmap 저장 완료: {cfg['plots']['heatmap_filename']}")

    # 4. GSEA / Enrichr 분석
    print("\n--- 4. GSEA / Enrichr 분석 ---")
    up_genes = res_sig[res_sig["log2FoldChange"] > 0].index.tolist()
    dn_genes = res_sig[res_sig["log2FoldChange"] < 0].index.tolist()

    run_gsea_enrichr(up_genes, os.path.join(outdir, "GSEA_Up_regulated"), cfg["gsea"]["libraries"])
    run_gsea_enrichr(dn_genes, os.path.join(outdir, "GSEA_Down_regulated"), cfg["gsea"]["libraries"])

    print("\n[Done] 모든 분석이 완료되었습니다. 결과를 'results/final_analysis' 폴더에서 확인하세요.")
