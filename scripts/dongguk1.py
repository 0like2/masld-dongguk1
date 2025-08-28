# 터미널에서 scripts 폴더로 이동 후 -> python dongguk1.py --config config.yaml
import os
import argparse
import warnings
import glob
import shutil

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import yaml
import gseapy as gp
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

from utils import (mkdir_p, plot_pca, plot_volcano, plot_heatmap, plot_violin,
                   plot_enrichment_results, plot_filtered_enrichment_results, plot_ssgsea_scores,
                   plot_dot_expression)


def load_config(path):
    """YAML 설정 파일을 로드하고 기본값을 설정합니다."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    for k in ["counts", "metadata", "outdir"]:
        if k not in cfg or not cfg[k]:
            raise ValueError(f"config 파일에 '{k}' 경로가 반드시 필요합니다.")
    return cfg


def run_enrichment_on_degs(gene_list, out_prefix, libraries):
    """주어진 유의 유전자 목록으로 Enrichment 분석을 수행하고 결과를 반환합니다."""
    if not gene_list:
        print(f"{os.path.basename(out_prefix)}: 분석할 유전자가 없어 Enrichment를 건너뜁니다.")
        return None
    print(f"{os.path.basename(out_prefix)}: Enrichment 분석 중 ({len(gene_list)}개 유전자)...")
    try:
        # 'organism' 인자를 추가하여 마우스 기준으로 분석하도록 수정
        enr_res = gp.enrichr(gene_list=gene_list, gene_sets=libraries, organism='Mouse', outdir=None, cutoff=0.05)
        if enr_res is not None:
            enr_res.results.to_csv(f"{out_prefix}_summary.csv")
            return enr_res.results
    except Exception as e:
        print(f"Enrichment 분석 중 오류 발생: {e}")
        return None


def run_gsea_prerank(ranked_gene_list, out_prefix, libraries, processes=4):
    """주어진 순위 유전자 목록으로 Pre-ranked GSEA를 수행합니다."""
    if ranked_gene_list.empty:
        print("순위 목록이 비어 있어 Pre-ranked GSEA를 건너뜁니다.")
        return None
    print(f"{os.path.basename(out_prefix)}: Pre-ranked GSEA 분석 중...")
    try:
        pre_res = gp.prerank(rnk=ranked_gene_list, gene_sets=libraries, outdir=out_prefix,
                             min_size=15, max_size=500, permutation_num=1000,
                             threads=processes, seed=6, format='png')
        return pre_res
    except Exception as e:
        print(f"Pre-ranked GSEA 분석 중 오류 발생: {e}")
        return None


def run_ssgsea(expression_df, gene_sets, out_prefix, processes=4):
    """주어진 발현 데이터와 유전자 세트로 ssGSEA를 수행합니다."""
    print(f"{os.path.basename(out_prefix)}: ssGSEA 분석 중...")
    try:
        ssgsea_res = gp.ssgsea(data=expression_df, gene_sets=gene_sets,
                               outdir=None, threads=processes)
        ssgsea_res.res2d.to_csv(f"{out_prefix}_scores.csv")
        return ssgsea_res.res2d
    except Exception as e:
        print(f"ssGSEA 분석 중 오류 발생: {e}")
        return None


def generate_summary_report(stats_data: dict, outfile: str, step_num: int):
    """분석 통계 요약 리포트를 CSV 파일로 생성합니다."""
    print(f"\n--- {step_num}. 요약 리포트 생성 ---")

    up_lipid_count = 0
    if stats_data['up_enrichment_results'] is not None and stats_data['lipid_keywords']:
        pattern = '|'.join(stats_data['lipid_keywords'])
        up_lipid_count = stats_data['up_enrichment_results']['Term'].str.contains(pattern, case=False).sum()

    dn_lipid_count = 0
    if stats_data['down_enrichment_results'] is not None and stats_data['lipid_keywords']:
        pattern = '|'.join(stats_data['lipid_keywords'])
        dn_lipid_count = stats_data['down_enrichment_results']['Term'].str.contains(pattern, case=False).sum()

    summary_data = {
        "Category": [
            "Total Genes in Count Matrix", "Total Samples",
            "Condition: Experimental", "Condition: Control",
            "Total Significant DEGs", "Up-regulated DEGs", "Down-regulated DEGs",
            "GSEA: Total Enriched Pathways",
            "Enrichment (Up): Total Pathways", "Enrichment (Up): Lipid-related",
            "Enrichment (Down): Total Pathways", "Enrichment (Down): Lipid-related",
            "ssGSEA: Senescence Signatures Analyzed",
            "ssGSEA: Up-regulated Lipid GO Signatures",
            "ssGSEA: Down-regulated Lipid GO Signatures"
        ],
        "Value": [
            stats_data['total_genes'], stats_data['n_samples'],
            stats_data['conditions'].get('Experimental', 0), stats_data['conditions'].get('Control', 0),
            stats_data['total_degs'], stats_data['up_degs'], stats_data['down_degs'],
            stats_data['gsea_pathways'],
            stats_data['enrichment_up_pathways'], up_lipid_count,
            stats_data['enrichment_down_pathways'], dn_lipid_count,
            len(stats_data['signatures']) if stats_data['signatures'] else 0,
            stats_data['lipid_go_terms_up_count'],
            stats_data['lipid_go_terms_down_count']
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(outfile, index=False)
    print(f"요약 리포트 저장 완료: {os.path.basename(outfile)}")


def organize_output_files(output_directory: str, step_num: int):
    """결과 폴더의 모든 PNG 파일을 'figures' 하위 폴더로 이동합니다."""
    print(f"\n--- {step_num}. 결과 파일 정리 ---")
    figure_dir = os.path.join(output_directory, "figures")
    mkdir_p(figure_dir)

    png_files = glob.glob(os.path.join(output_directory, "**", "*.png"), recursive=True)

    for f in png_files:
        if os.path.dirname(f) != figure_dir:
            try:
                shutil.move(f, figure_dir)
            except Exception as e:
                print(f"파일 이동 실패 {os.path.basename(f)}: {e}")

    print(f"모든 PNG 파일을 '{figure_dir}' 폴더로 이동했습니다.")


# --- 메인 분석 로직 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DESeq2, GSEA, 시각화를 수행하는 RNA-seq 분석 파이프라인")
    parser.add_argument("--config", required=True, help="YAML 설정 파일 경로")
    args = parser.parse_args()

    step_counter = 1

    # --- 1. 설정 및 데이터 로딩 ---
    print(f"--- {step_counter}. 설정 및 데이터 로딩 ---");
    step_counter += 1
    cfg = load_config(args.config)
    outdir = mkdir_p(cfg["outdir"])
    counts_df = pd.read_csv(cfg["counts"], index_col=0)
    meta_df = pd.read_csv(cfg["metadata"], dtype={'sample': str})
    print(f"Count Matrix 로드 완료: {counts_df.shape} (유전자 수 x 샘플 수)")
    print(f"Metadata 로드 완료: {meta_df.shape} (샘플 수 x 정보)")

    # --- 2. DEG 분석 및 시각화 ---
    print(f"\n--- {step_counter}. DEG 분석 및 시각화 ---");
    step_counter += 1
    meta_df_indexed = meta_df.set_index("sample")
    dds = DeseqDataSet(counts=counts_df.T, metadata=meta_df_indexed, design="~ condition")
    dds.deseq2()
    stat_res = DeseqStats(dds, contrast=["condition", "Experimental", "Control"])
    stat_res.summary()
    res_df = stat_res.results_df
    res_df.to_csv(os.path.join(outdir, "DEG_results_all.csv"))
    print("DEG 분석 완료. 전체 결과 저장: DEG_results_all.csv")

    padj_th = float(cfg["padj"])
    l2fc_th = float(cfg["l2fc"])
    plot_volcano(res_df, os.path.join(outdir, cfg["plots"]["volcano_filename"]), padj_th=padj_th, l2fc_th=l2fc_th)
    print(f"Volcano Plot 저장 완료: {cfg['plots']['volcano_filename']}")
    sig_mask = (res_df["padj"] < padj_th) & (res_df["log2FoldChange"].abs() > l2fc_th)
    res_sig = res_df.loc[sig_mask].sort_values("padj")
    res_sig.to_csv(os.path.join(outdir, "DEG_results_significant.csv"))
    print(f"유의미한 DEG {len(res_sig)}개 필터링 완료 및 저장: DEG_results_significant.csv")

    dds.layers['log1p_normed_counts'] = np.log1p(dds.layers['normed_counts'])
    log_norm_counts = pd.DataFrame(dds.layers['log1p_normed_counts'], index=dds.obs_names, columns=dds.var_names).T

    plot_pca(log_norm_counts, meta_df.rename(columns={"condition": "group"}),
             os.path.join(outdir, cfg["plots"]["pca_filename"]), hue="group")
    print(f"PCA Plot 저장 완료: {cfg['plots']['pca_filename']}")

    if len(res_sig) > 1:
        plot_heatmap(log_norm_counts, res_sig.index, meta_df.rename(columns={"condition": "group"}),
                     os.path.join(outdir, cfg["plots"]["heatmap_filename"]), top_n=min(50, len(res_sig)))
        print(f"Heatmap 저장 완료: {cfg['plots']['heatmap_filename']}")

    # --- 3. GSEA Prerank 분석 ---
    print(f"\n--- {step_counter}. GSEA Prerank 분석 ---");
    step_counter += 1
    res_df_ranked = res_df.dropna(subset=['pvalue', 'log2FoldChange']).copy()
    res_df_ranked['rank_metric'] = -np.log10(res_df_ranked['pvalue'].replace(0, 1e-300)) * np.sign(
        res_df_ranked['log2FoldChange'])
    ranked_gene_list = res_df_ranked[['rank_metric']].squeeze().sort_values(ascending=False)
    gsea_prefix = os.path.join(outdir, "GSEA_Prerank")
    gsea_res = run_gsea_prerank(ranked_gene_list, gsea_prefix, cfg["gsea"]["libraries"],
                                processes=cfg["gsea"].get("processes", 4))

    # --- 4. 유의 유전자 기반 Enrichment 분석 ---
    print(f"\n--- {step_counter}. 유의 유전자 기반 Enrichment 분석 ---");
    step_counter += 1
    up_genes = res_sig[res_sig["log2FoldChange"] > 0].index.tolist()
    dn_genes = res_sig[res_sig["log2FoldChange"] < 0].index.tolist()

    # 먼저 Enrichment 분석을 한 번에 실행하여 전체 결과를 얻습니다.
    up_prefix_base = os.path.join(outdir, "Enrichment_Up_regulated")
    up_enrichment = run_enrichment_on_degs(up_genes, up_prefix_base, cfg["gsea"]["libraries"])
    dn_prefix_base = os.path.join(outdir, "Enrichment_Down_regulated")
    down_enrichment = run_enrichment_on_degs(dn_genes, dn_prefix_base, cfg["gsea"]["libraries"])

    # 각 라이브러리별로 결과를 필터링하여 개별적으로 그림을 생성합니다.
    if up_enrichment is not None:
        print("\n[Up-regulated] 라이브러리별 Enrichment 결과 시각화 중...")
        for lib_name in cfg["gsea"]["libraries"]:
            lib_results = up_enrichment[up_enrichment['Gene_set'] == lib_name]
            lib_prefix = f"{up_prefix_base}_{lib_name.replace(' ', '_').split('(')[0]}"
            plot_enrichment_results(lib_results, lib_prefix)

    if down_enrichment is not None:
        print("\n[Down-regulated] 라이브러리별 Enrichment 결과 시각화 중...")
        for lib_name in cfg["gsea"]["libraries"]:
            lib_results = down_enrichment[down_enrichment['Gene_set'] == lib_name]
            lib_prefix = f"{dn_prefix_base}_{lib_name.replace(' ', '_').split('(')[0]}"
            plot_enrichment_results(lib_results, lib_prefix)


    # --- 5. Lipid 관련 경로 필터링 및 시각화 ---
    print(f"\n--- {step_counter}. Lipid 관련 경로 필터링 및 시각화 ---");
    step_counter += 1
    if cfg["gsea"].get("lipid_keywords"):
        plot_filtered_enrichment_results(up_enrichment, cfg["gsea"]["lipid_keywords"], up_prefix_base)
        plot_filtered_enrichment_results(down_enrichment, cfg["gsea"]["lipid_keywords"], dn_prefix_base)

    # --- 6. Senescence Signature ssGSEA 분석 ---
    print(f"\n--- {step_counter}. Senescence Signature ssGSEA 분석 ---");
    step_counter += 1
    signatures = cfg.get("signatures")
    if signatures:
        senescence_prefix = os.path.join(outdir, "ssGSEA_Signatures")
        signature_scores = run_ssgsea(log_norm_counts, signatures, senescence_prefix,
                                      processes=cfg["gsea"].get("processes", 4))
        if signature_scores is not None:
            plot_ssgsea_scores(signature_scores, meta_df, senescence_prefix, prefix="Signatures_", plot_type='grid')

    # --- 7. Lipid-term ssGSEA 분석 ---
    print(f"\n--- {step_counter}. Lipid-term ssGSEA 분석 ---");
    step_counter += 1
    lipid_gene_sets_up = {}
    if up_enrichment is not None:
        go_bp_results = up_enrichment[up_enrichment['Gene_set'] == 'GO_Biological_Process_2023']
        if not go_bp_results.empty and cfg["gsea"].get("lipid_keywords"):
            pattern = '|'.join(cfg["gsea"]["lipid_keywords"])
            lipid_go_terms = go_bp_results[go_bp_results['Term'].str.contains(pattern, case=False)]
            if not lipid_go_terms.empty:
                top_lipid_terms = lipid_go_terms.sort_values("Adjusted P-value").head(10)

                lipid_gene_sets_up = {
                    row['Term']: row['Genes'].split(';')
                    for _, row in top_lipid_terms.iterrows()
                }

                lipid_ssgsea_prefix = os.path.join(outdir, "ssGSEA_Lipid_GO_Terms_Up")
                lipid_scores = run_ssgsea(log_norm_counts, lipid_gene_sets_up, lipid_ssgsea_prefix,
                                          processes=cfg["gsea"].get("processes", 4))
                if lipid_scores is not None:
                    plot_ssgsea_scores(lipid_scores, meta_df, lipid_ssgsea_prefix, prefix="Lipid_GO_Up_",
                                       plot_type='grid')

    lipid_gene_sets_down = {}
    if down_enrichment is not None:
        go_bp_results_dn = down_enrichment[down_enrichment['Gene_set'] == 'GO_Biological_Process_2023']
        if not go_bp_results_dn.empty and cfg["gsea"].get("lipid_keywords"):
            pattern = '|'.join(cfg["gsea"]["lipid_keywords"])
            lipid_go_terms_dn = go_bp_results_dn[go_bp_results_dn['Term'].str.contains(pattern, case=False)]
            if not lipid_go_terms_dn.empty:
                top_lipid_terms_dn = lipid_go_terms_dn.sort_values("Adjusted P-value").head(10)
                lipid_gene_sets_down = {
                    row['Term']: row['Genes'].split(';')
                    for _, row in top_lipid_terms_dn.iterrows()
                }
                lipid_ssgsea_prefix_dn = os.path.join(outdir, "ssGSEA_Lipid_GO_Terms_Down")
                lipid_scores_dn = run_ssgsea(log_norm_counts, lipid_gene_sets_down, lipid_ssgsea_prefix_dn,
                                             processes=cfg["gsea"].get("processes", 4))
                if lipid_scores_dn is not None:
                    plot_ssgsea_scores(lipid_scores_dn, meta_df, lipid_ssgsea_prefix_dn, prefix="Lipid_GO_Down_",
                                       plot_type='grid')

    # --- 8. 관심 유전자 발현 시각화 ---
    print(f"\n--- {step_counter}. 관심 유전자 발현 시각화 ---");
    step_counter += 1
    if "lpl_genes" in cfg and cfg["lpl_genes"]:
        plot_violin(norm_counts=log_norm_counts, gene_list=cfg["lpl_genes"],
                    meta=meta_df.rename(columns={"condition": "group"}),
                    path=os.path.join(outdir, cfg["plots"]["violin_filename"]), title="LPL-related Gene Expression")
        plot_dot_expression(
            norm_counts=log_norm_counts,
            gene_list=cfg["lpl_genes"],
            meta=meta_df.rename(columns={"condition": "group"}),
            path=os.path.join(outdir, "LPL_genes_dotplot.png"),
            title="LPL-related Gene Expression"
        )

    # --- 9. 요약 리포트 및 파일 정리 ---
    stats_to_report = {
        'total_genes': counts_df.shape[0],
        'n_samples': meta_df.shape[0],
        'conditions': meta_df['condition'].value_counts().to_dict(),
        'padj_th': padj_th,
        'l2fc_th': l2fc_th,
        'total_degs': len(res_sig),
        'up_degs': len(up_genes),
        'down_degs': len(dn_genes),
        'up_enrichment_results': up_enrichment,
        'down_enrichment_results': down_enrichment,
        'enrichment_up_pathways': len(up_enrichment) if up_enrichment is not None else 0,
        'enrichment_down_pathways': len(down_enrichment) if down_enrichment is not None else 0,
        'gsea_pathways': len(gsea_res.res2d) if gsea_res is not None else 0,
        'lipid_keywords': cfg["gsea"].get("lipid_keywords"),
        'signatures': signatures,
        'lipid_go_terms_up_count': len(lipid_gene_sets_up),
        'lipid_go_terms_down_count': len(lipid_gene_sets_down)
    }
    generate_summary_report(stats_to_report, os.path.join(outdir, "summary_report.csv"), step_num=step_counter);
    step_counter += 1
    organize_output_files(outdir, step_num=step_counter);
    step_counter += 1

    print("\n[Done] 모든 분석이 완료되었습니다. 결과를 'results/final_analysis' 폴더에서 확인하세요.")
