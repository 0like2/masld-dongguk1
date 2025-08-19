import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ===================================================================
# 설정: 분석할 파일과 결과를 저장할 폴더 경로를 직접 지정합니다.
# ===================================================================
COUNTS_PATH = "../results/count_matrix_annotated.csv"
METADATA_PATH = "../data/GSE288534_meta.csv"
RESULTS_DIR = "../results/"


# ===================================================================

def run_eda():
    """
    EDA, 전처리, 최종 PCA 시각화를 수행합니다.
    """
    print("🚀 EDA 및 전처리 스크립트를 시작합니다.")

    # --- 1. 데이터 로딩 ---
    try:
        counts_df = pd.read_csv(COUNTS_PATH, index_col=0)
        metadata_df = pd.read_csv(METADATA_PATH, index_col=0)
    except FileNotFoundError as e:
        print(f"❌ 오류: 파일을 찾을 수 없습니다. '{e.filename}'")
        return

    # --- 2. 원본 데이터 요약 (Transcript 수준) ---
    print("\n" + "=" * 50)
    print("📊 2. 원본 데이터 요약 (Transcript 수준)")
    print("=" * 50)
    print("\n--- [원본 target_id (전사체) 요약] ---")
    unique_ids = counts_df.index.nunique()
    print(f"  - 고유한 target_id 수: {unique_ids}개")
    print("\n--- [원본 gene_symbol (유전자) 요약] ---")
    unique_symbols = counts_df[counts_df['gene_symbol'] != 'N/A']['gene_symbol'].nunique()
    print(f"  - 고유한 gene_symbol 수: {unique_symbols}개")

    # --- 3. 샘플별 발현 유전자 수 요약 ---
    print("\n" + "=" * 50)
    print("🧬 3. 샘플별 발현 유전자 수 요약 (Count > 0)")
    print("=" * 50)
    count_columns = [col for col in counts_df.columns if col != 'gene_symbol']
    for sample in count_columns:
        expressed_mask = counts_df[sample] > 0
        expressed_target_ids = expressed_mask.sum()
        expressed_symbols = counts_df.loc[expressed_mask & (counts_df['gene_symbol'] != 'N/A'), 'gene_symbol'].nunique()
        print(f"  - [{sample}]: target_id {expressed_target_ids}개 / gene_symbol {expressed_symbols}개")

    # --- 4. 1차 전처리: 발현량 없는 전사체 제거 ---
    print("\n" + "=" * 50)
    print("🛠️  4. 1차 전처리: 발현량 없는 전사체(target_id) 제거")
    print("=" * 50)
    original_transcript_count = len(counts_df)
    original_symbol_count = counts_df[counts_df['gene_symbol'] != 'N/A']['gene_symbol'].nunique()
    genes_to_keep = counts_df[count_columns].sum(axis=1) > 0
    transcript_filtered_df = counts_df[genes_to_keep]
    filtered_transcript_count = len(transcript_filtered_df)
    num_removed = original_transcript_count - filtered_transcript_count
    print(f"  - 모든 샘플에서 발현량이 0인 target_id {num_removed}개를 제거했습니다.")
    filtered_symbol_count = transcript_filtered_df[transcript_filtered_df['gene_symbol'] != 'N/A'][
        'gene_symbol'].nunique()
    print("\n--- [전처리 전/후 ID 개수 비교] ---")
    print(f"  - target_id: {original_transcript_count}개  ->  {filtered_transcript_count}개")
    print(f"  - gene_symbol: {original_symbol_count}개  ->  {filtered_symbol_count}개")

    # --- 5. 2차 전처리: Gene Symbol 기준으로 데이터 최종 집계 ---
    print("\n" + "=" * 50)
    print("✨ 5. 2차 전처리: Gene Symbol 기준으로 데이터 최종 집계")
    print("=" * 50)
    known_genes_df = transcript_filtered_df[transcript_filtered_df['gene_symbol'] != 'N/A']
    gene_level_counts = known_genes_df.groupby('gene_symbol').sum()
    print(f"  - 최종 분석에 사용될 고유 유전자(gene_symbol)의 수: {len(gene_level_counts)}개")

    # --- 6. 최종 전처리된 데이터 저장 ---
    print("\n" + "=" * 50)
    print("💾 6. 최종 전처리된 데이터 파일로 저장")
    print("=" * 50)
    output_path = os.path.join(RESULTS_DIR, 'count_matrix_preprocessed.csv')
    gene_level_counts.to_csv(output_path)
    print(f"  ✅ 최종 전처리된 데이터(Gene 수준)가 다음 경로에 저장되었습니다:\n     {output_path}")

    # --- 7. PCA 분석 및 시각화 (Gene 수준 데이터 사용) ---
    print("\n" + "=" * 50)
    print("📈 7. PCA 분석 및 시각화 (샘플 클러스터링 확인)")
    print("=" * 50)

    # PCA를 위해 데이터를 전치 (샘플 x 유전자)하고 log 변환 및 표준화
    log_transformed_data = np.log1p(gene_level_counts.T)
    scaled_data = StandardScaler().fit_transform(log_transformed_data)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_data)

    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=metadata_df.index)
    pca_df = pd.concat([pca_df, metadata_df], axis=1)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='condition', data=pca_df, s=150, alpha=0.8)

    for i in range(pca_df.shape[0]):
        plt.text(pca_df['PC1'][i] + 0.3, pca_df['PC2'][i] + 0.3, pca_df.index[i], fontsize=9)

    plt.title('PCA of Samples (Control vs Experimental)', fontsize=16)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    plt.legend(title='Condition')
    plt.grid(True, linestyle='--', alpha=0.6)

    pca_plot_path = os.path.join(RESULTS_DIR, 'pca_plot_eda.png')
    plt.savefig(pca_plot_path)
    print(f"  ✅ PCA Plot이 다음 경로에 저장되었습니다:\n     {pca_plot_path}")
    plt.close()

    print("\n--- EDA 및 전처리 종료 ---")


if __name__ == '__main__':
    run_eda()
