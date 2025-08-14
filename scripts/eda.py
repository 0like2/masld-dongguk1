import pandas as pd

# =============================================
# 설정: 분석할 파일 경로를 지정하세요.
# (현재 위치: 'scripts' 폴더 안)
# =============================================
COUNTS_PATH = "../data/GSE288534_Palbo_Genecount.txt"
META_PATH = "../data/GSE288534_meta.csv"


def explore_data():
    """
    카운트 데이터와 메타데이터의 기본 정보를 요약하고,
    데이터 오류를 확인 및 수정하는 탐색적 분석을 수행합니다.
    """
    print("🚀 데이터 탐색 및 요약 스크립트를 시작합니다.")

    # 1. 카운트 데이터 파일 분석
    # ---------------------------------------------
    print("\n" + "=" * 40)
    print("📊 1. 카운트 데이터 파일 분석 (GSE288534_Palbo_Genecount.txt)")
    print("=" * 40)

    try:
        # 탭으로 구분된 텍스트 파일이므로 sep='\t' 사용
        df_counts = pd.read_csv(COUNTS_PATH, sep='\t')
        gene_col = df_counts.columns[0]  # 첫 번째 열을 유전자 ID 컬럼으로 지정

        print(f"  - 총 유전자(행) 개수: {df_counts.shape[0]}개")
        print(f"  - 총 샘플(열) 개수: {df_counts.shape[1] - 1}개")

        print("\n--- [데이터 원본 상위 5줄 (head)] ---")
        print(df_counts.head())

    except FileNotFoundError:
        print(f"❌ 오류: 카운트 파일을 찾을 수 없습니다. 경로를 확인하세요: {COUNTS_PATH}")
        return

    # 2. 데이터 오류 확인 및 수정
    # ---------------------------------------------
    print("\n" + "=" * 40)
    print("🛠️  2. 데이터 오류 확인 및 수정")
    print("=" * 40)

    # 2-1. 엑셀 날짜 변환 오류 수정
    print("\n--- [2-1. 엑셀 날짜 변환 오류 수정] ---")
    replacements = {'1-Mar': 'MARCH1', '2-Mar': 'MARCH2'}
    df_counts[gene_col] = df_counts[gene_col].replace(replacements)
    print("  - '1-Mar' -> 'MARCH1', '2-Mar' -> 'MARCH2'로 수정 완료.")

    # 2-2. 중복 유전자 이름 고유하게 만들기 (접미사 추가)
    print("\n--- [2-2. 중복 유전자 이름에 접미사 추가] ---")

    # 각 유전자 이름별로 그룹을 지어 0부터 순번을 매김 (첫 번째는 0, 두 번째는 1, ...)
    counts = df_counts.groupby(gene_col).cumcount()

    # 순번이 1 이상인 경우(즉, 두 번째 이후의 중복 항목)에만 접미사를 붙일 마스크 생성
    mask = counts > 0

    if mask.any():
        print(f"  - 중복된 유전자 {mask.sum()}개에 접미사를 추가합니다.")
        # 예: 'MARCH1'의 두 번째 항목 -> 'MARCH1_1'
        df_counts.loc[mask, gene_col] = df_counts.loc[mask, gene_col] + '_' + counts[mask].astype(str)

        # 확인을 위해 수정된 중복 항목의 일부를 출력
        print("\n--- [수정된 유전자 이름 확인 (예시)] ---")
        # 원래 중복이었던 유전자들의 현재 상태를 보여줌
        original_duplicates = df_counts[gene_col].str.startswith('MARCH')
        print(df_counts[original_duplicates])
    else:
        print("  - 접미사를 추가할 중복 유전자가 없습니다.")

    # 2-3. 중복 여부 최종 확인
    print("\n--- [2-3. 중복 여부 최종 확인] ---")
    num_duplicates_after = df_counts[gene_col].duplicated().sum()
    if num_duplicates_after == 0:
        print("✅  좋습니다: 모든 유전자 이름이 고유합니다.")
    else:
        print(f"⚠️  경고: 아직 {num_duplicates_after}개의 중복이 남아있습니다.")

    # 3. 메타데이터 파일 분석
    # ---------------------------------------------
    # (이하 내용은 이전과 동일)
    print("\n" + "=" * 40)
    print("📝 3. 메타데이터 파일 분석 (GSE288534_meta.csv)")
    print("=" * 40)
    try:
        df_meta = pd.read_csv(META_PATH)
        print(f"  - 총 샘플 수: {df_meta.shape[0]}개")

        print("\n--- [메타데이터 상위 5줄 (head)] ---")
        print(df_meta.head())

        print("\n--- [그룹별 샘플 개수] ---")
        print(df_meta['group'].value_counts())

    except FileNotFoundError:
        print(f"❌ 오류: 메타데이터 파일을 찾을 수 없습니다. 경로를 확인하세요: {META_PATH}")
        return

    # 4. 두 데이터 간 샘플 이름 일치 여부 확인
    # ---------------------------------------------
    print("\n" + "=" * 40)
    print("🔗 4. 카운트 & 메타데이터 일치 여부 확인")
    print("=" * 40)

    counts_samples = set(df_counts.columns[1:])
    meta_samples = set(df_meta['sample'])

    if counts_samples == meta_samples:
        print("✅ 좋습니다: 카운트 데이터와 메타데이터의 샘플 이름이 모두 일치합니다!")
    else:
        print("⚠️  경고: 두 파일의 샘플 이름이 일치하지 않습니다!")
        if meta_samples - counts_samples:
            print(f"  - 메타데이터에만 있는 샘플: {meta_samples - counts_samples}")
        if counts_samples - meta_samples:
            print(f"  - 카운트 데이터에만 있는 샘플: {counts_samples - meta_samples}")

    # 5. 수정된 데이터 파일로 저장
    # ---------------------------------------------
    print("\n" + "=" * 40)
    print("💾 5. 수정된 데이터 파일로 저장")
    print("=" * 40)

    print("  - 최종 저장을 위해 유전자 ID를 인덱스로 설정합니다.")
    df_counts = df_counts.set_index(gene_col)
    print("  - 인덱스 설정 후 카운트 데이터의 상위 5줄을 출력합니다:")
    print(df_counts.head())

    # 원본과 구분되도록 '_preprocessed.tsv' 접미사를 붙여 저장합니다.
    OUTPUT_COUNTS_PATH = "../data/GSE288534_Palbo_Genecount_preprocessed.txt"

    df_counts.to_csv(OUTPUT_COUNTS_PATH, sep='\t', index=True)

    print(f" 수정된 카운트 데이터가 다음 경로에 저장되었습니다:\n   {OUTPUT_COUNTS_PATH}")

if __name__ == "__main__":
    explore_data()
