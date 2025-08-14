import pandas as pd

# 카운트 데이터 파일 경로 (dongguk1.py와 같은 위치에서 실행 기준)
counts_file_path = "../data/GSE288534_Palbo_Genecount.txt"

# 데이터 불러오기
try:
    df = pd.read_csv(counts_file_path, sep="\t")
    # 첫 번째 열을 유전자 열로 가정
    gene_col = df.columns[0]

    # 중복된 유전자 수 계산
    num_duplicates = df[gene_col].duplicated().sum()

    if num_duplicates > 0:
        print(f"🚨 총 {num_duplicates}개의 중복된 유전자 이름을 찾았습니다.")

        # 실제 중복된 유전자들 보기 (상위 10개 그룹)
        print("\n--- 중복된 유전자 목록 (예시) ---")
        duplicated_genes = df[df[gene_col].duplicated(keep=False)]
        print(duplicated_genes.sort_values(by=gene_col).head(20))
    else:
        print("✅ 중복된 유전자가 없습니다.")

except FileNotFoundError:
    print(f"❌ 파일을 찾을 수 없습니다: {counts_file_path}")
