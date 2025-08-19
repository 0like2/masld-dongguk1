import pandas as pd
import os
import re  # 파일 이름 분석을 위해 re 라이브러리 추가
import mygene


def make_matrix_with_annotation():
    """
    data 폴더의 abundance.tsv 파일들을 동적으로 탐색하여 유전자 이름(Gene Symbol)이 포함된
    하나의 Count Matrix CSV 파일을 생성합니다.
    """
    # --- 1. 경로 설정 ---
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, '../data')
    results_dir = os.path.join(script_dir, '../results')
    os.makedirs(results_dir, exist_ok=True)

    # --- 2. data 폴더에서 파일 동적 탐색 ---
    print("1. data 폴더에서 샘플 파일 동적 탐색 중...")
    # 파일 이름 패턴 정의: (GSM번호)_(샘플이름)_abundance.tsv
    file_pattern = re.compile(r"(GSM\d+)_(Veh_\d+|DpC_\d+)_abundance\.tsv")

    sample_files_to_process = []
    for filename in sorted(os.listdir(data_dir)):
        match = file_pattern.match(filename)
        if match:
            # 패턴과 일치하는 파일만 처리 목록에 추가
            sample_name = match.group(2)
            file_path = os.path.join(data_dir, filename)
            sample_files_to_process.append({'name': sample_name, 'path': file_path})
            print(f"  > 발견: {filename} (샘플명: {sample_name})")

    if not sample_files_to_process:
        print("\n에러: data 폴더에서 'GSM..._Veh_X_abundance.tsv' 또는 'GSM..._DpC_X_abundance.tsv' 형식의 파일을 찾을 수 없습니다.")
        print("파일 이름을 확인해주세요.")
        return

    # --- 3. 파일 읽기 및 병합 ---
    all_sample_dfs = []
    print("\n2. 개별 샘플 파일 읽는 중...")
    for sample_info in sample_files_to_process:
        temp_df = pd.read_csv(sample_info['path'], sep='\t')[['target_id', 'est_counts']]
        temp_df.rename(columns={'est_counts': sample_info['name']}, inplace=True)
        all_sample_dfs.append(temp_df)

    print("3. 모든 샘플을 하나의 Matrix로 병합하는 중...")
    merged_df = all_sample_dfs[0]
    for i in range(1, len(all_sample_dfs)):
        merged_df = pd.merge(merged_df, all_sample_dfs[i], on='target_id', how='outer')

    merged_df.set_index('target_id', inplace=True)
    merged_df.fillna(0, inplace=True)
    count_matrix = merged_df.round(0).astype(int)

    # --- 4. Annotation (유전자 이름 변환) ---
    print("4. 유전자 ID를 Gene Symbol로 변환하는 중 (Annotation)...")
    print("   (유전자 수가 많아 몇 분 정도 소요될 수 있습니다.)")

    mg = mygene.MyGeneInfo()
    gene_ids_no_version = count_matrix.index.str.split('.').str[0].tolist()

    gene_info = mg.querymany(
        gene_ids_no_version,
        scopes='ensembl.transcript',
        fields='symbol',
        species='mouse',
        verbose=False  # 불필요한 경고 메시지 끄기
    )

    id_to_symbol_map = {info['query']: info.get('symbol', 'N/A') for info in gene_info if 'query' in info}

    count_matrix['gene_symbol'] = count_matrix.index.str.split('.').str[0].map(id_to_symbol_map)
    count_matrix['gene_symbol'].fillna('N/A', inplace=True)

    cols = ['gene_symbol'] + [col for col in count_matrix.columns if col != 'gene_symbol']
    annotated_count_matrix = count_matrix[cols]

    # --- 5. CSV 파일로 저장 ---
    output_path = os.path.join(results_dir, 'count_matrix_annotated.csv')
    annotated_count_matrix.to_csv(output_path)

    print("-" * 50)
    print(f"✅ Annotation된 Count Matrix 생성 완료!")
    print(f"   - 저장 위치: {output_path}")
    print("\n미리보기 (상위 5개):")
    print(annotated_count_matrix.head())


if __name__ == '__main__':
    try:
        import mygene
    except ImportError:
        print("에러: 'mygene' 라이브러리가 설치되지 않았습니다.")
        print("터미널에서 'pip install mygene'을 실행해주세요.")
    else:
        make_matrix_with_annotation()
