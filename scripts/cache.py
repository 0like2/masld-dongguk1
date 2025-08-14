#!/usr/bin/env python3

"""
무인자 캐싱 스크립트 (dongguk#1.py와 호환)
- 메타/카운트 TSV를 읽어서 AnnData(.h5ad) 캐시를 생성
- obs 인덱스: "{FileName}_{CellID}"
- 출력: data/adata_cached.h5ad  (dongguk#1.py가 읽는 경로와 동일)

필요 패키지: pandas, polars, scipy, anndata
"""

import os
import time
import pandas as pd
import polars as pl
import scipy.sparse as sp
from anndata import AnnData

# =============================
# 사용자 설정 (dongguk#1.py와 맞춤)
# =============================
PT_META   = "../data/GSE166504_cell_metadata.20220204.tsv"
PT_COUNTS = "../data/GSE166504_cell_raw_counts.20220204.txt"
OUT_H5AD  = "data/adata_cached.h5ad"

GENE_COL = None                 # None이면 counts 파일의 첫 컬럼 사용
OBS_INDEX_FMT = "{FileName}_{CellID}"   # meta 컬럼을 이용해 obs 인덱스 생성
SEP_META = "\t"
SEP_COUNTS = "\t"
# =============================

def mkdir_p(p):
    os.makedirs(p, exist_ok=True)
    return p

def main():
    t0 = time.time()
    mkdir_p(os.path.dirname(OUT_H5AD) or ".")

    print("[cache] 1) 메타데이터 로딩 (pandas)")
    meta = pd.read_csv(PT_META, sep=SEP_META)
    if "FileName" not in meta.columns or "CellID" not in meta.columns:
        raise ValueError("메타데이터에 'FileName' 또는 'CellID' 컬럼이 없습니다.")

    # obs 인덱스 생성
    meta["__obs_index__"] = meta.apply(lambda r: OBS_INDEX_FMT.format(**r), axis=1)
    if meta["__obs_index__"].duplicated().any():
        dup = meta["__obs_index__"][meta["__obs_index__"].duplicated()].unique()[:5]
        raise ValueError(f"중복 obs 인덱스 발견 (예시): {dup}")

    print("[cache] 2) 카운트 로딩 (Polars) — 대용량 TSV")
    df_pl = pl.read_csv(
        PT_COUNTS,
        separator=SEP_COUNTS,
        has_header=True,
        truncate_ragged_lines=True
    )
    gene_col = GENE_COL or df_pl.columns[0]

    print(f"[cache]   gene_col='{gene_col}', 총 컬럼={len(df_pl.columns)}")
    if gene_col not in df_pl.columns:
        raise ValueError(f"gene_col '{gene_col}' 이(가) 카운트 파일 컬럼에 없습니다.")

    print("[cache] 3) Polars → pandas → CSR 변환")
    df_pd = df_pl.to_pandas()
    var_names = df_pd[gene_col].astype(str).tolist()
    obs_names = [str(x) for x in df_pd.columns[1:]]

    counts_mat = sp.csr_matrix(df_pd.drop(columns=gene_col).values.T, dtype="float32")

    print("[cache] 4) meta 정렬/체크")
    # meta 인덱스: counts의 셀 순서와 동일해야 함
    meta = meta.set_index("__obs_index__")
    try:
        meta = meta.loc[obs_names]
    except KeyError:
        # 어떤 셀이 누락되었는지 보여주기
        missing = [x for x in obs_names if x not in meta.index][:10]
        raise AssertionError(f"메타와 카운트의 셀 식별자 불일치 (예시 10개): {missing}")

    # 순서 확인
    assert list(meta.index) == obs_names, "셀 순서가 일치하지 않습니다."
    # 유전자 중복 체크
    if pd.Series(var_names).duplicated().any():
        dupg = pd.Series(var_names)[pd.Series(var_names).duplicated()].unique()[:5]
        raise ValueError(f"유전자명 중복 발견 (예시): {dupg}")

    print("[cache] 5) AnnData 구성")
    adata = AnnData(
        X=counts_mat,                # (cells x genes) 형태로 들어감 (지금은 transpose 해서 맞춤)
        obs=meta,
        var=pd.DataFrame(index=var_names)
    )
    # 여기서는 counts_mat가 (samples x genes)가 아니라 (cells x genes)로 맞춰져 있음
    # (위에서 transpose 했기 때문에 OK)

    print(f"[cache]   cells={adata.n_obs:,}, genes={adata.n_vars:,}")

    print(f"[cache] 6) 저장 → {OUT_H5AD} (compression=lzf)")
    adata.write_h5ad(OUT_H5AD, compression="lzf")

    dt = time.time() - t0
    print(f"[cache] 완료! 경로: {OUT_H5AD}  (경과 {dt:.1f}s)")

if __name__ == "__main__":
    main()
