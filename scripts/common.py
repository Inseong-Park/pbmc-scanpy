# common.py — 여러 스크립트가 같이 쓰는 경로, 그림 설정, 데이터 읽기
# 다른 스크립트에서 `from common import ...`로 불러 쓴다. 직접 실행하는 파일이 아니다.

import warnings
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
R_PROJECT = ROOT.parent / "pbmc-scrna"  # 비교 대상인 R 프로젝트
DATA = R_PROJECT / "data" / "filtered_gene_bc_matrices" / "hg19"
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
WORK = ROOT / "work"
for d in (FIG, TAB, WORK):
    d.mkdir(parents=True, exist_ok=True)

# 그림에 한글이 깨지지 않게 (Windows 기본 한글 글꼴)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# R의 01_qc.R에서 정하고, Python에서도 그대로 쓰기로 한 기준
MIN_FEAT = 200
MAX_FEAT = 2500
MAX_MT = 5


def load_pbmc():
    """10x 행렬을 읽고 R의 CreateSeuratObject(min.cells = 3, min.features = 200)와 같게 거른다.

    원래는 sc.read_10x_mtx(DATA, var_names="gene_symbols") 한 줄이면 된다.
    그런데 그 안에서 쓰는 scipy의 빠른 행렬 읽기(C++)가 Windows에서 경로에 한글이 있으면
    파일을 제대로 열지 못한다 ("Not a Matrix Market file" 오류).
    그래서 파일을 Python으로 직접 연 뒤 내용을 넘기는 방식으로, read_10x_mtx가 하는 일을 손으로 한다.
    """
    with open(DATA / "matrix.mtx", "rb") as f:
        X = scipy.io.mmread(f, spmatrix=False)  # 유전자 x 세포 (파일에 저장된 모양 그대로, R과 같음)
    genes = pd.read_csv(DATA / "genes.tsv", sep="\t", header=None, names=["gene_ids", "symbol"])
    barcodes = pd.read_csv(DATA / "barcodes.tsv", sep="\t", header=None)[0]

    with warnings.catch_warnings():  # "Variable names are not unique" 경고: 바로 아래에서 해결하므로 숨긴다
        warnings.simplefilter("ignore", UserWarning)
        adata = ad.AnnData(
            X=sp.csr_matrix(X.T, dtype=np.float32),  # 뒤집어서 세포 x 유전자로 (AnnData 관례)
            obs=pd.DataFrame(index=barcodes.values),
            var=pd.DataFrame({"gene_ids": genes["gene_ids"].values}, index=genes["symbol"].values),
        )
    # 같은 기호가 두 번 나오는 유전자가 있다 → 뒤에 -1을 붙여 구분한다 (Seurat은 .1을 붙임)
    adata.var_names_make_unique()
    n_genes_raw = adata.n_vars

    # Seurat과 같은 순서: 세포 먼저 거르고, 남은 세포로 유전자를 거른다
    sc.pp.filter_cells(adata, min_genes=MIN_FEAT)  # 유전자 200개 미만 바코드 제외
    sc.pp.filter_genes(adata, min_cells=3)         # 세포 3개 미만에서만 검출된 유전자 제외

    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    adata.uns["n_genes_raw"] = n_genes_raw
    return adata
