# 01_qc.py — 세포 품질 관리(QC), R의 01_qc.R을 scanpy로 다시 한 것
#
# 하는 일:
#   1. 10x 행렬을 AnnData 객체로 읽는다 (R과 같은 데이터 폴더를 그대로 읽음)
#   2. 너무 드문 유전자, 유전자가 너무 적은 바코드를 뺀다 (R의 min.cells = 3, min.features = 200)
#   3. 세포마다 QC 지표 3개를 계산한다
#        n_genes_by_counts : 검출된 유전자 수   (R의 nFeature_RNA)
#        total_counts      : 전체 UMI 수        (R의 nCount_RNA)
#        pct_counts_mt     : 미토콘드리아 비율  (R의 percent.mt)
#   4. 기준으로 세포를 거르고, R 결과(notes.md Day 1)와 숫자를 비교한다
#   5. 거른 객체를 work/pbmc_qc.h5ad로 저장한다 (다음 단계의 입력)
#
# 실행: 프로젝트 폴더(pbmc-scanpy)에서 가상환경을 켠 뒤
#   python scripts\01_qc.py
#
# Seurat 객체 ↔ AnnData 객체
#   Seurat: 행 = 유전자, 열 = 세포,  세포 정보는 pbmc@meta.data
#   AnnData: 행 = 세포,  열 = 유전자, 세포 정보는 adata.obs, 유전자 정보는 adata.var
#   → 행과 열이 R과 반대다. Python 쪽 관례(행 = 관측치)를 따른 것이다.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sp

# ── 경로 ────────────────────────────────────────────────────
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:  # Jupyter 등에서 셀로 실행할 때
    ROOT = Path.cwd()

DATA = ROOT.parent / "pbmc-scrna" / "data" / "filtered_gene_bc_matrices" / "hg19"
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
WORK = ROOT / "work"
for d in (FIG, TAB, WORK):
    d.mkdir(parents=True, exist_ok=True)

# 그림에 한글이 깨지지 않게 (Windows 기본 한글 글꼴)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ── 1. 데이터 읽기 ──────────────────────────────────────────
# 원래는 sc.read_10x_mtx(DATA, var_names="gene_symbols") 한 줄이면 된다.
# 그런데 그 안에서 쓰는 scipy의 빠른 행렬 읽기(C++)가 Windows에서 경로에 한글(박인성, 미니 프로젝트)이
# 있으면 파일을 제대로 열지 못한다 ("Not a Matrix Market file" 오류).
# 그래서 파일을 Python으로 직접 연 뒤 내용을 넘기는 방식으로, read_10x_mtx가 하는 일을 손으로 한다.
with open(DATA / "matrix.mtx", "rb") as f:
    X = scipy.io.mmread(f, spmatrix=False)                   # 유전자 x 세포 (파일에 저장된 모양 그대로, R과 같음)
genes = pd.read_csv(DATA / "genes.tsv", sep="\t", header=None, names=["gene_ids", "symbol"])
barcodes = pd.read_csv(DATA / "barcodes.tsv", sep="\t", header=None)[0]

adata = ad.AnnData(
    X=sp.csr_matrix(X.T, dtype=np.float32),    # 뒤집어서 세포 x 유전자로 (AnnData 관례)
    obs=pd.DataFrame(index=barcodes.values),
    var=pd.DataFrame({"gene_ids": genes["gene_ids"].values}, index=genes["symbol"].values),
)
# 유전자 이름은 ENSG 번호가 아니라 기호(CD3E 등)로 쓴다 (R의 Read10X 기본값과 같음)
adata.var_names_make_unique()  # 같은 기호가 두 번 나오면 뒤에 -1을 붙인다 (Seurat은 .1을 붙임)
print(f"처음: {adata.n_vars:,} 유전자 x {adata.n_obs:,} 세포")

# ── 2. 드문 유전자 / 빈 바코드 빼기 ─────────────────────────
# Seurat의 CreateSeuratObject와 같은 순서: 세포 먼저 거르고, 남은 세포로 유전자를 거른다
sc.pp.filter_cells(adata, min_genes=200)  # 유전자 200개 미만 바코드 제외
sc.pp.filter_genes(adata, min_cells=3)    # 세포 3개 미만에서만 검출된 유전자 제외
print(f"거른 후: {adata.n_vars:,} 유전자 x {adata.n_obs:,} 세포   (R: 13,714 유전자 x 2,700 세포)\n")

# ── 3. QC 지표 ──────────────────────────────────────────────
adata.var["mt"] = adata.var_names.str.startswith("MT-")
print("MT- 유전자 수:", int(adata.var["mt"].sum()), "  (R: 13)\n")

# 세 지표를 한 번에 계산해 adata.obs에 붙인다
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

qc = adata.obs[["n_genes_by_counts", "total_counts", "pct_counts_mt"]].copy()
qc.columns = ["검출 유전자 수", "UMI 수", "MT 비율(%)"]

summ = qc.quantile([0, 0.01, 0.5, 0.99, 1]).T
summ.columns = ["최소", "1%", "중앙값", "99%", "최대"]
print("분포 요약 (Python)")
print(summ.round(2).to_string())
print("\nR 결과 (notes.md Day 1)")
print(pd.DataFrame(
    [[212, 325, 816, 1739, 3400], [546, 659, 2196, 5925, 15818], [0, 0.59, 2.03, 5.88, 22.6]],
    index=summ.index, columns=summ.columns,
).to_string())
print()

# ── 4. 기준 정하기 ──────────────────────────────────────────
# ▶ 직접 정할 값
#   지금 값은 R에서 직접 정한 기준(notes.md Day 1)과 같다.
#   R과 결과를 비교하려면 같은 기준을 써야 하므로 처음에는 그대로 두고,
#   위 분포 요약과 그림이 R과 같은지 확인한 뒤 바꿀지 판단한다.
MIN_FEAT = 200
MAX_FEAT = 2500
MAX_MT = 5

n_feat = adata.obs["n_genes_by_counts"]
pct_mt = adata.obs["pct_counts_mt"]

# R의 case_when과 같은 순서로 판정 (유전자 기준을 먼저 본다)
fail = np.select(
    [n_feat >= MAX_FEAT, pct_mt >= MAX_MT],
    ["유전자 많음 (doublet 의심)", "미토콘드리아 비율 높음"],
    default="통과",
)
adata.obs["qc_fail"] = pd.Categorical(fail)

fail_tab = adata.obs["qc_fail"].value_counts().rename("n").to_frame()
fail_tab["pct"] = (100 * fail_tab["n"] / fail_tab["n"].sum()).round(1)
print("판정 결과 (Python)")
print(fail_tab.to_string())
print("R: 통과 2,638 / MT 높음 57 / 유전자 많음 5\n")
fail_tab.to_csv(TAB / "table1_qc_filter.csv", encoding="utf-8-sig")

# 기준 하나만 바꿨을 때 몇 개가 남는지 (민감도 확인, R의 table2와 같은 격자)
sens = pd.DataFrame(
    [(f, m, int(((n_feat < f) & (pct_mt < m)).sum()))
     for f in (2000, 2500, 3000) for m in (5, 10, 20)],
    columns=["max_feat", "max_mt", "kept"],
)
print("기준별 남는 세포 수")
print(sens.pivot(index="max_feat", columns="max_mt", values="kept").add_prefix("mt<").to_string())
print()
sens.to_csv(TAB / "table2_qc_sensitivity.csv", index=False, encoding="utf-8-sig")

# ── 그림 ────────────────────────────────────────────────────
colors = {"통과": "grey", "유전자 많음 (doublet 의심)": "#D55E00", "미토콘드리아 비율 높음": "#0072B2"}
point_col = adata.obs["qc_fail"].map(colors).astype(str).values
rng = np.random.default_rng(0)

# 그림 1: 바이올린 + 점 (R의 fig1_qc_violin.png와 같은 구성)
fig, axes = plt.subplots(1, 3, figsize=(9, 4))
for ax, col, cuts in zip(axes, qc.columns, [(MIN_FEAT, MAX_FEAT), (), (MAX_MT,)]):
    vals = qc[col].values
    parts = ax.violinplot(vals, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor("lightgrey")
        b.set_edgecolor("grey")
    ax.scatter(1 + rng.uniform(-0.25, 0.25, len(vals)), vals, c=point_col, s=1, alpha=0.6)
    for c in cuts:
        ax.axhline(c, linestyle="--", color="red", linewidth=1)
    ax.set_xticks([])
    ax.set_ylabel(col)
fig.tight_layout()
fig.savefig(FIG / "fig1_qc_violin.png", dpi=150)
plt.close(fig)

# 그림 2: UMI vs 유전자 수, UMI vs MT 비율 (R의 fig2_qc_scatter.png)
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
for ax, ycol, cut in zip(axes, ["검출 유전자 수", "MT 비율(%)"], [MAX_FEAT, MAX_MT]):
    for name, color in colors.items():
        m = adata.obs["qc_fail"] == name
        ax.scatter(qc.loc[m, "UMI 수"], qc.loc[m, ycol], s=3, alpha=0.7, c=color, label=name)
    ax.axhline(cut, linestyle="--", color="red", linewidth=1)
    ax.set_xlabel("UMI 수")
    ax.set_ylabel(ycol)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(FIG / "fig2_qc_scatter.png", dpi=150)
plt.close(fig)

# ── 5. 거르고 저장 ──────────────────────────────────────────
keep = (n_feat > MIN_FEAT) & (n_feat < MAX_FEAT) & (pct_mt < MAX_MT)
adata = adata[keep].copy()
print(f"QC 후: {adata.n_vars:,} 유전자 x {adata.n_obs:,} 세포   (R: 13,714 x 2,638)")
adata.write_h5ad(WORK / "pbmc_qc.h5ad")
print("저장:", WORK / "pbmc_qc.h5ad")
