# 04_annotate.py — 마커 유전자 찾기 → 세포 유형 이름 붙이기, R의 04_annotate.R을 scanpy로 다시 한 것
#
# 입력: work/pbmc_cluster.h5ad (03_cluster.py 결과, R과 같은 군집 9개)
#       ../pbmc-scrna/results/tables/table6_all_markers.csv (R의 FindAllMarkers 결과)
# 출력: 그림 7a, 7b, 7e, 8, 표 6–6b, 8, work/pbmc_annot.h5ad
#
# 실행: python scripts\04_annotate.py
#   → 처음에는 비교 결과까지만 돌고 멈춘다. 표와 그림을 본 뒤 맨 아래 CELL_NAMES를 정하고 다시 실행한다.
#
# 마커 유전자: 한 군집에서 나머지 세포보다 특히 높게 켜진 유전자. 군집 하나 vs 나머지 모든 세포를 비교한다.
#
# 이 단계에서 Seurat(FindAllMarkers)과 scanpy(rank_genes_groups)의 기본값이 다른 곳
#   항목            | Seurat 기본값                                   | scanpy 기본값
#   검정 방법       | Wilcoxon 순위합 검정                             | t-검정
#   배수 차이 계산  | 세포마다 원래 크기로 되돌린 뒤 평균 → log2       | log 값의 평균을 되돌린 뒤 → log2
#   p-value 보정    | Bonferroni (아주 엄격)                           | Benjamini-Hochberg (덜 엄격)
#   걸러 내기       | 25% 이상 세포에서 검출 + log2 배수 0.25 이상     | 걸러 내지 않음 (모든 유전자에 순위)
#                   | + 보정 전 p-value 0.01 미만만                     |
#   순위 기준       | (R 분석에서) 배수 차이가 큰 순서                 | 검정 통계량(점수)이 큰 순서
#
# 그래서 Python을 두 가지로 돌려 R과 비교한다 (앞 단계들과 같은 방식)
#   "default" : rank_genes_groups를 기본값 그대로 → 처음 scanpy를 쓰는 사람이 보는 결과
#   "seurat"  : Wilcoxon + Bonferroni + Seurat과 같은 배수 계산 + 같은 걸러 내기 → R과 같은 방법

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from common import FIG, R_PROJECT, TAB, WORK

R_MARKERS = R_PROJECT / "results" / "tables" / "table6_all_markers.csv"
TOP_N = 8  # R 표 6과 같은 수 (군집마다 상위 8개)

adata = sc.read_h5ad(WORK / "pbmc_cluster.h5ad")
groups = sorted(adata.obs["cluster"].cat.categories, key=int)
print(f"입력: {adata.n_obs:,} 세포, 군집 {len(groups)}개 (설정 {adata.uns['cluster_setting']}, "
      f"resolution {adata.uns['resolution']})   (R: 2,511개, 9개)\n")

# 유전자 이름을 R과 맞춰 둔다 (같은 기호가 두 번 나오면 Seurat은 ".1", scanpy는 "-1")
to_r = dict(zip(adata.var_names, adata.var["r_name"]))

r_all = pd.read_csv(R_MARKERS)
r_all["cluster"] = r_all["cluster"].astype(str)
r_top = (r_all.sort_values("avg_log2FC", ascending=False)
         .groupby("cluster").head(TOP_N))
print(f"R 마커: {len(r_all):,}개 (군집별로 걸러 낸 뒤)\n")

# ── 1. Seurat과 같은 방식의 배수 차이 직접 계산 ─────────────
# Seurat 5: log2( (그 군집 세포들의 원래 크기 합 + 1) ÷ 그 군집 세포 수 )
#         − log2( (나머지 세포들의 원래 크기 합 + 1) ÷ 나머지 세포 수 )
#   "원래 크기" = log1p로 바꾼 값을 expm1로 되돌린 값
#   1을 평균이 아니라 "합"에 더하기 때문에, 실제로 더해지는 양은 1 ÷ 세포 수로 아주 작다.
#   → 나머지 세포에서 거의 0인 유전자는 분모가 0에 가까워져 배수 차이가 아주 크게 나온다.
#   (R 결과와 소수점 여섯째 자리까지 같은 것을 확인했다)
# scanpy: log2( expm1(그 군집의 log 값 평균) ) − log2( expm1(나머지의 log 값 평균) )
#   → 평균을 내는 순서와 더하는 작은 값이 달라서 Seurat과 값이 다르다
X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
E = X.copy()
E.data = np.expm1(E.data)
lab = adata.obs["cluster"].astype(str).to_numpy()
seurat_fc, pct_in, pct_out = {}, {}, {}
detected = (X > 0).astype(np.float32)
for g in groups:
    m = lab == g
    sum_in = np.asarray(E[m].sum(axis=0)).ravel()
    sum_out = np.asarray(E[~m].sum(axis=0)).ravel()
    seurat_fc[g] = np.log2((sum_in + 1) / m.sum()) - np.log2((sum_out + 1) / (~m).sum())
    pct_in[g] = np.asarray(detected[m].mean(axis=0)).ravel()
    pct_out[g] = np.asarray(detected[~m].mean(axis=0)).ravel()

# ── 2. 두 설정으로 마커 찾기 ────────────────────────────────
sc.tl.rank_genes_groups(adata, "cluster", key_added="rgg_default")  # 기본값: t-검정, BH 보정
sc.tl.rank_genes_groups(adata, "cluster", method="wilcoxon", corr_method="bonferroni",
                        tie_correct=True, key_added="rgg_seurat")


def table(key, g):
    df = sc.get.rank_genes_groups_df(adata, g, key=key).set_index("names")
    idx = adata.var_names.get_indexer(df.index)
    df["seurat_log2FC"] = seurat_fc[g][idx]
    df["pct.1"], df["pct.2"] = pct_in[g][idx], pct_out[g][idx]
    df["r_name"] = [to_r[n] for n in df.index]
    return df


py = {"default": {}, "seurat": {}}
for g in groups:
    d = table("rgg_default", g)
    py["default"][g] = d  # 기본값: 걸러 내지 않고 점수 순서 그대로
    s = table("rgg_seurat", g)
    # Seurat의 FindAllMarkers(only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)와 같은 걸러 내기
    #   + FindAllMarkers의 기본 조건 return.thresh = 0.01: 보정 전 p-value가 0.01보다 작은 유전자만 돌려준다
    s = s[(np.maximum(s["pct.1"], s["pct.2"]) >= 0.25) & (s["seurat_log2FC"] >= 0.25) & (s["pvals"] < 0.01)]
    py["seurat"][g] = s.sort_values("seurat_log2FC", ascending=False)  # R 표 6과 같은 순위 기준

labels = {"default": "Python 기본값", "seurat": "Python Seurat 맞춤"}
r_type = adata.obs["r_cell_type"].astype(str)
type_of = {g: r_type[lab == g].mode()[0] for g in groups}

# ── 3. 비교: 군집별 상위 8개와 전체 마커 ─────────────────────
rows, top_rows = [], []
for g in groups:
    r_set_top = list(r_top.query("cluster == @g")["gene"])
    r_set_all = set(r_all.query("cluster == @g")["gene"])
    row = {"군집": g, "R 세포 유형": type_of[g], "R 상위 8개": ", ".join(r_set_top), "R 마커 수": len(r_set_all)}
    for s in ("default", "seurat"):
        top = list(py[s][g]["r_name"][:TOP_N])
        row[f"{labels[s]} 상위 8개"] = ", ".join(top)
        row[f"{labels[s]}: 상위 8개 중 R과 같은 수"] = len(set(top) & set(r_set_top))
        for rank, n in enumerate(top, 1):
            top_rows.append({"설정": labels[s], "군집": g, "R 세포 유형": type_of[g], "순위": rank, "유전자": n})
    s_all = set(py["seurat"][g]["r_name"])
    row["Seurat 맞춤 마커 수"] = len(s_all)
    row["Seurat 맞춤: R 마커와 겹친 수"] = len(s_all & r_set_all)
    rows.append(row)
cmp_tab = pd.DataFrame(rows)

print("군집별 상위 8개 마커 비교 (R은 배수 차이 순서)")
for _, r in cmp_tab.iterrows():
    print(f"\n[{r['군집']}] {r['R 세포 유형']}")
    print(f"  R            : {r['R 상위 8개']}")
    for s in ("default", "seurat"):
        print(f"  {labels[s]:<13}: {r[f'{labels[s]} 상위 8개']}   ← R과 같은 수 {r[f'{labels[s]}: 상위 8개 중 R과 같은 수']}/8")

summary = cmp_tab[["군집", "R 세포 유형", "Python 기본값: 상위 8개 중 R과 같은 수",
                   "Python Seurat 맞춤: 상위 8개 중 R과 같은 수", "R 마커 수", "Seurat 맞춤 마커 수",
                   "Seurat 맞춤: R 마커와 겹친 수"]]
print("\n요약")
print(summary.to_string(index=False))
print(f"\n전체: R 마커 {summary['R 마커 수'].sum():,}개 / Seurat 맞춤 {summary['Seurat 맞춤 마커 수'].sum():,}개 / "
      f"겹침 {summary['Seurat 맞춤: R 마커와 겹친 수'].sum():,}개\n")
cmp_tab.to_csv(TAB / "table6b_marker_compare.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(top_rows).to_csv(TAB / "table6_top_markers.csv", index=False, encoding="utf-8-sig")
pd.concat({g: py["seurat"][g] for g in groups}, names=["cluster"]).to_csv(
    TAB / "table6_all_markers_seurat_style.csv", encoding="utf-8-sig")

# 배수 차이 계산 방식 비교: R 마커 유전자에서 R의 avg_log2FC vs Python의 두 계산
fc = []
for g in groups:
    r_g = r_all.query("cluster == @g").set_index("gene")
    d = py["default"][g].set_index("r_name")
    common = r_g.index.intersection(d.index)
    fc.append(pd.DataFrame({"R": r_g.loc[common, "avg_log2FC"], "seurat_formula": d.loc[common, "seurat_log2FC"],
                            "scanpy_formula": d.loc[common, "logfoldchanges"], "pct.1": d.loc[common, "pct.1"]}))
fc = pd.concat(fc)
print("배수 차이(log2FC) 비교 — R이 마커로 고른 유전자에서")
print(f"  R vs Seurat 방식으로 Python에서 계산: 상관 {fc['R'].corr(fc['seurat_formula']):.3f}, "
      f"평균 차이 {np.mean(np.abs(fc['R'] - fc['seurat_formula'])):.3f}")
print(f"  R vs scanpy 기본 계산              : 상관 {fc['R'].corr(fc['scanpy_formula']):.3f}, "
      f"평균 차이 {np.mean(np.abs(fc['R'] - fc['scanpy_formula'])):.3f}\n")

# ── 그림 ────────────────────────────────────────────────────
# 그림 7e: 배수 차이 계산 방식 비교 (점 색 = 그 군집에서 검출된 세포 비율)
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, col, title in zip(axes, ["seurat_formula", "scanpy_formula"],
                          ["Seurat 방식 계산 (Python)", "scanpy 기본 계산"]):
    sca = ax.scatter(fc["R"], fc[col], s=4, c=fc["pct.1"], cmap="viridis", vmin=0, vmax=1)
    lim = [min(fc["R"].min(), fc[col].min()), max(fc["R"].max(), fc[col].max())]
    ax.plot(lim, lim, "--", color="grey", linewidth=1)
    ax.set_xlabel("R avg_log2FC")
    ax.set_ylabel(f"Python log2FC: {title}")
    ax.set_title(title)
fig.colorbar(sca, ax=axes, shrink=0.8, label="그 군집에서 검출된 세포 비율 (pct.1)")
fig.savefig(FIG / "fig7e_log2fc_compare.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 그림 7a: 군집별 상위 8개 중 R과 같은 수
fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(groups))
for i, s in enumerate(("default", "seurat")):
    ax.bar(x + (i - 0.5) * 0.38, cmp_tab[f"{labels[s]}: 상위 8개 중 R과 같은 수"], width=0.38, label=labels[s])
ax.set_xticks(x)
ax.set_xticklabels([f"{g}\n{type_of[g]}" for g in groups], fontsize=8)
ax.set_ylim(0, 8.5)
ax.set_ylabel("상위 8개 중 R과 같은 유전자 수")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "fig7a_top_marker_overlap.png", dpi=150)
plt.close(fig)

# 그림 7b: 알려진 마커 dot plot (R 그림 7b와 같은 유전자, 같은 순서)
#   점 크기 = 그 군집에서 검출된 세포 비율, 점 색 = 평균 발현 (유전자마다 0–1로 맞춤)
known = ["CD3E", "IL7R", "CCR7", "S100A4", "CD8A", "GNLY", "NKG7", "MS4A1", "CD79A",
         "CD14", "LYZ", "FCGR3A", "MS4A7", "FCER1A", "CST3", "PPBP"]
dp = sc.pl.dotplot(adata, known, groupby="cluster", standard_scale="var", show=False, return_fig=True)
dp.savefig(FIG / "fig7b_known_markers_dot.png", dpi=150, bbox_inches="tight")
plt.close("all")

print("그림 저장: fig7a_top_marker_overlap, fig7b_known_markers_dot, fig7e_log2fc_compare")

# ═══ ▶ 직접 정할 값 ═════════════════════════════════════════
# 군집마다 세포 이름을 정한다. 위에 출력된 상위 마커와 그림 7b(알려진 마커)를 보고 정한다.
# 쓸 수 있는 이름 예: "Naive CD4 T", "Memory CD4 T", "CD8 T", "NK", "B",
#                     "CD14+ Mono", "FCGR3A+ Mono", "DC", "Platelet"
# 두 군집이 같은 세포라고 판단하면 같은 이름을 넣는다. 판단이 어려우면 "Unknown"이라고 넣어도 된다.
CELL_NAMES = {
    "0": "Naive CD4 T",
    "1": "CD14+ Mono",
    "2": "Memory CD4 T",
    "3": "B",
    "4": "CD8 T",
    "5": "FCGR3A+ Mono",
    "6": "NK",
    "7": "DC",
    "8": "Platelet",
}

if any(v is None for v in CELL_NAMES.values()):
    print("\n▶ 여기서 멈춤: 출력과 그림을 본 뒤 스크립트 맨 아래 CELL_NAMES의 None을 세포 이름으로 바꾸고 다시 실행하세요.")
    sys.exit()

adata.obs["cell_type"] = pd.Categorical(adata.obs["cluster"].astype(str).map(CELL_NAMES))
same = (adata.obs["cell_type"].astype(str) == r_type).mean()
print(f"\nR의 세포 유형 이름과 같은 세포: {100 * same:.1f}%")

type_tab = adata.obs["cell_type"].value_counts().rename("세포수").to_frame()
type_tab["비율"] = (100 * type_tab["세포수"] / adata.n_obs).round(1)
print(type_tab.to_string())
type_tab.to_csv(TAB / "table8_cell_types.csv", encoding="utf-8-sig")

# 그림 8: 이름 붙인 UMAP
fig, ax = plt.subplots(figsize=(7, 6))
xy = adata.obsm["X_umap"]
for j, t in enumerate(sorted(adata.obs["cell_type"].cat.categories)):
    m = (adata.obs["cell_type"] == t).to_numpy()
    ax.scatter(xy[m, 0], xy[m, 1], s=3, color=plt.get_cmap("tab10")(j))
    ax.text(*np.median(xy[m], axis=0), t, fontsize=9, weight="bold")
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("PBMC 세포 유형 (Python)")
fig.tight_layout()
fig.savefig(FIG / "fig8_umap_celltypes.png", dpi=150)
plt.close(fig)

for key in ("rgg_default", "rgg_seurat"):  # 저장 파일을 가볍게
    adata.uns.pop(key, None)
adata.write_h5ad(WORK / "pbmc_annot.h5ad")
print("\n저장: work/pbmc_annot.h5ad")
