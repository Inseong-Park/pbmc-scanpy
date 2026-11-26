# 02_pca.py — 정규화 → 변이 큰 유전자(HVG) → 스케일링 → PCA, R의 02_pca.R을 scanpy로 다시 한 것
#
# 입력: work/pbmc_qc_dbl.h5ad (01b_doublet.py 결과, R과 같은 2,511개 세포)
#       work/r_export/ (R 결과를 CSV로 내보낸 것. 없으면 먼저 R\export_seurat.R 실행)
# 출력: 그림 3, 4a–4d, 표 3–3c, work/pbmc_pca.h5ad
#
# 실행: python scripts\02_pca.py
#   → 처음에는 비교 결과까지만 돌고 멈춘다. 그림과 표를 본 뒤 맨 아래 SETTING과 N_PCS를 정하고 다시 실행한다.
#
# 이번 단계의 비교 설계: Python을 두 가지 설정으로 돌려 둘 다 R과 비교한다
#   "default" : scanpy를 아무 설정 없이 기본값으로 쓴 경우
#   "seurat"  : scanpy 설정을 Seurat(R)과 최대한 같게 맞춘 경우
#   → R vs "seurat"의 차이 = 같은 방법인데 구현(코드)이 달라서 생기는 차이
#     R vs "default"의 차이 = 위 차이 + 기본 설정이 달라서 생기는 차이
#
# 두 설정이 다른 곳 (R은 "seurat"과 같음)
#   단계             | default (scanpy 기본값)                    | seurat (Seurat에 맞춤)
#   정규화 목표 총량 | 세포 UMI 총량의 중앙값 (이 데이터 약 2,200) | 10,000
#   HVG 고르는 방법  | flavor="seurat": log 값의 분산/평균(dispersion), 기준값을 넘는 유전자 모두 | flavor="seurat_v3": 원래 UMI 개수로 Seurat vst와 같은 방식, 2,000개
#   스케일링 상한    | 없음                                       | 10 (Seurat ScaleData와 같음)

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from common import DATA, FIG, TAB, WORK

R_EXP = WORK / "r_export"
if not (R_EXP / "hvg.csv").exists():
    sys.exit(r'R 결과가 없습니다. 먼저 실행하세요: & "C:\Program Files\R\R-4.4.3\bin\Rscript.exe" R\export_seurat.R')

N_SHOW = 15  # 비교하고 그림으로 볼 PC 개수 (R에서 히트맵으로 본 개수와 같음)

# ── 입력 ────────────────────────────────────────────────────
base = sc.read_h5ad(WORK / "pbmc_qc_dbl.h5ad")
base.layers["counts"] = base.X.copy()  # 원래 UMI 개수를 따로 보관
print(f"입력: {base.n_vars:,} 유전자 x {base.n_obs:,} 세포   (R: 13,714 x 2,511)\n")

# 유전자 이름을 R과 맞춘다: 같은 기호가 두 번 나오면 scanpy는 "-1", Seurat은 ".1"을 붙인다
genes = pd.read_csv(DATA / "genes.tsv", sep="\t", header=None, names=["gene_ids", "symbol"])
symbol = dict(zip(genes["gene_ids"], genes["symbol"]))


def to_r_name(py_name, gene_id):
    s = symbol[gene_id]
    r = s.replace("_", "-")  # Seurat은 유전자 이름의 _를 -로 바꾼다
    return r if py_name == s else f"{r}.{py_name[len(s) + 1:]}"


base.var["r_name"] = [to_r_name(n, g) for n, g in zip(base.var_names, base.var["gene_ids"])]

# R 결과
r_hvg = pd.read_csv(R_EXP / "hvg.csv")
r_emb = pd.read_csv(R_EXP / "pca_embeddings.csv", index_col=0).loc[base.obs_names]
r_sd = pd.read_csv(R_EXP / "pca_stdev.csv")["stdev"].to_numpy()
r_load = pd.read_csv(R_EXP / "pca_loadings.csv", index_col=0)
r_type = pd.read_csv(R_EXP / "cell_types.csv", index_col="barcode").loc[base.obs_names, "cell_type"]


# ── 두 설정으로 실행 ────────────────────────────────────────
def run(setting):
    a = base.copy()
    if setting == "default":
        sc.pp.normalize_total(a)                 # 목표 총량 = 세포 UMI 총량의 중앙값
        sc.pp.log1p(a)
        sc.pp.highly_variable_genes(a)           # flavor="seurat", 기준값(min_mean, max_mean, min_disp)으로 고름
        h = a[:, a.var["highly_variable"]].copy()
        sc.pp.scale(h)                           # 상한 없음
        a.var["hvg_score"] = a.var["dispersions_norm"]
    else:
        sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=2000)  # 정규화 전 UMI 개수로 (Seurat vst와 같음)
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        h = a[:, a.var["highly_variable"]].copy()
        sc.pp.scale(h, max_value=10)
        a.var["hvg_score"] = a.var["variances_norm"]
    sc.tl.pca(h, n_comps=50, random_state=0)
    return a, h


runs = {s: run(s) for s in ("default", "seurat")}
labels = {"R": "R (Seurat)", "default": "Python 기본값", "seurat": "Python Seurat 맞춤"}

# ── 1. 정규화 확인 ──────────────────────────────────────────
# "seurat" 설정은 Seurat의 LogNormalize(총량 10,000 → log(1+x))와 같은 계산이다.
# 한 유전자로 직접 확인: LYZ 값이 가장 큰 세포
a_s = runs["seurat"][0]
lyz = a_s[:, "LYZ"].layers["counts"].toarray().ravel()
i = int(lyz.argmax())
tot = float(a_s.layers["counts"][i].sum())
print("정규화 확인 (LYZ가 가장 많은 세포)")
print(f"  UMI {lyz[i]:.0f}개 / 총 {tot:,.0f}개 → log(1 + {lyz[i]:.0f}/{tot:,.0f} x 10,000) = {np.log1p(lyz[i] / tot * 1e4):.3f}")
print(f"  Python seurat 설정 값: {a_s[i, 'LYZ'].X.toarray().item():.3f}")
med = float(np.median(np.asarray(base.layers["counts"].sum(axis=1)).ravel()))
print(f"  Python 기본값은 10,000 대신 UMI 총량 중앙값 {med:,.0f}으로 맞춘다\n")

# ── 2. 변이가 큰 유전자 비교 ────────────────────────────────
r_set = set(r_hvg["gene"])
r_top10 = list(r_hvg["gene"][:10])
print(f"R 상위 10개: {', '.join(r_top10)}")
hvg_rows = []
for s, (a, h) in runs.items():
    hv = a.var[a.var["highly_variable"]].sort_values("hvg_score", ascending=False)
    py_set = set(hv["r_name"])
    both = len(py_set & r_set)
    top10 = list(hv["r_name"][:10])
    print(f"{labels[s]} 상위 10개: {', '.join(top10)}")
    hvg_rows.append({
        "설정": labels[s], "HVG 수": len(py_set), "R과 겹침": both,
        "R 2,000개 중 겹친 비율(%)": round(100 * both / len(r_set), 1),
        "상위 10개 중 R 상위 10개와 같은 수": len(set(top10) & set(r_top10)),
    })
hvg_tab = pd.DataFrame(hvg_rows)
print("\n변이가 큰 유전자 비교 (R: 2,000개)")
print(hvg_tab.to_string(index=False))
print()
hvg_tab.to_csv(TAB / "table3b_hvg_overlap.csv", index=False, encoding="utf-8-sig")

# ── 3. PCA 비교 ─────────────────────────────────────────────
def var_pct(v):
    return 100 * v / v.sum()  # R과 같게 "PC 50개 안에서"의 비율


vp = pd.DataFrame({"PC": np.arange(1, 51), "R": var_pct(r_sd ** 2)})
for s, (a, h) in runs.items():
    vp[s] = var_pct(h.uns["pca"]["variance"])
vp.round(3).to_csv(TAB / "table3_pca_variance.csv", index=False, encoding="utf-8-sig")

R = r_emb.to_numpy()[:, :N_SHOW]
corr = {}
pc_rows = []
for s, (a, h) in runs.items():
    P = h.obsm["X_pca"][:, :N_SHOW]
    C = np.corrcoef(R.T, P.T)[:N_SHOW, N_SHOW:]  # 행 = R의 PC, 열 = Python의 PC
    corr[s] = C
    load = pd.DataFrame(h.varm["PCs"][:, :N_SHOW], index=h.var["r_name"])
    for k in range(N_SHOW):
        r_top = set(r_load.iloc[:, k].abs().nlargest(10).index)
        p_top = set(load.iloc[:, k].abs().nlargest(10).index)
        pc_rows.append({"설정": s, "PC": k + 1, "|상관|": abs(C[k, k]),
                        "로딩 상위 10개 겹침": len(r_top & p_top)})
pc_tab = pd.DataFrame(pc_rows).pivot(index="PC", columns="설정")
pc_tab.columns = [f"{m} ({labels[s]})" for m, s in pc_tab.columns]
pc_tab.insert(0, "설명 분산 R(%)", vp["R"][:N_SHOW].round(2).to_numpy())
pc_tab.insert(1, "Python 기본값(%)", vp["default"][:N_SHOW].round(2).to_numpy())
pc_tab.insert(2, "Seurat 맞춤(%)", vp["seurat"][:N_SHOW].round(2).to_numpy())
print("PC별 비교 (|상관|: 같은 번호 PC끼리 세포 점수의 상관, 1이면 같은 축. 부호는 무작위라 절댓값으로 본다)")
print(pc_tab.round(3).to_string())
print()
pc_tab.round(3).to_csv(TAB / "table3c_pc_compare.csv", encoding="utf-8-sig")

# ── 그림 ────────────────────────────────────────────────────
# 그림 3: 변이가 큰 유전자 (설정마다)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, s, ylab in zip(axes, runs, ["표준화한 dispersion", "표준화한 분산 (Seurat vst와 같은 값)"]):
    v = runs[s][0].var
    hv = v["highly_variable"]
    x = v["means"] if s == "default" else np.log10(v["means"] + 1e-6)
    ax.scatter(x[~hv], v.loc[~hv, "hvg_score"], s=2, c="lightgrey", label="나머지")
    ax.scatter(x[hv], v.loc[hv, "hvg_score"], s=2, c="#D55E00", label=f"HVG {int(hv.sum()):,}개")
    for g in v[hv].nlargest(10, "hvg_score").index:
        ax.annotate(g, (x[g], v.loc[g, "hvg_score"]), fontsize=7)
    ax.set_xlabel("평균 발현량" + (" (log 정규화 값)" if s == "default" else " (log10 UMI)"))
    ax.set_ylabel(ylab)
    ax.set_title(labels[s])
    ax.legend(markerscale=4)
fig.tight_layout()
fig.savefig(FIG / "fig3_variable_genes.png", dpi=150)
plt.close(fig)

# 그림 4a: elbow plot 세 개 겹쳐 그리기
fig, ax = plt.subplots(figsize=(7, 4))
for col, color in [("R", "black"), ("default", "#56B4E9"), ("seurat", "#D55E00")]:
    ax.plot(vp["PC"], vp[col], marker="o", ms=3, color=color, label=labels[col])
ax.set_xticks(range(0, 51, 5))
ax.set_xlabel("주성분(PC)")
ax.set_ylabel("설명 분산 (%, PC 50개 안에서)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "fig4a_elbow.png", dpi=150)
plt.close(fig)

# 그림 4b: PC1–PC2, R의 최종 세포 유형으로 색칠
#   PC의 부호(+/-)는 계산할 때마다 임의로 정해지므로, 비교하기 쉽게 Python PC의 부호를 R에 맞춰 뒤집었다
types = sorted(r_type.unique())
cmap = plt.get_cmap("tab10")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
panels = [("R", R[:, :2])] + [(s, runs[s][1].obsm["X_pca"][:, :2] * np.sign(np.diag(corr[s])[:2])) for s in runs]
for ax, (s, xy) in zip(axes, panels):
    for j, t in enumerate(types):
        m = (r_type == t).to_numpy()
        ax.scatter(xy[m, 0], xy[m, 1], s=3, color=cmap(j), label=t)
    ax.set_title(labels[s])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
handles, lab = axes[0].get_legend_handles_labels()
fig.legend(handles, lab, loc="lower center", ncol=5, frameon=False, markerscale=4)
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(FIG / "fig4b_pca_pc1_pc2.png", dpi=150)
plt.close(fig)


# 그림 4c: PC별 히트맵 (R의 DimHeatmap과 같은 그림) — 설정마다 한 장
#   세포: PC 점수가 가장 낮은 250개 + 가장 높은 250개를 점수 순서로
#   유전자: 로딩이 가장 큰 15개 + 가장 작은 15개
#   뒤쪽 PC로 갈수록 위아래 무늬가 흐려지면 그 PC는 잡음에 가깝다
def pc_heatmaps(h, path, title):
    X = np.asarray(h.X)
    load = h.varm["PCs"]
    fig, axes = plt.subplots(5, 3, figsize=(12, 20))
    for k, ax in enumerate(axes.flat):
        order = np.argsort(h.obsm["X_pca"][:, k])
        cells = np.r_[order[:250], order[-250:]]
        g = np.argsort(load[:, k])
        gsel = np.r_[g[-15:][::-1], g[:15][::-1]]
        ax.imshow(np.clip(X[np.ix_(cells, gsel)].T, -2.5, 2.5), aspect="auto", cmap="RdBu_r",
                  vmin=-2.5, vmax=2.5, interpolation="nearest")
        ax.set_yticks(range(len(gsel)))
        ax.set_yticklabels(h.var_names[gsel], fontsize=5)
        ax.set_xticks([])
        ax.set_title(f"PC{k + 1}", fontsize=10)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=120)
    plt.close(fig)


for s in runs:
    pc_heatmaps(runs[s][1], FIG / f"fig4c_pc_heatmaps_{s}.png", labels[s])

# 그림 4d: R의 PC와 Python의 PC 사이 상관 (|r|). 대각선이 진하면 같은 번호 PC가 같은 축이다.
#   대각선 밖에 진한 칸이 있으면 순서가 바뀐 것, 한 줄에 여러 칸이 옅게 퍼져 있으면 축이 섞인 것
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, s in zip(axes, runs):
    im = ax.imshow(np.abs(corr[s]), vmin=0, vmax=1, cmap="Greys")
    ax.set_xticks(range(N_SHOW))
    ax.set_xticklabels(range(1, N_SHOW + 1), fontsize=7)
    ax.set_yticks(range(N_SHOW))
    ax.set_yticklabels(range(1, N_SHOW + 1), fontsize=7)
    ax.set_xlabel(f"{labels[s]} PC")
    ax.set_ylabel("R PC")
fig.colorbar(im, ax=axes, shrink=0.8, label="|상관|")
fig.savefig(FIG / "fig4d_pc_correlation.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("그림 저장: fig3_variable_genes, fig4a_elbow, fig4b_pca_pc1_pc2, fig4c_pc_heatmaps_default/seurat, fig4d_pc_correlation")

# ═══ ▶ 직접 정할 값 ═════════════════════════════════════════
# SETTING: 다음 단계(군집, UMAP)에 넘길 설정 — "default" 또는 "seurat"
#   "seurat"  : R과 같은 방법. 다음 단계의 차이를 군집화 방법의 차이로만 볼 수 있다.
#   "default" : scanpy를 기본값으로 쓴 결과. 처음 scanpy를 쓰는 사람이 얻는 결과에 가깝다.
# N_PCS: 다음 단계에 쓸 PC 개수 (R에서는 9개)
#   그림 4a(elbow), 4c(히트맵), 표 3c를 보고 정한다.
SETTING = "seurat"
N_PCS = 9

if SETTING is None or N_PCS is None:
    print("\n▶ 여기서 멈춤: 그림과 표를 본 뒤 스크립트 맨 아래 SETTING과 N_PCS를 정하고 다시 실행하세요.")
    sys.exit()
if SETTING not in runs:
    sys.exit('SETTING은 "default" 또는 "seurat"이어야 합니다.')

a, h = runs[SETTING]
a.obsm["X_pca"] = h.obsm["X_pca"]
a.uns["pca"] = {"variance": h.uns["pca"]["variance"], "variance_ratio": h.uns["pca"]["variance_ratio"]}
a.uns["n_pcs"] = int(N_PCS)
a.uns["pca_setting"] = SETTING
a.obs["r_cell_type"] = pd.Categorical(r_type.to_numpy())  # 나중에 R 결과와 비교할 때 쓴다
a.write_h5ad(WORK / "pbmc_pca.h5ad")
print(f"\n저장: work/pbmc_pca.h5ad (설정 {SETTING}, PC {N_PCS}개 사용)   (R: PC 9개)")
