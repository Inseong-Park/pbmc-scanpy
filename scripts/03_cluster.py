# 03_cluster.py — 이웃 그래프 → 군집 → UMAP, R의 03_cluster.R을 scanpy로 다시 한 것
#
# 입력: work/pbmc_pca.h5ad (02_pca.py 결과, R과 같은 PCA, PC 9개 사용)
#       work/r_export/ (R의 군집·UMAP. 없거나 오래됐으면 R\export_seurat.R을 다시 실행)
# 출력: 그림 5a, 5e, 6, 표 4–4c, work/pbmc_cluster.h5ad
#
# 실행: python scripts\03_cluster.py
#   → 처음에는 비교 결과까지만 돌고 멈춘다. 그림과 표를 본 뒤 맨 아래 세 값을 정하고 다시 실행한다.
#
# 이 단계에서 Seurat과 scanpy가 다른 곳은 두 군데다
#   ① 이웃 그래프: 세포끼리 누가 누구와 얼마나 강하게 이어졌나
#      Seurat : 이웃 20개, "이웃을 얼마나 공유하나"(SNN, Jaccard)로 연결 세기를 정한다
#      scanpy : 이웃 15개, UMAP과 같은 방식(거리가 가까울수록 강하게)으로 연결 세기를 정한다
#   ② 군집 알고리즘: 그래프에서 촘촘한 무리를 찾는 방법
#      Seurat : Louvain
#      scanpy : Leiden (Louvain을 고친 방법. 무리 안이 끊어져 있는 군집이 생기지 않게 보장한다)
#
# 그래서 Python을 네 가지로 돌려 R과 비교한다 (앞 단계처럼 차이를 나눠 보기 위해)
#   "default"    : scanpy 기본값                       (그래프 ①도 다르고 알고리즘 ②도 다름)
#   "snn_leiden" : Seurat 방식 그래프 + Leiden         (그래프는 R과 같은 방식, 알고리즘만 다름)
#   "seurat"     : Seurat 방식 그래프 + Louvain        (둘 다 R과 같은 방식 → 남는 차이는 구현 차이)
#   "r_graph"    : R이 실제로 만든 그래프 + Louvain    (그래프까지 R과 똑같음 → 남는 차이는 Louvain 프로그램 차이)
#   → "snn_leiden" vs "default"  = 이웃 그래프 방식의 차이
#     "seurat" vs "snn_leiden"   = 군집 알고리즘(Louvain vs Leiden)의 차이
#     "r_graph" vs "seurat"      = 그래프를 만드는 구현(근사 vs 정확한 이웃 찾기)의 차이
#     R vs "r_graph"             = Louvain을 돌리는 프로그램(Seurat vs igraph)의 차이

import random
import sys

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from common import FIG, TAB, WORK

R_EXP = WORK / "r_export"
if not (R_EXP / "clusters.csv").exists():
    sys.exit(r'R 군집 결과가 없습니다. 먼저 실행하세요: & "C:\Program Files\R\R-4.4.3\bin\Rscript.exe" R\export_seurat.R')

RES_TRY = [0.3, 0.5, 0.8, 1.2]  # R과 같은 네 값

adata = sc.read_h5ad(WORK / "pbmc_pca.h5ad")
n_pcs = int(adata.uns["n_pcs"])
X = adata.obsm["X_pca"][:, :n_pcs]
print(f"입력: {adata.n_obs:,} 세포, PC {n_pcs}개 (설정 {adata.uns['pca_setting']})   (R: 2,511개, PC 9개)\n")

r_clu = pd.read_csv(R_EXP / "clusters.csv", index_col="barcode").loc[adata.obs_names]
r_umap = pd.read_csv(R_EXP / "umap.csv", index_col=0).loc[adata.obs_names].to_numpy()
r_type = adata.obs["r_cell_type"].astype(str)


def by_size(labels):
    """군집 번호를 큰 군집부터 0, 1, 2…로 다시 붙인다 (Seurat과 같은 규칙)"""
    s = pd.Series(labels).astype(str)
    order = s.value_counts().index
    return s.map({c: str(i) for i, c in enumerate(order)}).to_numpy()


# ── 1. 이웃 그래프 ──────────────────────────────────────────
# (가) Seurat 방식 SNN 그래프를 직접 만든다 (FindNeighbors와 같은 계산)
#   1) PC 공간에서 세포마다 가장 가까운 20개를 찾는다 (자기 자신 포함, Seurat과 같음)
#   2) 두 세포가 이웃 목록을 얼마나 공유하나 = 공유한 수 ÷ 합친 목록의 수 (Jaccard)
#   3) 1/15보다 작은 연결은 지운다 (Seurat의 prune.SNN 기본값)
K = 20
nn_idx = NearestNeighbors(n_neighbors=K).fit(X).kneighbors(X, return_distance=False)
M = sp.csr_matrix((np.ones(nn_idx.size), (np.repeat(np.arange(len(X)), K), nn_idx.ravel())), shape=(len(X), len(X)))
shared = (M @ M.T).tocoo()
jac = shared.data / (2 * K - shared.data)
keep = (jac >= 1 / 15) & (shared.row != shared.col)
snn = sp.csr_matrix((jac[keep], (shared.row[keep], shared.col[keep])), shape=(len(X), len(X)))
print(f"Seurat 방식 SNN 그래프: 연결 {snn.nnz // 2:,}개 (세포당 평균 {snn.nnz / len(X):.1f}개)")

# (나) R에서 Seurat이 실제로 만든 SNN 그래프를 그대로 읽는다
#   Seurat은 이웃을 빠른 근사 방법(annoy)으로 찾고, 위 (가)는 정확한 방법으로 찾는다.
#   두 그래프가 얼마나 같은지 확인하고, R 그래프로도 군집을 나눠 본다.
r_cells = (R_EXP / "snn_cells.txt").read_text(encoding="utf-8").split()
g = pd.read_csv(R_EXP / "snn_graph.csv")
pos = pd.Series(np.arange(adata.n_obs), index=adata.obs_names).loc[r_cells].to_numpy()  # R 순서 → Python 순서
off = g["i"] != g["j"]
r_snn = sp.csr_matrix((g["w"][off], (pos[g["i"][off] - 1], pos[g["j"][off] - 1])), shape=snn.shape)
same_edge = snn.multiply(r_snn > 0).nnz
w_py, w_r = np.asarray(snn[r_snn.nonzero()]).ravel(), np.asarray(r_snn[r_snn.nonzero()]).ravel()
print(f"R(Seurat) SNN 그래프:   연결 {r_snn.nnz // 2:,}개 / Python (가)와 같은 연결 {100 * same_edge / r_snn.nnz:.1f}%, "
      f"연결 세기 상관 {np.corrcoef(w_py, w_r)[0, 1]:.3f}")

# (다) scanpy 기본 이웃 그래프 (이웃 15개)
sc.pp.neighbors(adata, n_pcs=n_pcs, random_state=0)
print(f"scanpy 기본 그래프:      연결 {adata.obsp['connectivities'].nnz // 2:,}개 "
      f"(세포당 평균 {adata.obsp['connectivities'].nnz / len(X):.1f}개)\n")


# ── 2. 세 설정 × resolution 네 개로 군집 ────────────────────
def louvain(adj, res):
    random.seed(0)  # igraph는 Python의 random을 쓴다 → 결과 고정
    g = ig.Graph.Weighted_Adjacency(sp.triu(adj).tocoo(), mode="undirected")
    return g.community_multilevel(weights="weight", resolution=res).membership


labels = {"R": "R (Seurat)", "default": "Python 기본값", "snn_leiden": "Seurat 그래프 + Leiden",
          "seurat": "Seurat 맞춤 (그래프 + Louvain)", "r_graph": "R의 그래프 + Python Louvain"}
PY = ("default", "snn_leiden", "seurat", "r_graph")
clusters = {"R": {r: r_clu[f"res_{r}"].astype(str).to_numpy() for r in RES_TRY}}
for s in PY:
    clusters[s] = {}
    for r in RES_TRY:
        if s == "default":
            sc.tl.leiden(adata, resolution=r, key_added="tmp", random_state=0)
            lab = adata.obs["tmp"].to_numpy()
        elif s == "snn_leiden":
            sc.tl.leiden(adata, resolution=r, adjacency=snn, key_added="tmp", random_state=0)
            lab = adata.obs["tmp"].to_numpy()
        elif s == "seurat":
            lab = louvain(snn, r)
        else:
            lab = louvain(r_snn, r)
        clusters[s][r] = by_size(lab)
        adata.obs[f"{s}_res_{r}"] = pd.Categorical(clusters[s][r])
del adata.obs["tmp"]

# 표 4: resolution별 군집 수와 R과의 일치도
#   ARI (adjusted Rand index): 두 군집 결과가 "같은 세포끼리 같은 군집에 넣었나"를 0–1로 잰 값
#     1 = 완전히 같게 나눔, 0 = 우연 수준. 번호가 달라도 묶음이 같으면 1이다.
rows = []
for s in clusters:
    for r in RES_TRY:
        lab = clusters[s][r]
        size = pd.Series(lab).value_counts()
        rows.append({"설정": labels[s], "resolution": r, "군집 수": len(size),
                     "가장 작은 군집": int(size.min()), "가장 큰 군집": int(size.max()),
                     "R 같은 resolution과 ARI": round(adjusted_rand_score(clusters["R"][r], lab), 3),
                     "R 최종 세포 유형과 ARI": round(adjusted_rand_score(r_type, lab), 3)})
res_tab = pd.DataFrame(rows)
print("resolution별 군집 (ARI: 1 = R과 같은 세포끼리 같은 군집, 0 = 우연 수준)")
print(res_tab.to_string(index=False))
print()
res_tab.to_csv(TAB / "table4_resolution.csv", index=False, encoding="utf-8-sig")

# 표 4b: resolution 0.5에서 R의 군집 하나하나가 Python에서 어떻게 나뉘었나
#   "한 군집에 모인 비율": R 군집의 세포 중 가장 많이 간 Python 군집 하나에 들어간 비율 (100%면 그대로 유지)
r05 = pd.Series(clusters["R"][0.5], index=adata.obs_names)
match_rows = []
for c in sorted(r05.unique(), key=int):
    m = (r05 == c).to_numpy()
    row = {"R 군집 (0.5)": c, "R 세포 유형": r_type[m].mode()[0], "세포 수": int(m.sum())}
    for s in PY:
        vc = pd.Series(clusters[s][0.5][m]).value_counts()
        row[f"{labels[s]}: 한 군집에 모인 비율(%)"] = round(100 * vc.iloc[0] / m.sum(), 1)
        row[f"{labels[s]}: 그 군집"] = vc.index[0]
    match_rows.append(row)
match_tab = pd.DataFrame(match_rows)
print("resolution 0.5에서 R 군집이 Python에서 그대로 유지됐나")
print(match_tab.to_string(index=False))
print()
match_tab.to_csv(TAB / "table4b_cluster_match.csv", index=False, encoding="utf-8-sig")

# ── 3. UMAP ────────────────────────────────────────────────
# "default": scanpy 기본값 (위 이웃 15개 그래프, min_dist 0.5)
# "seurat" : Seurat RunUMAP 기본값에 맞춤 (이웃 30개, cosine 거리, min_dist 0.3)
#   단, R은 uwot, Python은 umap-learn이라는 다른 프로그램으로 계산한다
sc.tl.umap(adata, random_state=0, key_added="X_umap_default")
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=n_pcs, metric="cosine", key_added="umap30", random_state=0)
sc.tl.umap(adata, min_dist=0.3, neighbors_key="umap30", random_state=0, key_added="X_umap_seurat")
# 비교 기준: 같은 설정에서 무작위 시작값(seed)만 바꾼 UMAP
sc.tl.umap(adata, min_dist=0.3, neighbors_key="umap30", random_state=1, key_added="X_umap_seurat_seed1")
umaps = {"R": r_umap, "default": adata.obsm["X_umap_default"], "seurat": adata.obsm["X_umap_seurat"]}


def neighbor_agreement(a, b, k=15):
    """두 UMAP 그림에서 세포마다 가장 가까운 k개 이웃이 얼마나 같은가 (평균 Jaccard, 1 = 완전히 같음)"""
    na = NearestNeighbors(n_neighbors=k + 1).fit(a).kneighbors(a, return_distance=False)[:, 1:]
    nb = NearestNeighbors(n_neighbors=k + 1).fit(b).kneighbors(b, return_distance=False)[:, 1:]
    return float(np.mean([len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(na, nb)]))


umap_tab = pd.DataFrame([
    {"비교": "R vs Python 기본값", "이웃 일치율": neighbor_agreement(r_umap, umaps["default"])},
    {"비교": "R vs Python Seurat 맞춤", "이웃 일치율": neighbor_agreement(r_umap, umaps["seurat"])},
    {"비교": "Python Seurat 맞춤: seed 0 vs seed 1 (같은 설정, 시작값만 다름)",
     "이웃 일치율": neighbor_agreement(umaps["seurat"], adata.obsm["X_umap_seurat_seed1"])},
]).round(3)
print("UMAP 비교 (세포마다 그림 위 가장 가까운 15개 이웃이 얼마나 같은가, 1 = 완전히 같음)")
print(umap_tab.to_string(index=False))
print()
umap_tab.to_csv(TAB / "table4c_umap_neighbors.csv", index=False, encoding="utf-8-sig")

# ── 그림 ────────────────────────────────────────────────────
cmap = plt.get_cmap("tab20")

# 그림 5a: 모든 군집 결과를 R의 UMAP 위에 칠한다 → 그림은 같고 색(군집)만 다르다
fig, axes = plt.subplots(5, 4, figsize=(16, 20))
for i, s in enumerate(clusters):
    for j, r in enumerate(RES_TRY):
        ax = axes[i, j]
        lab = clusters[s][r]
        for c in sorted(set(lab), key=int):
            m = lab == c
            ax.scatter(r_umap[m, 0], r_umap[m, 1], s=2, color=cmap(int(c) % 20))
            ax.text(*np.median(r_umap[m], axis=0), c, fontsize=8, weight="bold")
        ari = res_tab.query("설정 == @labels[@s] and resolution == @r")["R 같은 resolution과 ARI"].item()
        ax.set_title(f"{labels[s]}\nres {r}: {len(set(lab))}개" + ("" if s == "R" else f", ARI {ari:.2f}"), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
fig.suptitle("모든 군집 결과를 R의 UMAP 위에 칠함 (행 = 설정, 열 = resolution)")
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(FIG / "fig5a_clusters_on_r_umap.png", dpi=120)
plt.close(fig)

# 그림 5e: resolution 0.5에서 R 군집(행)의 세포가 Python 군집(열)으로 몇 개씩 갔나
r_names = {c: f"{c} {r_type[(r05 == c).to_numpy()].mode()[0]}" for c in r05.unique()}
fig, axes = plt.subplots(2, 2, figsize=(16, 13))
for ax, s in zip(axes.flat, PY):
    tab = pd.crosstab(pd.Series(clusters["R"][0.5]).map(r_names), pd.Series(clusters[s][0.5]))
    tab = tab.loc[sorted(tab.index, key=lambda x: int(x.split()[0])), sorted(tab.columns, key=int)]
    ax.imshow(tab.to_numpy(), cmap="Blues", aspect="auto")
    for (y, x), v in np.ndenumerate(tab.to_numpy()):
        if v > 0:
            ax.text(x, y, v, ha="center", va="center", fontsize=7)
    ax.set_xticks(range(tab.shape[1]))
    ax.set_xticklabels(tab.columns)
    ax.set_yticks(range(tab.shape[0]))
    ax.set_yticklabels(tab.index, fontsize=8)
    ax.set_xlabel(f"{labels[s]} 군집 (res 0.5)")
    ax.set_ylabel("R 군집 (res 0.5)")
fig.tight_layout()
fig.savefig(FIG / "fig5e_cluster_match.png", dpi=150)
plt.close(fig)

# 그림 6: UMAP 세 개를 R의 최종 세포 유형으로 칠한다
types = sorted(r_type.unique())
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
titles = {"R": "R (Seurat, uwot)", "default": "Python 기본값 (이웃 15, min_dist 0.5)",
          "seurat": "Python Seurat 맞춤 (이웃 30, cosine, min_dist 0.3)"}
for ax, (s, xy) in zip(axes, umaps.items()):
    for j, t in enumerate(types):
        m = (r_type == t).to_numpy()
        ax.scatter(xy[m, 0], xy[m, 1], s=2, color=plt.get_cmap("tab10")(j), label=t)
    ax.set_title(titles[s], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
handles, lab = axes[0].get_legend_handles_labels()
fig.legend(handles, lab, loc="lower center", ncol=5, frameon=False, markerscale=5)
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(FIG / "fig6_umap_compare.png", dpi=150)
plt.close(fig)

print("그림 저장: fig5a_clusters_on_r_umap, fig5e_cluster_match, fig6_umap_compare")

# ═══ ▶ 직접 정할 값 ═════════════════════════════════════════
# SETTING    : 다음 단계(마커, 세포 유형)에 넘길 군집 설정
#              Python 군집: "default", "snn_leiden", "seurat", "r_graph"
#              R 군집 그대로: "R" (다음 단계를 R과 같은 군집에서 시작하고 싶을 때)
# RESOLUTION : 0.3, 0.5, 0.8, 1.2 중 하나 (R에서는 0.5)
# UMAP       : 그림에 쓸 UMAP — "default" 또는 "seurat"
#   그림 5a, 5e, 6과 표 4–4c를 보고 정한다.
SETTING = "R"
RESOLUTION = 0.5
UMAP = "seurat"

if None in (SETTING, RESOLUTION, UMAP):
    print("\n▶ 여기서 멈춤: 그림과 표를 본 뒤 스크립트 맨 아래 SETTING, RESOLUTION, UMAP을 정하고 다시 실행하세요.")
    sys.exit()
if SETTING not in (*PY, "R") or RESOLUTION not in RES_TRY or UMAP not in ("default", "seurat"):
    sys.exit('SETTING은 "default"/"snn_leiden"/"seurat"/"r_graph"/"R", RESOLUTION은 0.3/0.5/0.8/1.2, UMAP은 "default"/"seurat" 중 하나여야 합니다.')

if SETTING == "R":
    adata.obs["R_res_" + str(RESOLUTION)] = pd.Categorical(clusters["R"][RESOLUTION])
adata.obs["cluster"] = adata.obs[f"{SETTING}_res_{RESOLUTION}"]
adata.obs["r_cluster"] = pd.Categorical(clusters["R"][RESOLUTION])  # 같은 resolution의 R 군집 (비교용)
adata.obsm["X_umap"] = adata.obsm[f"X_umap_{UMAP}"]
adata.obsm["X_umap_r"] = r_umap
adata.uns.update({"cluster_setting": SETTING, "resolution": RESOLUTION, "umap_setting": UMAP})
del adata.obsm["X_umap_seurat_seed1"]
for key in ("neighbors", "umap30", "leiden", "umap"):  # 저장할 때 문제가 되는 설정 기록은 뺀다
    adata.uns.pop(key, None)
adata.write_h5ad(WORK / "pbmc_cluster.h5ad")

clu_tab = adata.obs.groupby("cluster", observed=True).agg(
    세포수=("cluster", "size"), UMI_중앙값=("total_counts", "median"),
    유전자_중앙값=("n_genes_by_counts", "median"), MT_중앙값=("pct_counts_mt", "median"))
clu_tab["비율"] = (100 * clu_tab["세포수"] / adata.n_obs).round(1)
clu_tab = clu_tab.loc[sorted(clu_tab.index, key=int)]
print("\n군집별 요약 (R 표 5와 같은 표)")
print(clu_tab.round(2).to_string())
clu_tab.round(2).to_csv(TAB / "table5_clusters.csv", encoding="utf-8-sig")
print(f"\n저장: work/pbmc_cluster.h5ad (군집 {SETTING}, resolution {RESOLUTION} → {clu_tab.shape[0]}개, UMAP {UMAP})"
      "   (R: resolution 0.5 → 9개)")
