# 05_robustness.py — Python 방식으로 바꿔도 최종 세포 유형이 R과 같은가? (민감도 확인)
#
# 입력: 원본 행렬 (common.load_pbmc), R의 doublet 판정 (../pbmc-scrna/work/doublet_scores.csv),
#       Scrublet 판정 (work/doublet_scores_py.csv), R의 최종 세포 유형 (work/r_export/cell_types.csv)
# 출력: 그림 9a–9c, 표 9–9c
#
# 실행: python scripts\05_robustness.py   (2–3분 걸린다. 직접 정할 값은 없다)
#
# 왜 하나:
#   2–6단계에서는 단계마다 입력을 R과 똑같이 맞추고, 그 단계의 방법만 바꿔서 비교했다.
#   그래서 "각 단계에서 무엇이 달랐나"는 알지만, "그 차이들이 최종 결론(세포 유형과 비율)까지 바꾸나"는 아직 모른다.
#   여기서는 원본에서 처음부터 끝까지 다시 돌리면서, Python 방식을 한 단계씩 넣어 본다.
#   R 노트 Day 5에서 기준을 하나씩 바꿔 본 민감도 분석과 같은 방식이다.
#
# 바꿔 볼 설정 (py_match에서 한 가지만 바꾼다 → 차이가 나면 그 단계 때문이라고 말할 수 있다)
#   py_match        : 모든 단계를 Seurat에 맞춘 Python. 군집만 Python의 Louvain (5단계에서 R과 ARI 0.764)
#   dbl_scrublet    : doublet을 Scrublet 판정으로 뺀다                      (3단계의 차이)
#   pca_default     : 정규화·HVG·PCA를 scanpy 기본값으로                     (4단계의 차이)
#   cluster_default : 이웃 그래프와 군집을 scanpy 기본값(이웃 15개 + Leiden)으로 (5단계의 차이)
#   all_default     : 위 세 가지를 모두 → scanpy를 처음부터 끝까지 기본값으로 쓴 경우
#   모든 설정에서 QC 기준(유전자 200–2,500, MT 5%), PC 9개, resolution 0.5는 같다.
#
# 이름 붙이기: 새로 나온 군집에는 이름이 없으므로 R 민감도 분석과 같은 방법을 쓴다.
#   R에도 있는 세포(공통 세포)의 R 이름을 빌려서, 군집마다 가장 많은 이름을 붙인다(다수결).
#   그 이름을 가진 세포의 비율을 "순도"로 본다. R에 없던 세포(Scrublet만으로 뺀 경우 남은 doublet)는
#   자기가 들어간 군집의 이름을 받는다.

import random

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from common import FIG, MAX_FEAT, MAX_MT, MIN_FEAT, R_PROJECT, TAB, WORK, load_pbmc

N_PCS, RES = 9, 0.5

r_type = pd.read_csv(WORK / "r_export" / "cell_types.csv", index_col="barcode")["cell_type"]
r_clu = pd.read_csv(WORK / "r_export" / "cell_types.csv", index_col="barcode")["cluster"].astype(str)
dbl_r = pd.read_csv(R_PROJECT / "work" / "doublet_scores.csv", index_col="barcode")["dbl_class"]
dbl_py = pd.read_csv(WORK / "doublet_scores_py.csv", index_col=0)["dbl_py"]
TYPES = list(r_type.value_counts().index)  # 세포 수가 많은 순서

raw = load_pbmc()
raw = raw[(raw.obs["n_genes_by_counts"] > MIN_FEAT) & (raw.obs["n_genes_by_counts"] < MAX_FEAT)
          & (raw.obs["pct_counts_mt"] < MAX_MT)].copy()
raw.layers["counts"] = raw.X.copy()
print(f"QC 후: {raw.n_obs:,} 세포   (R: 2,638)\n")


def pca_step(a, how):
    """4단계(02_pca.py)와 같은 계산"""
    if how == "default":
        sc.pp.normalize_total(a)
        sc.pp.log1p(a)
        sc.pp.highly_variable_genes(a)
        h = a[:, a.var["highly_variable"]].copy()
        sc.pp.scale(h)
    else:
        sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=2000)
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        h = a[:, a.var["highly_variable"]].copy()
        sc.pp.scale(h, max_value=10)
    sc.tl.pca(h, n_comps=50, random_state=0)
    return h.obsm["X_pca"][:, :N_PCS]


def cluster_step(a, X, how):
    """5단계(03_cluster.py)와 같은 계산"""
    if how == "default":
        a.obsm["X_use"] = X
        sc.pp.neighbors(a, use_rep="X_use", random_state=0)
        sc.tl.leiden(a, resolution=RES, random_state=0, key_added="clu")
        return a.obs["clu"].astype(str).to_numpy()
    k = 20  # Seurat 방식 SNN 그래프 + Louvain
    idx = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X, return_distance=False)
    M = sp.csr_matrix((np.ones(idx.size), (np.repeat(np.arange(len(X)), k), idx.ravel())), shape=(len(X), len(X)))
    s = (M @ M.T).tocoo()
    j = s.data / (2 * k - s.data)
    keep = (j >= 1 / 15) & (s.row != s.col)
    snn = sp.csr_matrix((j[keep], (s.row[keep], s.col[keep])), shape=(len(X), len(X)))
    random.seed(0)
    g = ig.Graph.Weighted_Adjacency(sp.triu(snn).tocoo(), mode="undirected")
    return np.array([str(c) for c in g.community_multilevel(weights="weight", resolution=RES).membership])


SETTINGS = {
    "py_match": ("r", "seurat", "seurat", "Seurat 맞춤 Python"),
    "dbl_scrublet": ("python", "seurat", "seurat", "doublet만 Scrublet"),
    "pca_default": ("r", "default", "seurat", "PCA만 scanpy 기본값"),
    "cluster_default": ("r", "seurat", "default", "군집만 scanpy 기본값"),
    "all_default": ("python", "default", "default", "모두 scanpy 기본값"),
}

results = {}
for name, (dbl, pca, clu, label) in SETTINGS.items():
    print(f"실행 중: {label} ...")
    keep = (dbl_r if dbl == "r" else dbl_py).loc[raw.obs_names] == "singlet"
    a = raw[keep.to_numpy()].copy()
    X = pca_step(a, pca)
    lab = cluster_step(a, X, clu)
    results[name] = pd.Series(lab, index=a.obs_names)


# ── 이름 빌려 오기 (다수결) ────────────────────────────────
def transfer(clusters):
    df = pd.DataFrame({"cluster": clusters, "r": r_type.reindex(clusters.index)})
    common = df.dropna(subset=["r"])
    vote = common.groupby("cluster")["r"].agg(lambda x: x.value_counts().index[0])
    purity = common.groupby("cluster")["r"].agg(lambda x: x.value_counts(normalize=True).iloc[0])
    df["label"] = df["cluster"].map(vote).fillna("R에 없는 군집")
    df["purity"] = df["cluster"].map(purity)
    return df


rows, props, tables = [], {}, {}
for name, clusters in results.items():
    df = transfer(clusters)
    tables[name] = df
    common = df.dropna(subset=["r"])
    rows.append({
        "설정": SETTINGS[name][3], "세포 수": len(df), "공통 세포": len(common), "R에 없던 세포": int(df["r"].isna().sum()),
        "군집 수": df["cluster"].nunique(),
        "R 군집과 ARI": round(adjusted_rand_score(r_clu.reindex(common.index), common["cluster"]), 3),
        "R 세포 유형과 ARI": round(adjusted_rand_score(common["r"], common["cluster"]), 3),
        "이름 일치율(%)": round(100 * (common["label"] == common["r"]).mean(), 1),
        "가장 낮은 순도(%)": round(100 * df.groupby("cluster")["purity"].first().min(), 1),
        "이름이 사라진 세포 유형": ", ".join(t for t in TYPES if t not in set(df["label"])) or "없음",
    })
    props[SETTINGS[name][3]] = (100 * df["label"].value_counts(normalize=True)).reindex(TYPES).fillna(0)

summary = pd.DataFrame(rows)
print("\n설정별 요약 (이름 일치율: 공통 세포 중 빌려 온 이름이 R 이름과 같은 비율)")
print(summary.to_string(index=False))
summary.to_csv(TAB / "table9_robustness_summary.csv", index=False, encoding="utf-8-sig")

prop_tab = pd.DataFrame(props)
prop_tab.insert(0, "R", (100 * r_type.value_counts(normalize=True)).reindex(TYPES))
print("\n세포 유형 비율 (%)")
print(prop_tab.round(1).to_string())
prop_tab.round(2).to_csv(TAB / "table9c_robustness_proportions.csv", encoding="utf-8-sig")

# 표 9b: 설정별 군집 → 빌려 온 이름, 세포 수, 순도
clu_rows = []
for name, df in tables.items():
    g = df.groupby("cluster").agg(세포수=("label", "size"), 이름=("label", "first"), 순도=("purity", "first"))
    g["설정"] = SETTINGS[name][3]
    clu_rows.append(g.reset_index())
clu_tab = pd.concat(clu_rows)[["설정", "cluster", "이름", "세포수", "순도"]]
clu_tab["순도"] = (100 * clu_tab["순도"]).round(1)
clu_tab.to_csv(TAB / "table9b_robustness_clusters.csv", index=False, encoding="utf-8-sig")

# 순도가 낮은 군집: 서로 다른 R 세포 유형이 한 군집에 섞인 곳
low = clu_tab[clu_tab["순도"] < 90].sort_values("순도")
print("\n순도 90% 미만인 군집 (서로 다른 R 세포 유형이 섞인 군집)")
for _, r in low.iterrows():
    df = tables[[k for k, v in SETTINGS.items() if v[3] == r["설정"]][0]]
    mix = df[df["cluster"] == r["cluster"]]["r"].value_counts().head(3)
    print(f"  {r['설정']:<16} 군집 {r['cluster']:>2} ({r['세포수']}개, 순도 {r['순도']}%): "
          + ", ".join(f"{k} {v}" for k, v in mix.items()))

# ── 그림 ────────────────────────────────────────────────────
names = [SETTINGS[n][3] for n in SETTINGS]

# 그림 9a: 설정별 이름 일치율
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(names, summary["이름 일치율(%)"], color=["#999999", "#56B4E9", "#E69F00", "#009E73", "#D55E00"])
for i, v in enumerate(summary["이름 일치율(%)"]):
    ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
ax.set_ylim(min(80, summary["이름 일치율(%)"].min() - 5), 101)
ax.set_ylabel("R과 같은 이름을 받은 세포 (%)")
ax.tick_params(axis="x", labelsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig9a_robustness_agreement.png", dpi=150)
plt.close(fig)

# 그림 9b: 모두 scanpy 기본값일 때 R 세포 유형(행) → 빌려 온 이름(열)
df = tables["all_default"].dropna(subset=["r"])
tab = pd.crosstab(df["r"], df["label"]).reindex(index=TYPES)
tab = tab[[t for t in TYPES if t in tab.columns] + [c for c in tab.columns if c not in TYPES]]
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(np.log1p(tab.to_numpy()), cmap="Blues", aspect="auto")
for (y, x), v in np.ndenumerate(tab.to_numpy()):
    if v > 0:
        ax.text(x, y, v, ha="center", va="center", fontsize=8)
ax.set_xticks(range(tab.shape[1]))
ax.set_xticklabels(tab.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(tab.shape[0]))
ax.set_yticklabels(tab.index, fontsize=8)
ax.set_xlabel("모두 scanpy 기본값: 빌려 온 이름")
ax.set_ylabel("R 세포 유형")
fig.tight_layout()
fig.savefig(FIG / "fig9b_all_default_vs_r.png", dpi=150)
plt.close(fig)

# 그림 9c: 세포 유형 비율
fig, ax = plt.subplots(figsize=(12, 4.5))
w = 0.8 / prop_tab.shape[1]
for i, col in enumerate(prop_tab.columns):
    ax.bar(np.arange(len(TYPES)) + (i - prop_tab.shape[1] / 2 + 0.5) * w, prop_tab[col], width=w, label=col)
ax.set_xticks(range(len(TYPES)))
ax.set_xticklabels(TYPES, fontsize=8)
ax.set_ylabel("비율 (%)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig9c_robustness_proportions.png", dpi=150)
plt.close(fig)

print("\n그림 저장: fig9a_robustness_agreement, fig9b_all_default_vs_r, fig9c_robustness_proportions")
