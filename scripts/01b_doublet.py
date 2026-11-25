# 01b_doublet.py — doublet 찾기 (Scrublet), R의 01b_qc_extra.R Part B·C를 scanpy로 다시 한 것
#
# 하는 일:
#   Part A. Scrublet으로 세포마다 doublet 점수를 매기고 판정한다 (R처럼 2,700개 세포 전체에서)
#   Part B. R의 scDblFinder 판정(../pbmc-scrna/work/doublet_scores.csv)과 세포 하나하나 비교한다
#   Part C. ▶ 어느 판정으로 doublet을 뺄지 정하고, work/pbmc_qc_dbl.h5ad로 저장한다
#
# 실행: 프로젝트 폴더(pbmc-scanpy)에서 가상환경을 켠 뒤
#   python scripts\01b_doublet.py
#   → 처음에는 Part B까지만 돌고 멈춘다. 그림과 표를 본 뒤 맨 아래 DBL_SOURCE를 정하고 다시 실행한다.
#
# Scrublet vs scDblFinder — 둘 다 "가짜 doublet을 만들어 닮은 세포를 찾는다"
#   공통: 실제 세포 두 개의 UMI를 더해 가짜 doublet을 만들고, 실제 세포와 함께 PCA 공간에 놓는다.
#         실제 세포의 이웃 중 가짜 doublet이 많으면 doublet 점수가 높다.
#   차이: Scrublet  — 이웃 중 가짜 doublet 비율을 그대로 점수로 쓴다. 기준선은 점수 분포의 골짜기로 정한다.
#         scDblFinder — 이웃 정보에 UMI 수, 유전자 수 등을 더해 분류 모델(gradient boosting)을 학습시키고,
#                       예상 doublet 비율에 맞춰 기준선을 정한다.

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr

from common import FIG, MAX_FEAT, MAX_MT, R_PROJECT, TAB, WORK, load_pbmc

# ═══ Part A. Scrublet ═══════════════════════════════════════
# R과 똑같이 QC로 거르기 전 2,700개 세포에서 시작한다.
# (죽은 세포가 섞여 있어도 doublet 판정에 영향이 작고, R과 같은 세포 집합이어야 비교할 수 있다)
adata = load_pbmc()
print(f"시작: {adata.n_vars:,} 유전자 x {adata.n_obs:,} 세포   (R: 13,714 x 2,700)\n")

# ▶ 직접 정할 값 (1): 판정 기준선
#   None : Scrublet이 점수 분포의 골짜기에서 자동으로 정한다 (처음에는 이렇게 실행)
#   숫자 : 그림 2c 오른쪽(점수 분포)을 보고 자동 기준선이 이상하면 직접 정한다 (예: 0.25)
THRESHOLD = None

# random_state: 가짜 doublet을 무작위로 만들기 때문에 결과를 고정한다 (R의 set.seed(42)와 같은 역할)
# expected_doublet_rate: 기본값 0.05 그대로 (R에서도 scDblFinder 기본값을 썼다)
sc.pp.scrublet(adata, threshold=THRESHOLD, random_state=0, verbose=False)

obs = adata.obs
thr = adata.uns["scrublet"]["threshold"]
obs["dbl_py"] = np.where(obs["predicted_doublet"], "doublet", "singlet")
n_dbl = int((obs["dbl_py"] == "doublet").sum())
print(f"Scrublet 기준선: {thr:.3f}  ({'자동' if THRESHOLD is None else '직접 정함'})")
print(f"Scrublet doublet: {n_dbl} / {adata.n_obs:,} ({100 * n_dbl / adata.n_obs:.1f}%)"
      "   (R scDblFinder: 133 / 2,700, 4.9%)\n")

# 유전자 수 기준(2,500 이상)으로 잡은 큰 doublet 5개를 Scrublet도 잡았나?
over_feat = (obs["n_genes_by_counts"] >= MAX_FEAT).rename("유전자 2,500 이상")
print(pd.crosstab(obs["dbl_py"].rename("Scrublet"), over_feat).to_string())
print("R: 유전자 2,500 이상 5개 모두 scDblFinder doublet\n")

# doublet 판정 세포는 보통 세포와 무엇이 다른가? (R의 table2b와 같은 표)
dbl_summary = obs.groupby("dbl_py").agg(
    세포수=("dbl_py", "size"),
    UMI_중앙값=("total_counts", "median"),
    유전자_중앙값=("n_genes_by_counts", "median"),
    MT_중앙값=("pct_counts_mt", "median"),
    MT_5이상=("pct_counts_mt", lambda x: int((x >= MAX_MT).sum())),
).round(2)
print("판정별 요약 (Python Scrublet)")
print(dbl_summary.to_string())
print("R scDblFinder: singlet UMI 2,166 / 유전자 806 / MT 2.03,  doublet UMI 3,446 / 유전자 1,186 / MT 1.95\n")
dbl_summary.to_csv(TAB / "table2b_doublet_summary.csv", encoding="utf-8-sig")

# 서로 다른 세포 종류의 마커가 한 세포에서 같이 잡히는 비율 (R의 그림 2d와 같은 확인)
markers = {"T세포": "CD3E", "B세포": "MS4A1", "단핵구": "LYZ", "NK세포": "NKG7", "혈소판": "PPBP"}
mk = pd.DataFrame(
    (adata[:, list(markers.values())].X > 0).toarray(),
    index=obs.index, columns=list(markers.keys()),
)
n_types = mk[["T세포", "B세포", "단핵구"]].sum(axis=1)
co = (n_types >= 2).groupby(obs["dbl_py"]).mean()
print("T·B·단핵구 마커가 2개 이상 동시에 검출된 세포 비율")
print((100 * co).round(1).astype(str).add("%").to_string())
print("R scDblFinder: singlet 24.0%, doublet 51.9%\n")

# ═══ Part B. R의 scDblFinder와 비교 ═════════════════════════
r = pd.read_csv(R_PROJECT / "work" / "doublet_scores.csv", index_col="barcode")
cmp = obs[["doublet_score", "dbl_py", "total_counts", "n_genes_by_counts"]].join(
    r[["dbl_score", "dbl_class"]], how="inner"
)
print(f"두 도구에 공통으로 있는 세포: {len(cmp):,}개")

tab = pd.crosstab(
    cmp["dbl_class"].rename("R scDblFinder"), cmp["dbl_py"].rename("Python Scrublet"), margins=True, margins_name="합계"
)
print(tab.to_string())
tab.to_csv(TAB / "table2c_doublet_compare.csv", encoding="utf-8-sig")

is_r = cmp["dbl_class"] == "doublet"
is_py = cmp["dbl_py"] == "doublet"
both = int((is_r & is_py).sum())
union = int((is_r | is_py).sum())
rho = spearmanr(cmp["doublet_score"], cmp["dbl_score"]).statistic
print(f"\n둘 다 doublet: {both}개 / 둘 중 하나라도 doublet: {union}개  → 겹침 비율 {100 * both / union:.1f}%")
print(f"R doublet 중 Scrublet도 잡은 비율: {100 * both / is_r.sum():.1f}%")
print(f"Scrublet doublet 중 R도 잡은 비율: {100 * both / max(is_py.sum(), 1):.1f}%")
print(f"두 점수의 순위 상관 (Spearman): {rho:.2f}   (1이면 순서가 완전히 같음, 0이면 무관)\n")

# 기준선 민감도: 기준선을 옮기면 무엇이 바뀌나 (R에서 MT 기준을 5/10/20%로 바꿔 본 것과 같은 확인)
#   Scrublet 점수는 "이웃 중 가짜 doublet 비율"이라 띄엄띄엄한 값만 나온다.
#   그래서 실제로 나온 점수 값 사이에 기준선을 하나씩 놓아 본다.
sim = adata.uns["scrublet"]["doublet_scores_sim"]
score_py = cmp["doublet_score"]
levels = np.sort(score_py[(score_py >= 0.1) & (score_py <= 0.4)].unique())
cuts = (levels[:-1] + levels[1:]) / 2  # 이웃한 점수 값의 가운데
sens = pd.DataFrame({"기준선": cuts})
sens["실제 doublet 수"] = [int((score_py > c).sum()) for c in cuts]
sens["가짜 doublet 중 넘은 비율(%)"] = [round(100 * (sim > c).mean(), 1) for c in cuts]
sens["R과 겹침"] = [int(((score_py > c) & is_r).sum()) for c in cuts]
sens["R doublet 중 잡은 비율(%)"] = (100 * sens["R과 겹침"] / is_r.sum()).round(1)
sens["기준선"] = sens["기준선"].round(3)
print("기준선별 결과 (실제로 나온 점수 값 사이마다)")
print(sens.to_string(index=False))
sens.to_csv(TAB / "table2d_scrublet_threshold.csv", index=False, encoding="utf-8-sig")

# 원래 Scrublet 패키지가 보여 주는 추정: 잡을 수 있는 doublet의 비율로 전체 doublet 비율을 거꾸로 추정한다
#   가짜 doublet 중 기준선을 넘는 비율 = 이 기준선으로 "잡을 수 있는" doublet의 비율
#   실제 세포 중 기준선을 넘는 비율   = 실제로 잡은 비율
#   잡은 비율 ÷ 잡을 수 있는 비율      = 기준선 때문에 놓친 것까지 합친 전체 doublet 비율 추정
detectable = (sim > thr).mean()
detected = (obs["doublet_score"] > thr).mean()
print(f"\nScrublet 추정 (현재 기준선 {thr:.3f})")
print(f"  잡을 수 있는 doublet 비율 (가짜 doublet 중 기준선을 넘은 비율): {100 * detectable:.1f}%")
print(f"  실제로 잡은 비율: {100 * detected:.1f}%")
print(f"  → 추정 전체 doublet 비율: {100 * detected / detectable:.1f}%   (= 잡은 비율 ÷ 잡을 수 있는 비율)\n")

# 일치 여부로 세포를 네 무리로 나눈다
agree = np.select(
    [is_r & is_py, is_r & ~is_py, ~is_r & is_py],
    ["둘 다 doublet", "R만 doublet", "Python만 doublet"],
    default="둘 다 singlet",
)
cmp["agree"] = agree
agree_colors = {
    "둘 다 singlet": "lightgrey", "R만 doublet": "#E69F00",
    "Python만 doublet": "#56B4E9", "둘 다 doublet": "#D55E00",
}
# 무리별 크기 비교: 한쪽만 잡은 doublet은 어떤 세포인가?
print("일치 여부별 요약")
print(cmp.groupby("agree").agg(
    세포수=("agree", "size"),
    UMI_중앙값=("total_counts", "median"),
    유전자_중앙값=("n_genes_by_counts", "median"),
).reindex(agree_colors.keys()).to_string())
print()

cmp.to_csv(WORK / "doublet_scores_py.csv", encoding="utf-8-sig")

# ── 그림 ────────────────────────────────────────────────────
# 그림 2c: Scrublet 판정 (R의 fig2c_doublets.png와 같은 구성) + 점수 분포
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [3, 2]})
order = obs.sort_values("dbl_py", ascending=False)  # doublet을 위에 그리기
ax = axes[0]
for name, color in [("singlet", "lightgrey"), ("doublet", "#D55E00")]:
    m = order["dbl_py"] == name
    ax.scatter(order.loc[m, "total_counts"], order.loc[m, "n_genes_by_counts"], s=3, c=color, label=name)
ax.axhline(MAX_FEAT, linestyle="--", color="red", linewidth=1)
ax.set_xlabel("UMI 수")
ax.set_ylabel("검출 유전자 수")
ax.legend(title="Scrublet", markerscale=3)

ax = axes[1]
bins = np.linspace(0, 1, 51)
ax.hist(obs["doublet_score"], bins=bins, density=True, color="grey", alpha=0.7, label="실제 세포")
ax.hist(adata.uns["scrublet"]["doublet_scores_sim"], bins=bins, density=True,
        histtype="step", color="#D55E00", linewidth=1.5, label="가짜 doublet")
ax.axvline(thr, linestyle="--", color="red", linewidth=1, label=f"기준선 {thr:.2f}")
ax.set_xlabel("doublet 점수")
ax.set_ylabel("밀도")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "fig2c_doublets.png", dpi=150)
plt.close(fig)

# 그림 2e: R 점수 vs Python 점수, 그리고 일치 여부를 UMI-유전자 그림에 표시
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for name, color in agree_colors.items():
    m = cmp["agree"] == name
    axes[0].scatter(cmp.loc[m, "dbl_score"], cmp.loc[m, "doublet_score"], s=4, c=color, label=f"{name} ({m.sum()})")
    axes[1].scatter(cmp.loc[m, "total_counts"], cmp.loc[m, "n_genes_by_counts"], s=4, c=color)
axes[0].axhline(thr, linestyle="--", color="#56B4E9", linewidth=1)
axes[0].set_xlabel("R scDblFinder 점수")
axes[0].set_ylabel("Python Scrublet 점수")
axes[0].set_title(f"순위 상관 {rho:.2f}")
axes[1].axhline(MAX_FEAT, linestyle="--", color="red", linewidth=1)
axes[1].set_xlabel("UMI 수")
axes[1].set_ylabel("검출 유전자 수")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, markerscale=3)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(FIG / "fig2e_doublet_compare.png", dpi=150)
plt.close(fig)

print("그림 저장: fig2c_doublets.png, fig2e_doublet_compare.png")

# ═══ Part C. ▶ 직접 정할 값 (2): 어느 판정으로 doublet을 뺄지 ═══
#   "python" : Scrublet 판정으로 뺀다 → Python만으로 끝까지 분석한 결과가 된다.
#              다만 R과 세포 집합이 달라져서, 다음 단계에서 R과 차이가 나면
#              doublet 차이 때문인지 분석 방법 차이 때문인지 섞인다.
#   "r"      : R의 scDblFinder 판정으로 뺀다 → R과 똑같은 2,511개 세포에서 출발한다.
#              다음 단계(정규화, PCA, 군집)의 차이를 순수하게 도구 차이로 볼 수 있다.
#   "none"   : 빼지 않는다 (2,638개 그대로)
# 어느 쪽이든 두 판정(dbl_py, dbl_r)은 열로 남겨 두므로 나중에 다시 확인할 수 있다.
DBL_SOURCE = "r"

if DBL_SOURCE is None:
    print("\n▶ 여기서 멈춤: 그림과 표를 본 뒤 스크립트 맨 아래 DBL_SOURCE를 정하고 다시 실행하세요.")
    sys.exit()
if DBL_SOURCE not in ("python", "r", "none"):
    sys.exit('DBL_SOURCE는 "python", "r", "none" 중 하나여야 합니다.')

q = sc.read_h5ad(WORK / "pbmc_qc.h5ad")  # 01_qc.py 결과 (2,638개)
q.obs["doublet_score"] = cmp.loc[q.obs_names, "doublet_score"].values
q.obs["dbl_py"] = cmp.loc[q.obs_names, "dbl_py"].values
q.obs["dbl_r"] = cmp.loc[q.obs_names, "dbl_class"].values
before = q.n_obs
if DBL_SOURCE == "python":
    q = q[q.obs["dbl_py"] == "singlet"].copy()
elif DBL_SOURCE == "r":
    q = q[q.obs["dbl_r"] == "singlet"].copy()
print(f"\ndoublet 제거 ({DBL_SOURCE}): {before:,} → {q.n_obs:,} 세포   (R: 2,638 → 2,511)")
q.write_h5ad(WORK / "pbmc_qc_dbl.h5ad")
print("저장:", WORK / "pbmc_qc_dbl.h5ad")
