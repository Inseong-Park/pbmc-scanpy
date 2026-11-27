# pbmc-scanpy

PBMC3k scRNA-seq 데이터를 Python(scanpy)으로 다시 분석하고, R(Seurat) 분석 결과(`../pbmc-scrna`)와 비교하는 프로젝트.

## 데이터
- 새로 받지 않고 `../pbmc-scrna/data/filtered_gene_bc_matrices/hg19/`를 그대로 읽는다. 두 도구를 같은 입력으로 비교하기 위해서다.

## 폴더
- `scripts/`: 분석 스크립트 (단계별 번호)
- `results/figures`, `results/tables`: 최종 그림과 표
- `work/`: 중간 결과 (`.h5ad`), git 제외

## 스크립트 (순서대로 실행)
| 스크립트 | 하는 일 | R 대응 |
|---|---|---|
| `scripts/common.py` | 공통 경로, 데이터 읽기 (직접 실행하지 않음) | |
| `scripts/01_qc.py` | QC 지표, 세포 거르기 → `work/pbmc_qc.h5ad` | `01_qc.R` |
| `scripts/01b_doublet.py` | Scrublet doublet 판정, scDblFinder와 비교 → `work/pbmc_qc_dbl.h5ad` | `01b_qc_extra.R` |
| `R/export_seurat.R` | R 결과(HVG, PCA, 군집, UMAP, SNN 그래프, 세포 유형)를 CSV로 내보냄 → `work/r_export/` (Rscript로 실행) | |
| `scripts/02_pca.py` | 정규화 → HVG → PCA를 두 설정(scanpy 기본값, Seurat 맞춤)으로 하고 R과 비교 → `work/pbmc_pca.h5ad` | `02_pca.R` |
| `scripts/03_cluster.py` | 이웃 그래프 → 군집 → UMAP을 네 설정으로 하고 R과 비교 → `work/pbmc_cluster.h5ad` | `03_cluster.R` |
| `scripts/04_annotate.py` | 마커 유전자를 두 설정으로 찾고 R과 비교, 세포 유형 이름 붙이기 → `work/pbmc_annot.h5ad` | `04_annotate.R` |

단계별 판단과 R과의 비교는 [notes.md](notes.md)에 있다.

## 환경
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
