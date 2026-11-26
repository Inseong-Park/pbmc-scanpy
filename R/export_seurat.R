# export_seurat.R — R(Seurat) 결과를 Python에서 읽을 수 있게 CSV로 내보낸다
# R 프로젝트(../pbmc-scrna)의 결과를 그대로 꺼내기만 하고, 새로 분석하지 않는다.
#
# 실행 (pbmc-scanpy 폴더에서):
#   & "C:\Program Files\R\R-4.4.3\bin\Rscript.exe" R\export_seurat.R
#
# 출력: work/r_export/
#   hvg.csv          변이가 큰 유전자 2,000개 (Seurat 순위대로) + 표준화 분산
#   pca_embeddings.csv  세포 2,511개 x PC 50개 점수
#   pca_stdev.csv    PC별 표준편차
#   pca_loadings.csv 유전자 2,000개 x PC 50개 로딩
#   cell_types.csv   세포별 최종 세포 유형과 군집 번호 (resolution 0.5)

suppressPackageStartupMessages(library(Seurat))

r_dir <- "../pbmc-scrna/work"
out   <- "work/r_export"
dir.create(out, recursive = TRUE, showWarnings = FALSE)

pca   <- readRDS(file.path(r_dir, "pbmc_pca.rds"))
annot <- readRDS(file.path(r_dir, "pbmc_annot.rds"))

hvg  <- VariableFeatures(pca)
hvf  <- HVFInfo(pca)
write.csv(data.frame(rank = seq_along(hvg), gene = hvg,
                     variance_standardized = hvf[hvg, "variance.standardized"]),
          file.path(out, "hvg.csv"), row.names = FALSE)

write.csv(Embeddings(pca, "pca"), file.path(out, "pca_embeddings.csv"))
write.csv(data.frame(PC = seq_along(pca[["pca"]]@stdev), stdev = pca[["pca"]]@stdev),
          file.path(out, "pca_stdev.csv"), row.names = FALSE)
write.csv(Loadings(pca, "pca"), file.path(out, "pca_loadings.csv"))

write.csv(data.frame(barcode = colnames(annot),
                     cell_type = annot$cell_type,
                     cluster = annot$RNA_snn_res.0.5),
          file.path(out, "cell_types.csv"), row.names = FALSE)

cat("내보냄:", out, "\n")
cat("세포", ncol(pca), "/ 변이 큰 유전자", length(hvg), "/ PC", ncol(Embeddings(pca, "pca")), "\n")
