# Locked cross-source sequence validation protocol

Protocol locked on 2026-08-15 before generating model predictions for the
held-out source.

## Objective

Evaluate whether the fixed sequence-feature GBM trained only on the
sequence-covered `NCBI_HR` source transfers to the separately generated
`StaphStudy` assay matrix. This is a source-held-out external validation of the
sequence model. The StaphStudy labels were previously present in VHRnet and
used in the manuscript's four-feature source audit, so this is not described as
a prospectively blinded or newly collected cohort.

## Frozen data

- Interaction file: `data/raw/VirusHostInter.csv`
- SHA-256: `9B7D09FB01C5AB336E817BC2D7A34935E7E31342950DA7FC2DC82499D0A538DF`
- Training source: all sequence-covered `NCBI_HR` rows (expected 1,947 pairs;
  1,488 positive and 459 negative).
- Test source: all `StaphStudy` rows whose exact phage and host genomes resolve
  in the upstream VHIP example bundle (expected 1,053 pairs; 333 positive and
  720 negative; 39 phages by 27 hosts).
- No external labels may be used for fitting, feature selection, calibration,
  threshold selection, hyperparameter selection, or exclusions.

## Frozen inputs and features

- Training FASTAs: `external/phist_run/phages` and
  `external/phist_run/hosts`.
- Test FASTAs: upstream VHIP release at
  `external/upstream_vhip_tool/example/virus_genomes` and
  `external/upstream_vhip_tool/example/host_genomes`.
- The test VHIP feature table is byte-identical to the frozen interaction file;
  its four supplied columns are also required to match exactly after joining on
  phage and host identifiers.
- Use the unchanged feature definitions in `configs/default.yaml`: intrinsic
  node vectors; four locally recomputed pair-composition features; four VHIP
  pair features (`k3dist`, `k6dist`, `GCdiff`, `Homology`); and sixteen
  exact-word, CRISPR-like, and translated-protein proxy features.
- Report a sequence-only sensitivity model that excludes the four VHIP-supplied
  columns. This is secondary and does not replace the locked full model.

## Frozen model

- Primary model: the fixed headline XGBoost configuration in
  `precisionphage.models.fit_predict_gbm` (`n_estimators=400`,
  `max_depth=5`, `learning_rate=0.05`, `subsample=0.8`,
  `colsample_bytree=0.8`, seed 42, histogram tree method).
- Fit `StandardScaler` on NCBI_HR only, then fit one model on all training rows
  and apply it once to StaphStudy.
- The fixed 0.5 threshold is used only for threshold-dependent diagnostics.
  Ranking metrics are primary.

## Prespecified analyses

Primary:

1. Pooled AUROC and AUPRC on all eligible StaphStudy pairs.
2. Brier score and 10-bin expected calibration error.
3. Two-way phage-and-host cluster bootstrap 95% confidence intervals (2,000
   valid replicates) for AUROC and AUPRC. Row-wise intervals are prohibited.

Secondary:

1. Mean, median, range, and count of evaluable per-phage and per-host AUROCs.
2. Results after excluding test hosts whose species identifier occurs in the
   NCBI_HR training set; all StaphStudy phages must be unseen by identifier.
3. Exact-sequence and in-house MinHash/Mash-form distance overlap audits on
   both entity axes.
4. Sequence-only sensitivity model excluding the four VHIP-supplied columns.
5. Four-feature VHIP-only baseline for context.

## Interpretation rules

- Results apply to transfer from NCBI_HR to the StaphStudy source and may not be
  generalized to clinical efficacy.
- A poor result is retained and reported; the model and protocol are not tuned
  after viewing test performance.
- The StaphStudy matrix is at species/reference-genome resolution and is not a
  strain-resolved therapeutic panel.
- Shared species names and near-homologous sequences are reported explicitly;
  no claim of dual cold-start transfer is made unless the overlap audit supports
  it.
