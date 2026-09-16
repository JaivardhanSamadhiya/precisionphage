# PrecisionPhage generalization audit

## Bottom line

The original frozen NCBI_HR-to-StaphStudy result does not establish portable
cross-source discrimination: AUROC was 0.637 with a two-way sequence-cluster
bootstrap 95% CI of 0.498-0.739, and calibration was poor (ECE 0.499).  The
pipeline nevertheless has a separate, positive locked result in a clinically
relevant fixed-phage-bank setting.  A model developed only on the 402-host by
96-phage Picard matrix ranked productive, experimentally tested cocktails for
100 unseen ColoColi hosts with AUROC 0.807 (host-cluster bootstrap 95% CI
0.693-0.906) and average precision 0.951 (0.904-0.983).

This validates **cocktail scoring/prioritization for unseen Escherichia hosts
with a fixed known phage bank**.  It does not show that PrecisionPhage selected
the particular cocktails tested by Gaborieau et al., does not validate unseen
phages, and does not rescue the inconclusive Staphylococcus cross-source test.

## Why the original model did not transport

The dominant problem is dataset design, not GPU capacity or a simple coding
bug.

| Source | Pairs | Positive | Phages | Hosts | Observed matrix density | Median observations per phage | Median observations per host |
|---|---:|---:|---:|---:|---:|---:|---:|
| NCBI_HR (full) | 2,588 | 81.9% | 2,044 | 354 | 0.36% | 1 | 2 |
| NCBI_HR (sequence-covered model set) | 1,947 | 76.4% | 1,418 | 323 | 0.43% | 1 | 2 |
| NahantCollection | 5,208 | 6.1% | 248 | 21 | 100% | 21 | 248 |
| StaphStudy | 1,053 | 31.6% | 39 | 27 | 100% | 27 | 39 |
| Picard development matrix | 38,688 possible; 38,531 observed | 20.8% of observed | 96 | 402 | 99.6% | approximately 401 | approximately 96 |

NCBI_HR is a sparse, literature/record-curated collection dominated by
positives.  A phage usually appears once and a host only twice.  The model can
therefore learn which pairs were selected for reporting and source-specific
composition without learning a stable within-bank susceptibility function.
StaphStudy and Picard instead assay essentially complete matrices with many
true negatives per entity.  The 76.4% to 31.6% prevalence shift explains much
of the severe probability overprediction, while the very different entity
degree and sampling processes explain why simple recalibration cannot repair
ranking by itself.

Additional findings support this diagnosis:

- Exact sequence overlap between NCBI_HR and StaphStudy is zero, and excluding
  pairs near a training genome did not materially improve AUROC.
- Removing the four source-supplied VHIP features did not materially change
  StaphStudy ranking (AUROC 0.639 versus 0.637), whereas those four alone were
  below chance (0.444).  The problem is not confined to one suspect feature
  block.
- The StaphStudy full-model Brier score (0.466) and ECE (0.499) show that a
  decision threshold transported from the positive-heavy NCBI_HR source would
  be unsafe.
- A dense training matrix changes the learnable question: the Picard model can
  compare many phages on each host and many hosts for each phage, which is what
  fixed-bank cocktail prioritization requires.

## Locked external cocktail evaluation

`EXTERNAL_COCKTAIL_VALIDATION_PROTOCOL.md` was written before the ColoColi
outcome and cocktail-composition files were opened.  Experiment 29 used only
Picard interaction labels, five shared host covariates, core-genome distances,
and phage metadata; it selected global hyperparameters by host-grouped Picard
cross-validation and froze 9,600 pair scores.  Experiment 30 verified the
protocol and prediction checksums before joining the 100 held-out cocktail
outcomes.

| Evaluation | N | Positive | AUROC (95% CI) | Average precision (95% CI) |
|---|---:|---:|---:|---:|
| Primary cocktail score | 100 hosts | 82 | 0.807 (0.693-0.906) | 0.951 (0.904-0.983) |
| Selected constituent pairs (secondary) | 300 pairs | 170 | 0.668 (0.559-0.775) | 0.737 (0.599-0.870) |

The primary score used `1 - product(1 - p_i)` as a ranking score, not as a
biological independence model.  The cocktail-level Brier score was 0.127 and
ECE was 0.079, but calibration remains secondary because the component scores
were rank-normalized.

## Cocktail-selection audit

The first attempt to use the rank-normalized cocktail-scoring components for
de novo selection failed: nested grouped validation covered 60.8% of Picard
hosts versus 83.8% for a training-only generalist triplet.  This negative result
is retained in experiment 31.  The reason is specific and reproducible:
per-phage ranks are comparable across hosts for a fixed phage, but they discard
the between-phage breadth information needed to select among 96 phages.

A development-only correction used cross-phage-comparable raw estimates.
Grouped out-of-fold Picard coverage was 91.5% for distance-weighted nearest
neighbours and 90.5% for unbalanced logistic selection, versus 83.8% for the
generalist triplet.  These estimates are useful for selector development but
are not a locked external test, because selector-family choice occurred after
examining Picard cross-validation results.  No exact three-phage cocktail
selected by this corrected rule matched a tested ColoColi cocktail, so the
repository does not claim external wet-lab validation of its de novo choices.

## Scientifically supported submission claim

The defensible claim is that PrecisionPhage identifies leakage-driven optimism,
shows inconclusive transfer from a sparse curated Staphylococcus source, and
independently validates a locked cocktail-prioritization score on unseen hosts
when trained on a dense fixed-bank interaction matrix.  The work supports
computational triage of candidate cocktails for experimental testing.  It does
not establish clinical utility, therapeutic efficacy, resistance suppression,
or general de novo cocktail design across taxa.

## Reproduction

1. Run `python experiments/fetch_staph_validation_sources.py` to stage the
   pinned VHIP and Gaborieau repositories.
2. Run `python experiments/29_freeze_external_cocktail_predictions.py` in an
   environment where held-out outcome files have not been inspected.
3. Run `python experiments/30_evaluate_external_cocktails.py` once after the
   frozen manifest and prediction checksum have been written.
4. Run experiments 31 and 32 to reproduce the negative selection audit and the
   development-only corrected-selector audit.

No GPU is required for these analyses.
