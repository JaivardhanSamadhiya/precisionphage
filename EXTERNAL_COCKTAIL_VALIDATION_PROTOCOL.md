# Locked external cocktail validation protocol

Protocol locked on 2026-08-15 before reading the held-out cocktail-outcome
tables in `external/gaborieau_coli_phage_interactions_2023/dev/cocktails/data`.

## Objective

Test whether a model developed only from the Picard collection's complete
phage-by-bacterium interaction matrix can rank and score phage cocktails for the
separate 100-isolate ColoColi collection.  This is the clinically relevant
fixed-bank setting: the candidate phages are known, but the pathogenic host
isolates are new.

## Development data

- Interaction labels: `data/interactions/interaction_matrix.csv` in the
  Gaborieau et al. repository.
- Host covariates: the shared, outcome-independent columns available for both
  collections (`Clermont_Phylo`, `ST_Warwick`, `LPS_type`, `O-type`, and
  `H-type`).
- Core-genome distances:
  `picard+test_collection_phylogenetic_distances.tsv`.  Test labels are not
  used to derive these distances.
- Phage metadata: `data/genomics/phages/guelin_collection.csv`, including genus
  and isolation host.
- Model and hyperparameter selection use Picard labels only.  Closely related
  Picard hosts are kept in the same validation group using connected components
  at the published 1e-4 core-genome-distance threshold.

## Frozen model family

For every phage, fit two complementary predictors and average their
cross-validated rank-normalized probabilities:

1. a regularized one-hot logistic model using the five shared host covariates;
2. a distance-weighted nearest-neighbour model using core-genome distance to
   Picard strains.

The neighbour count is chosen globally from {5, 10, 20, 40} by mean
phage-level average precision in grouped Picard cross-validation.  Logistic
regularization is chosen globally from C in {0.1, 1, 10} by the same criterion.
Phages with no class variation in a development fold use the fold's empirical
prevalence.  After selection, both components are refit on all Picard hosts.

For a cocktail, the prespecified score is `1 - product(1 - p_i)` across its
constituent phages.  This is a ranking score, not a biological independence
claim.  A secondary score is the maximum constituent probability.

## Sealed external outcomes

Until predictions and checksums are written, the analysis must not read:

- `cocktails.csv`
- `cml_cocktails.csv`
- `cocktails_composition.csv`
- `cocktails_step.csv`

or notebook outputs containing ColoColi efficacy results.

## Primary evaluation

1. AUROC and average precision for productive cocktail activity on the 100
   held-out ColoColi strains, using the published empirical outcome definition.
2. Host-cluster bootstrap 95% confidence intervals (2,000 valid replicates).
3. Calibration diagnostics (Brier score and expected calibration error), with
   the independence-combination score treated as an uncalibrated probability.

Secondary analyses compare the maximum-constituent score, a training-prevalence
generalist score, and performance across generic versus tailored cocktails if
those labels are available without redefining the primary endpoint.

## Interpretation rules

- The ColoColi outcomes are used once after model and predictions are frozen.
- A negative result is retained.  No post-outcome tuning, exclusion, endpoint
  switching, or relabelling is permitted.
- Success supports transfer to new *Escherichia* isolates from a fixed phage
  bank.  It does not establish transfer to unseen phages, clinical efficacy, or
  generalization to *Staphylococcus*.
- This public-data validation is distinct from and does not overwrite the
  previously reported NCBI_HR-to-StaphStudy analysis.
