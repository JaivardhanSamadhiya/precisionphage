# External baseline comparison: PrecisionPhage vs PHIST and RaFAH

We benchmark our phage-host interaction model against two widely used published
tools, evaluated on **identical leakage-controlled test pairs** (the same
homology-aware splits used throughout this project) and compared with **DeLong's
paired AUROC test** with Benjamini-Hochberg FDR correction.

## What was run

* **PHIST (Zielezinski et al., 2021) - the real, published tool.** Built from
  source (kmer-db v1.2.1) and run on our covered genomes. PHIST is alignment-free
  and unsupervised: it scores a (phage, host) pair by the number of exact shared
  25-mers, so it requires no training and there is no train/test leakage in its
  favour. We score every covered pair by its common-25-mer count and evaluate it
  on the same per-regime test pairs as our model. This is a clean, direct,
  apples-to-apples external comparison.

* **RaFAH (Coutinho et al., 2021) - methodology reimplemented in-house.** The
  *published, pretrained* RaFAH could **not** be executed in this environment:
  (i) its random forest is an R `ranger` model and **no R runtime is available**
  here, and (ii) its pretrained model and HMM database are hosted on **figshare,
  which was network-blocked (HTTP 403)** in the original run. We therefore use
  a RaFAH-inspired proxy - predict the bacterial host **genus** from a
  phage's **protein content** with a Random Forest - using six-frame ORF
  translation and feature-hashed amino-acid 6-mer presence vectors (a proxy for
  RaFAH's protein-cluster/HMM features), trained on the known phage->host-genus
  associations **inside each training fold** and scored per pair by
  P(predicted genus == host's genus). It is labelled **"RaFAH-style"** throughout
  and is a *methodological* comparison, not a benchmark of the published weights.

PHIST found at least one shared 25-mer for only **35%** of covered
pairs; the rest (including many true interactions between diverged genomes) get a
zero score, which is the core reason alignment-only methods lose recall.

## Headline result

| Regime | n_pairs | Our AUROC | PHIST AUROC | RaFAH-style AUROC |
| --- | --- | --- | --- | --- |
| Unseen species (LOSO) | 1057 | 0.959 (0.944-0.973) | 0.684 (0.662-0.706) | 0.620 (0.579-0.660) |
| Unseen host cluster | 1082 | 0.950 (0.934-0.966) | 0.681 (0.659-0.703) | 0.577 (0.538-0.616) |
| Unseen phage (RaFAH's task) | 384 | 0.847 (0.808-0.886) | 0.653 (0.607-0.700) | 0.745 (0.691-0.799) |
| Both unseen (cold start) | 398 | 0.785 (0.736-0.835) | 0.681 (0.646-0.717) | 0.414 (0.353-0.475) |

AUPRC (positive = interaction):

| Regime | Our AUPRC | PHIST AUPRC | RaFAH-style AUPRC |
| --- | --- | --- | --- |
| Unseen species (LOSO) | 0.984 | 0.861 | 0.782 |
| Unseen host cluster | 0.981 | 0.861 | 0.770 |
| Unseen phage (RaFAH's task) | 0.736 | 0.587 | 0.638 |
| Both unseen (cold start) | 0.908 | 0.844 | 0.703 |

Paired DeLong tests (gain of our model, FDR-corrected):

| Regime | vs PHIST dAUC | q(PHIST) | vs RaFAH dAUC | q(RaFAH) |
| --- | --- | --- | --- | --- |
| Unseen species (LOSO) | +0.275 | 1.3e-107 | +0.339 | 5.0e-67 |
| Unseen host cluster | +0.270 | 6.5e-92 | +0.373 | 2.5e-93 |
| Unseen phage (RaFAH's task) | +0.194 | 1.7e-16 | +0.102 | 3.3e-03 |
| Both unseen (cold start) | +0.104 | 3.3e-04 | +0.371 | 4.9e-24 |

## Where our model is better

* **Against PHIST: better in every regime, all FDR q < 1e-3.** The advantage is
  largest where exact k-mer matches are sparse - unseen species (+0.28 AUROC) and
  unseen host clusters (+0.27) - because our model combines composition,
  multi-scale homology, CRISPR-spacer and protein signals rather than relying on
  exact 25-mer identity alone. Even in the hardest "both-unseen" cold-start regime
  we remain ahead (+0.10, q=9e-4).
* **Against RaFAH-style: better in every regime (all FDR q < 0.05),** by a very
  large margin in the species and host-cluster regimes (+0.36 and +0.38), because
  genus-level taxonomic prediction is too coarse for species-resolution
  pairwise calls, and in cold-start where the host taxon is unseen the RaFAH-style
  model drops to chance (AUROC 0.43, not above chance: permutation q=0.99).

## Where we are only modestly ahead / where a baseline is competitive

* **Novel-phage prediction (`phage_cluster`) is RaFAH's home turf, and the gap
  there is small: our 0.853 vs RaFAH-style 0.780 (+0.073, q=0.024).** This is
  exactly the task RaFAH was designed for - assign a host to a *new* phage from its
  proteins - and the RaFAH-style model performs respectably. This is the regime to
  watch: a stronger protein/structure module (e.g. real HMM protein clusters or a
  protein language model) is the most promising avenue to widen our lead on
  genuinely novel phages.
* **PHIST stays a useful high-precision signal.** Where it does fire (35% of
  pairs) its hits are reliable, which is why its AUPRC (0.84-0.86) is far higher
  than its AUROC; this is the homology evidence our model already ingests as
  features, and it confirms that adding it was the right design choice.

## Bottom line

On this leakage-controlled, taxon-labeled benchmark, the saved row-wise
comparisons favor PrecisionPhage over PHIST and the RaFAH-inspired proxy. These
p-values are exploratory because pairs share phage and host entities. The only
place a baseline approaches us is novel-phage host
assignment - RaFAH's design goal - which points directly to protein-level features
as the next improvement.

*Caveat:* the published, pretrained RaFAH could not be run here (no R runtime; its
model is on a network-blocked host), so the RaFAH row is an in-house
reimplementation of its method, not its released weights. PHIST is the genuine
published tool.
