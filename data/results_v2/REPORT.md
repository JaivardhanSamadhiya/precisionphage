# PrecisionPhage: results summary

Leakage-controlled phage–host interaction prediction, cocktail design, and eco-evolutionary therapy simulation. All numbers below are read directly from the pipeline's result artifacts.

## 1. Headline performance

- Easiest realistic regime (unseen species, LOSO): **AUROC 0.959** (0.944–0.973), ECE 0.021.
- Hardest regime (both phage and host clusters unseen — true cold start): **AUROC 0.785** (0.736–0.835).
- The feature-based GBM has higher row-pooled AUROC than the GNN in every saved regime. Independent-row tests are exploratory because pairs share phage and host entities.

### Table 1. Main results (GBM vs GNN per leakage regime)

| Regime | GBM AUC | GBM ECE | GBM > chance (q) | GNN AUC | GBM−GNN ΔAUC | DeLong q |
| --- | --- | --- | --- | --- | --- | --- |
| Unseen species (LOSO) | 0.959 (0.944–0.973) | 0.021 | 1.0e-03 | 0.869 (0.839–0.899) | +0.090 (0.064–0.116) | 1.9e-11 |
| Unseen host cluster | 0.950 (0.934–0.966) | 0.055 | 1.0e-03 | 0.853 (0.822–0.885) | +0.097 (0.071–0.124) | 4.6e-12 |
| Unseen phage cluster | 0.847 (0.808–0.886) | 0.427 | 1.0e-03 | 0.752 (0.702–0.801) | +0.095 (0.047–0.144) | 1.8e-04 |
| Both unseen (cold start) | 0.785 (0.736–0.835) | 0.163 | 1.0e-03 | 0.672 (0.613–0.732) | +0.113 (0.048–0.176) | 5.4e-04 |

## 2. Leakage hierarchy

AUC generally declines under stricter sequence-cluster holdouts. Taxonomic and sequence-cluster regimes are different axes and should not be treated as a strictly ordered scale.

| regime | mean_auc | ci_lo | ci_hi | pooled_auc | ece | folds_used |
| --- | --- | --- | --- | --- | --- | --- |
| loso_species | 0.903 | 0.751 | 0.982 | 0.959 | 0.021 | 28 |
| logo_genus | 0.884 | 0.833 | 0.929 | 0.92 | 0.057 | 28 |
| host_cluster | 0.899 | 0.795 | 0.976 | 0.95 | 0.055 | 26 |
| phage_cluster | 0.858 | 0.793 | 0.918 | 0.847 | 0.427 | 22 |
| combined_unseen | 0.799 | 0.752 | 0.854 | 0.785 | 0.163 | 5 |

## 3. Graph message-passing ablation

Message passing produced positive row-pooled AUROC differences in all four regimes; only the dual cold-start gain remained below the exploratory BH threshold (Δ=+0.097, q=0.0116). The GBM nevertheless remained the strongest headline model. These row-wise tests are not entity-independent. See `figure_main.png` panel c.

## 4. Cocktail optimisation

- Minimum cocktail (exact ILP) over targets with predicted coverage: **176 phages** (greedy 176, oracle minimum 188).
- Model-driven greedy is compared with a truth-informed greedy reference and random selection (see `cocktail_coverage.png`).

### Table 3. k-robust cocktails

| k | cocktail_size | true_cover_>=1 | true_cover_>=k |
| --- | --- | --- | --- |
| 1 | 176 | 0.896 | 0.896 |
| 2 | 282 | 0.93 | 0.452 |
| 3 | 358 | 0.97 | 0.313 |

## 5. Eco-evolutionary therapy simulation

In the assumption-driven sensitivity model, the redundant strategy also rebounds and does not prevent resistant takeover; its end resistant fraction is 1.000. This is not independent biological validation (see `temporal_dynamics.png`).

### Table 4. Therapy outcomes

| strategy | n_phages | end_load | nadir | log10_drop | resistant_frac_end | rebound |
| --- | --- | --- | --- | --- | --- | --- |
| control | 0 | 1.00e+09 | 5.00e+06 | 0.0 | 0.0 | False |
| monophage | 1 | 1.00e+09 | 5.00e+06 | 0.0 | 0.0 | False |
| cocktail_k1 | 3 | 1.00e+09 | 1.69e+04 | 2.47 | 1.0 | True |
| robust_k2 | 6 | 1.00e+09 | 3.98e+04 | 2.1 | 1.0 | True |

## Figures

- `figure_main.png` — generalisation, leakage hierarchy, GNN ablation
- `cocktail_coverage.png` — cocktail coverage vs size
- `temporal_dynamics.png` — therapy dynamics with resistance
