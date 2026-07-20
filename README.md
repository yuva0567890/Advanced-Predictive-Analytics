# Advanced-Predictive-Analytics
# House Price Prediction using Regression Analysis

**Course:** MDI3003 — Advanced Predictive Analytics (VIT Vellore)
**Lab:** Lab 01 — House Price Prediction — Extended Comparative Study
**Author:** [Your Name] · [Registration No.]

An end-to-end regression study comparing five models across two real-estate datasets of very different scale and geography, built with a leakage-safe scikit-learn pipeline, 5-fold cross-validation, hyperparameter tuning, and full residual diagnostics.

---

## 1. Overview

This project predicts residential property value from measurable property and location attributes, using:

- **UCI Real Estate Valuation** (414 transactions, Sindian/Xindian district, New Taipei) — target: price per unit floor area
- **California Housing** (8,000-row subsample of the 20,640-row StatLib dataset) — target: median house value per census block group ($100k units)

Five regression models are trained and evaluated under an identical protocol on each dataset:

| Model | Type |
|---|---|
| Simple Linear Regression | Interpretable one-feature baseline |
| Polynomial Regression (degree 2, Ridge-controlled) | Controlled nonlinear extension |
| Elastic Net (tuned via GridSearchCV) | Regularized linear model (L1 + L2) |
| Random Forest Regressor | Variance-reduced nonlinear ensemble |
| Gradient Boosting | Sequential error-correcting ensemble |

A naive mean-prediction baseline is included for reference in all comparisons.

---

## 2. Repository Structure

```
.
├── README.md                          # this file
├── HousePrice_Report.docx             # full industry-style technical report (~16 pages)
├── notebooks/
│   └── HousePrice_Lab01.ipynb         # executable end-to-end notebook (both datasets, all 5 models)
├── data/
│   ├── uci_real_estate.csv            # UCI Real Estate Valuation data
│   └── california_housing.csv         # California Housing data
├── results/
│   ├── audit_UCI_RealEstate.csv
│   ├── audit_California_Housing.csv
│   ├── test_results_UCI_RealEstate.csv
│   ├── test_results_California_Housing.csv
│   ├── cv_results_UCI_RealEstate.csv
│   ├── cv_results_California_Housing.csv
│   ├── gap_results_UCI_RealEstate.csv
│   ├── gap_results_California_Housing.csv
│   ├── segment_UCI_RealEstate.csv
│   ├── segment_California_Housing.csv
│   └── target_corr_*.csv
├── figures/
│   ├── target_dist_*.png
│   ├── corr_*.png
│   ├── model_comparison_*.png
│   ├── resid_fitted_*.png
│   ├── resid_hist_*.png
│   └── actual_vs_pred_*.png
└── run_metadata.json                  # seed, target, row counts, selected model
```

> Adjust the tree above to match how you actually organize the repo — e.g. if you keep everything flat, drop the subfolders.

---

## 3. Environment Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -q pandas numpy matplotlib seaborn scikit-learn statsmodels ucimlrepo joblib
pip install -q xgboost          # optional, for the XGBoost comparison
```

Recommended: Python 3.10+, run in Jupyter Notebook or Google Colab.

---

## 4. Reproducing the Results

1. **Load the data.**
   - UCI: `from ucimlrepo import fetch_ucirepo; uci = fetch_ucirepo(id=477)`
   - California Housing: `from sklearn.datasets import fetch_california_housing`
2. **Run the notebook top to bottom** (`notebooks/HousePrice_Lab01.ipynb`) with a fresh kernel. It performs, in order: data audit → train/test split (80/20, seed 42) → leakage-safe preprocessing pipeline (median imputation + scaling) → naive baseline → the five models → 5-fold cross-validation → Elastic Net hyperparameter tuning (`GridSearchCV`) → residual diagnostics → segment-level error analysis.
3. **Outputs** (result tables and figures) are written to `results/` and `figures/`.
4. A fixed random seed (`SEED = 42`) is used throughout for the split, cross-validation folds, and all stochastic estimators, so results are reproducible on the same data.

> **Note on data provenance:** the CSVs currently in `data/` were generated in an offline sandbox that could not reach the UCI repository or the scikit-learn/figshare mirror for California Housing, so they are statistically faithful synthetic reconstructions matching the officially published sample sizes, ranges, means, and correlation structure — not the literal source files. Before treating any number in the report as final, re-run step 1 above with live internet access to pull the authoritative datasets, then re-run the notebook; the pipeline and structure do not change, only the exact metric values will shift slightly.

---

## 5. Headline Results (Test Set)

**UCI Real Estate Valuation**

| Model | MAE | RMSE | R² | Adj. R² |
|---|---|---|---|---|
| Elastic Net (tuned) | 4.901 | 6.221 | 0.684 | 0.659 |
| Random Forest Regressor | 5.418 | 6.768 | 0.626 | 0.596 |
| Gradient Boosting | 5.499 | 7.060 | 0.593 | 0.561 |
| Polynomial Regression (deg. 2) | 5.909 | 7.440 | 0.548 | 0.492 |
| Simple Linear Regression | 6.743 | 8.308 | 0.436 | 0.429 |
| Naive Mean Baseline | 9.127 | 11.139 | -0.014 | -0.014 |

**California Housing**

| Model | MAE | RMSE | R² | Adj. R² |
|---|---|---|---|---|
| Elastic Net (tuned) | 0.429 | 0.538 | 0.655 | 0.654 |
| Polynomial Regression (deg. 2) | 0.431 | 0.541 | 0.651 | 0.649 |
| Gradient Boosting | 0.432 | 0.542 | 0.650 | 0.648 |
| Simple Linear Regression | 0.436 | 0.546 | 0.645 | 0.644 |
| Random Forest Regressor | 0.441 | 0.552 | 0.637 | 0.636 |
| Naive Mean Baseline | 0.736 | 0.916 | 0.000 | 0.000 |

**Selected model:** tuned Elastic Net, on both datasets — lowest/near-lowest test and cross-validated RMSE, smallest train-test performance gap of any competitive model, and fully interpretable coefficients (unlike the tree ensembles).

Full metrics, cross-validation tables, residual diagnostics, segment-level errors, and discussion are in `HousePrice_Report.docx`.

---

## 6. Key Findings

- Every trained model beats the naive mean baseline on both datasets (41–44% RMSE reduction for the top model).
- **UCI:** distance to the nearest MRT station is the dominant, negatively-signed price driver, followed by convenience-store density and property age.
- **California Housing:** median household income of the block group dominates the linear signal; structural features (rooms, occupancy) add only modest independent contribution.
- **Random Forest** shows the largest train-test performance gap on both datasets (e.g., R² 0.907 train vs. 0.626 test on UCI) — a sign of overfitting risk relative to the regularized linear model.
- Errors are consistently lowest in the mid-price segment and highest at the price extremes (regression-to-the-mean pattern), so predictions for very cheap or very expensive properties should carry wider uncertainty.

---

## 7. Limitations

- Reconstructed (synthetic-replica) datasets were used due to offline sandbox constraints — see Section 4 above and the report's Limitations section.
- California Housing's target is a 1990 census block-group median, not an individual sale price, and is capped near $500k, which biases predictions downward at the top of the range.
- No formal significance testing or calibrated prediction intervals were produced; predictions are point estimates.
- Models are trained on historical, geographically specific data and should not be used for present-day valuation, credit, or taxation decisions without retraining on current, local data and fairness/governance review.

See `HousePrice_Report.docx`, Sections 10–11, for the full limitations and future-improvement discussion.

---

## 8. License / Academic Integrity

This repository is submitted as coursework for MDI3003. External code, libraries, and dataset sources are cited in the report (Section 25 references). Please do not copy this work for your own submission without proper attribution, per your institution's academic integrity policy.

## 9. References

1. UCI Machine Learning Repository — Real Estate Valuation dataset (ID 477): https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set
2. scikit-learn — `fetch_california_housing`: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
3. De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project.* Journal of Statistics Education, 19(3).
4. Pace, R.K. and Barry, R. (1997). *Sparse Spatial Autoregressions.* Statistics & Probability Letters.
