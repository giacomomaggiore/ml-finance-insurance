# Project 1 – what to fix before submission

Review of `1-main.ipynb` on 4 Oct 2026. Status: **ready except the names**. The code, the answers and the Project Report are done and verified, and the saved outputs come from a fresh-kernel run.

Backups: before the NA fix and the finer grids `helpers/1-main-old-copy.ipynb`; before round 7 `helpers/1-main-before-round7.ipynb`; before round 9 `helpers/1-main-before-round9.ipynb`.

## 1. Blockers (must fix)

- [ ] **Write the group names** in the "Please write the names…" cell. This is Markdown, so no re-run is needed.
- [x] **Final run:** done after round 9 from a fresh kernel (nbclient, `python3` kernel of miniconda: Python 3.13.11, same package versions). Counts 1, 2, 3, 4, no errors, no warnings, about 2 min.
  - If you change any code cell later, do Kernel → Restart, then Run All, then save again.

## Already OK (no change needed)

- 1.a–1.c, the 2.a table and metrics, all of 2.b, the 3.b pipelines, CV and table.
- All small-polish items are done (see "Changes made", round 3).

## Changes made (4 Oct 2026)

All changes are inside existing cells: no cells were added, removed or moved. Each version was tested on a copy in a fresh kernel, with the Q3 imports added for the test only.

**Round 1 – missing descriptions** (printed text)
- 1.b: why the transformation matters (inference assumptions, multiplicative effects, influence of expensive houses).
- 1.d: the final shapes are printed, and the wrong "k-1 bit" comment is corrected (`get_dummies` keeps all k levels).
- 2.a: comparison of the two scales. The in-sample R² in USD (0.762) is below the test R² (0.881) because of two training outliers: Id 1299 and Id 524 cause about 49% of the training squared error in dollars.
- Q3: expanded the answers on why the intercept is not penalized, the grid ranges (3.b.iv), 3.c and 3.d.
- 3.e: the model is now chosen by the lowest mean CV MSE, not the lowest test MSE. The recommendation is Elastic Net, with a full discussion of the problem structure. Ridge, Lasso and Elastic Net are practically tied.

**Round 2 – NA fix and finer grids**
- 1.a `read_csv`: `"NA"` in `MasVnrType` and `Electrical` is now read as missing. In 1.d and in the Q3 pipeline these values are imputed with the training mode, instead of becoming a fake "NA" level.
  - Encoded columns: 313 → 311. Q2 results are unchanged.
- 3.b grids:
  - TPI `svd__n_components`: `[25, 75, 125, 175]` → `range(10, 251, 10)`.
  - Ridge `model__alpha`: `np.logspace(-4, 5, 10)` → `np.logspace(-4, 5, 37)` (four values per decade).
  - The "Grid ranges (3.b.iv)" text now explains the refinement.
- The rank of the all-feature design is now computed (`rank_all` = 254 of 312 columns) and used in the 3.b.iv and 3.e texts, instead of the hard-coded "256 of 314".
- New results:

  | Model | Selected | Mean CV MSE | Test R² |
  |---|---|---|---|
  | OLS, all features | – | – | 0.874 (was 0.875) |
  | TPI | 150 directions (was 125) | 0.02281 | 0.901 (was 0.898) |
  | Ridge | α = 17.8 (was 10) | 0.02252 | 0.909 (was 0.910) |
  | Lasso | α = 0.001 | 0.02301 | 0.916 (unchanged) |
  | Elastic Net | α = 0.0032, l1_ratio = 0.25 | 0.02245 | 0.912 (unchanged) |

  - Elastic Net is still the CV winner, and every selected value lies inside its grid.
  - The hand-typed numbers in the 3.e text were re-checked and still hold.

**Round 3 – small polish**
- 1.b: the hard-coded copies of the statistics and the "should deviate" text are replaced by what the plots show. Fixed the typo "Trasformation".
- 1.c: removed the redundant `X_train_raw.copy()`, because 1.d already copies the raw splits. Fixed the printed labels (`y_train`, not `Y_train`; no double space).
- 1.d: mean imputation now comes first. The standard deviation is then computed on the imputed training data with `ddof=0`, exactly as `StandardScaler` does in Q3.
  - Only the scale of the numerical coefficients changes, e.g. OverallQual 0.11434 → 0.11428 and LotFrontage −0.00131 → −0.00118.
  - All metrics are unchanged.
- 2.b(i)/(ii): fixed the double spaces. The NaN text is now generic: which SEs are huge and which are NaN depends on the machine.
- 3.a:
  - the MSE change is reported in USD², with the RMSE in USD (28,794 → 29,680);
  - a new line gives the rank of the all-feature design (254 of 312) and the link to 2.b;
  - `rank_all` moved from 3.b.iv up to 3.a.
- Q3:
  - removed the second import of the metrics;
  - the summary table is shown as a formatted DataFrame, with readable hyperparameters and thousands separators;
  - the small helper `format_params` formats the hyperparameters for the table and for 3.e.
- All Q3 results are unchanged. Each change was tested in a fresh kernel, and the notebook is identical to the tested copy.

**Round 4 – transformation comparison and two charts** (inspired by the other group's draft, kept simple)
- 1.b: a small table compares the skewness and excess kurtosis of none, sqrt, log1p, 1/y and Box-Cox.
  - Box-Cox estimates λ = −0.077 from the data, which is close to 0, i.e. the log. This is a data-driven argument for log1p (skewness 0.121).
  - We did not copy the other group's 4×2 plot grid or Shapiro–Wilk: with 1,460 houses, Shapiro–Wilk rejects every transformation.
- 2.a: a "predicted vs actual" plot in USD, training and test side by side. Ids 1299 and 524 are marked in red, which shows why the in-sample R² in USD is the lower one. Saved to `output/q2a_predicted_vs_actual_usd.png`.
- 3.b(iv): CV curves taken from `cv_results_` (no refitting): mean CV MSE ±1 std across folds for TPI, Ridge, Lasso and Elastic Net (at the selected l1_ratio). The selected values are marked, together with the best CV MSE of all methods. Saved to `output/q3b_iv_cv_curves.png`.
  - The figure shows every optimum inside its grid. The ±1 std bands are much wider than the gaps between the methods, which confirms the tie. The "Grid ranges (3.b.iv)" text now says this.
- No results changed. The new code is about 30 simple lines inside existing cells.

**Round 5 – fixes after the full check**
- The `%pip install --upgrade --force-reinstall` cell was deleted.
- The Q3 template cell is restored inside the existing cell, with the imports and all 174 professor lines. Each template block sits above the code that answers it. No code was moved or changed.
  - A fresh "Restart & Run All" now works. It was tested on a copy without any patch: it ran in 54 s with no warnings, and the outputs are identical to the previous run.
- Optional:
  - Done in round 7: the old files in `output/` (`distribution_*.png`, `ols_numerical_*.csv`) were deleted.
  - Not done: on matplotlib ≥ 3.11, `boxplot(vert=False)` in 1.b prints a deprecation warning. `orientation="horizontal"` would fix it, but needs matplotlib ≥ 3.10.

**Round 6 – Project Report**
- Rewrote our report cell in simple English, with the numbers of the latest run (about 990 words including the table).
- Removed the `## Analysis Synthesis` heading. The report now uses six `####` sections (Approach, Model selection, Key results, Interpretation, Robustness, Limitations) under the professor's "Project Report" heading. The professor's guidance cell is unchanged.
- Fixes compared with the old report:
  - The recommendation is now **Elastic Net**, chosen by the lowest mean CV MSE, not Lasso by test score. The selection rule is stated: tune and choose with CV, and use the test set only once for the final evaluation.
  - Corrected "stable minimum-norm fitted values": the *coefficients* are smallest-norm, and the fitted values are the same for every solution.
  - Updated numbers: 311 columns, TPI 150 directions, Ridge α = 17.8, OLS all features test R² 0.874.
- New content:
  - a key results table with mean CV MSE, test R² and test RMSE (USD);
  - the CV tie between Ridge, Lasso and Elastic Net;
  - the two outliers (Ids 1299 and 524);
  - the strongest effects (quality +12%, year built +10%, garage capacity +6% per standard deviation);
  - the rank deficiency of both designs;
  - the grid refinement and reproducibility;
  - the retransformation (median vs mean) limitation.

**Round 7 – fixes after the second full check**

All changes are inside existing cells: no cells were added, removed or moved. Professor content is unchanged. The notebook was then run from a fresh kernel and saved (counts 1–4, no errors, no warnings).

- Q1:
  - `output_dir` is now created in the Q1 cell (moved from Q2). The two 1.b figures are saved as `output/q1b_saleprice_before_transformation.png` and `output/q1b_saleprice_after_transformation.png`.
  - 1.d: a small table shows the truly missing values (train/test counts) and how they are filled.
- 2.a:
  - The two tables use the existing `show_and_save` helper (about 20 lines less).
  - New "Units" line: log scale, effect per standard deviation, about 100·b %. It also warns that the 8 area coefficients are not unique.
- 2.b(vi): the statsmodels `summary()` is shown, with the feature names. Its note [2] is the singularity warning that the text mentions.
- 3.a:
  - Explicit answer: "no" on both scales. The train–test R² gap is now also given on the log scale (−0.013 → +0.081), because the USD gap is distorted by Ids 1299 and 524.
  - Diagnostic: test house **Id 811** dominates the error. It has a pool (648 sq ft, PoolQC 'Fa'), and only 5 training houses have a pool. The PoolArea coefficient goes from −0.021 to +0.266 per SD, so the all-feature OLS predicts $575,124 for a $181,000 house: 40% of the test squared USD error. Without it, the all-feature OLS has test R² 0.925 vs 0.882. The house stays in the test set.
  - The table is formatted with thousands separators. `num_metrics` now comes from `evaluate`.
- 3.b:
  - 3.b.i: a comment explains the NA/None treatment (inherited from `read_csv` in 1.a). An assert checks that the preprocessor reproduces the 1.d matrices exactly. The columns per CV training fold (305–310 vs 311) show why `handle_unknown="ignore"` is needed.
  - Grids: the Lasso and Elastic Net alpha grid is now `np.logspace(-4, 0, 17)`, four values per decade like Ridge. With two per decade, the Lasso optimum lay between grid points.
  - Ridge uses the default solver. `lsqr` was not exact at tiny alpha; the selected alpha is unchanged.
  - The ConvergenceWarning filter was removed, because every fit converges.
  - 3.b.v: a new printed comparison with both OLS models (test R², test RMSE, prediction for Id 811).
  - 3.b.iv: the Elastic Net panel title shows the l1_ratio, and the grid text is updated.
- 3.d:
  - TPI coefficients mapped back to the original columns: 311 of 311 are non-zero.
  - Ridge shrinkage: L2 norm 0.36, vs 3.97 for the all-feature OLS.
  - Sparsity is now explained by the selected L1 strengths. Elastic Net's α·l1_ratio = 0.00079 is larger than Lasso's α = 0.00056, so Elastic Net keeps fewer columns (111 vs 122).
- 3.e:
  - A fold-by-fold (paired) comparison on the same 8 folds. Elastic Net is better in only 3–5 of 8 folds, and every mean difference is ≤ 0.44 standard errors.
  - All four models, TPI included, are now in the CV tie (gap ≤ 0.00035), and an argument for Elastic Net over TPI was added.
  - The numbers in the text are computed, not typed.
- Project Report:
  - new Lasso and Ridge numbers, and Id 811;
  - the strongest effects are named from the 2.a model, with the living-area caveat;
  - TPI is in the CV tie, with the paired comparison;
  - Robustness mentions the assert, the grid refinement and convergence.
  - Length is now about 1,150 words.
- Changed results (all others unchanged, including the Elastic Net recommendation):

  | Model | Before | After |
  |---|---|---|
  | Lasso | α = 0.001, CV 0.02301, test R² 0.916, 94 non-zero | α = 0.00056, CV 0.02277, test R² 0.931, 122 non-zero |
  | Ridge | test MSE 632.7M (lsqr) | test MSE 632.6M (default solver), same α = 17.78 and R² 0.909 |

- Not done (optional): `boxplot` orientation (see round 5).

**Round 8 – shorter Project Report**
- Cut the report from about 1,150 to about 800 words (including the table). It keeps the same six sections, the same table and the same numbers.
- Removed the repetition between sections, for example the grids, leakage and outlier details that appeared twice.
- Markdown only: no code or outputs changed, and the fresh-run outputs (counts 1–4) are still valid.

**Round 9 – fixes after the third full check**

No cells were added, removed or moved. Professor content is unchanged (checked line by line against the template). All results are unchanged, including every number in the Q3 table and in the Project Report.

- Recommended:
  - 2.b: each part (i)–(vi) now has a printed label; "Task 2a/2b" → "Task 2.a/2.b".
  - 2.b(vi): residual check. Skewness −2.26 and kurtosis 25.6 (Jarque–Bera rejects normality), mostly from Ids 1299 and 524 (without them −0.36 and 6.5). So the classical SEs and t-tests are only approximate. Robust (HC3) SEs are typically 1.28× the classical ones, and 6.5× for PoolArea.
  - 3.b.iv: the check now flags values "at or near" the edge (two lowest or two highest grid values). The Elastic Net `l1_ratio` grid is `[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75, 0.95]`; the selection is unchanged (α = 0.0032, l1_ratio = 0.25). The best CV MSE per l1_ratio is 0.02245–0.02276.
  - Report: effects use the 100·b rule (+11%, +9%, +6%); one new Limitations line on the residuals and the HC3 SEs.
- Polish:
  - 1.b: comment that log1p has no fitted parameter (no leakage) and that Box-Cox is only a comparison; removed the repeated skewness/kurtosis prints.
  - 1.a: `Path(...).read_text()` instead of an unclosed `open()`.
  - 2.a: the metrics table is built with one loop; large values are shown with thousands separators; the Id 1299/524 numbers are computed, not typed.
  - 3.a: shorter source lines; the rank text is wrapped; the OLS table is saved to `output/q3a_ols_numerical_vs_all.csv`.
  - 3.b: long lines wrapped; the final table is saved to `output/q3b_v_model_comparison.csv`.
  - 3.d: one dict for the model coefficients instead of three repeated lines.
  - 3.e: notes that the paired SE understates the uncertainty, because folds share training data.
- Run time is now about 2 min (larger Elastic Net grid).
