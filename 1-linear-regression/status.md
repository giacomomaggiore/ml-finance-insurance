# Project 1 status

- Project: Linear Regression and Regularization.
- Dataset: Ames housing data; target is `SalePrice`.
- Notebook: `1-main.ipynb` (the only project notebook).
- Deadline: 14 October 2026.

## Completed

- Question 1.a: loaded 1,460 observations with 79 predictors (35 numerical, 44 categorical); `MSSubClass` is handled as categorical. Valid categorical `NA`/`None` levels are preserved. `NA` in `MasVnrType` and `Electrical` is not a documented level, so it is read as missing.
- Question 1.b: `SalePrice` is strongly right-skewed (skewness 1.883; excess kurtosis 6.536), so the target is `log1p` transformed. The transformed skewness and excess kurtosis are 0.121 and 0.810.
- Question 1.c: used a reproducible 70/30 split (`random_state=42`): 1,022 training and 438 test observations. Raw splits are retained as `X_train_raw` and `X_test_raw`.
- Question 1.d: fitted numerical mean imputation and then standardization (std of the imputed training data, ddof=0, as StandardScaler), categorical modal imputation, and one-hot encoding using only training data; test dummy columns are aligned to the training columns. The prepared matrices are asserted to have identical columns. The final shapes (1,022 × 311 and 438 × 311) are printed. A table shows the truly missing values (LotFrontage, MasVnrType, MasVnrArea, Electrical, GarageYrBlt) and how they are filled.
- Question 2.a: fitted sklearn OLS with numerical predictors. Metrics are reported on log and USD scales and saved to `output/q2a_ols_numerical_*.csv`.
  - Log scale: train MSE/R² = 0.02200 / 0.85810; test MSE/R² = 0.02187 / 0.87109.
  - USD scale: train MSE/R² = 1,433,790,871.40 / 0.76177; test MSE/R² = 829,116,179.68 / 0.88118.
- Question 2.b: completed matrix-algebra OLS, uncertainty, rank, pseudoinverse, and statsmodels checks; results are saved to `output/q2b_*`.
  - The training design matrix is 1,022 × 36 with rank 34. Two exact dependencies involve the basement-area and living-area totals; the two smallest singular values are about $10^{-14}$, while the next is 9.92.
  - `np.linalg.solve` gives arbitrary coefficients and huge/NaN standard errors for the eight collinear area variables. Fitted values and the remaining coefficients agree with sklearn.
  - In-sample matrix-algebra metrics match Question 2.a; $\hat\sigma^2 = 0.02279841$ using 986 residual degrees of freedom.
  - The Moore–Penrose pseudoinverse matches sklearn and statsmodels coefficients. Statsmodels uses rank-based residual df = 988, producing standard errors scaled by 0.99899 relative to the matrix-algebra convention. The statsmodels `summary()` is shown; its note [2] warns about the singular design.

- Question 3.a: OLS with all features was evaluated. Categories increase test MSE by 51,769,393 USD² (test RMSE 28,794 → 29,680 USD) and reduce test $R^2$ by 0.0074. Answer: no improvement with plain OLS; the log-scale train–test R² gap grows from −0.013 to +0.081. The all-feature design has rank 254 of 312.
  - One test house dominates: Id 811 (pool, `PoolQC` 'Fa'; only 5 pool houses in training) is predicted at $575,124 for $181,000 and causes 40% of the test squared USD error. Without it, all-feature OLS has test R² 0.925 vs 0.882 (diagnostic only).
- Question 3.b: completed leakage-safe 8-fold CV from raw splits for truncated pseudoinverse, Ridge, Lasso, and Elastic Net. No selected hyperparameter is at or near the edge of its final grid, so no grid extension is required.
  - The pipeline preprocessor reproduces the 1.d matrices exactly (assert). CV training folds have 305–310 columns, so `handle_unknown="ignore"` is needed.
  - Grids: TPI 10–250 directions in steps of 10; Ridge, Lasso and Elastic Net four alpha values per decade; Elastic Net l1_ratio 0.05–0.95 (steps of 0.05 up to 0.3). Ridge uses the default solver.
  - Selected settings: TPI `n_components=150`; Ridge `alpha=17.78`; Lasso `alpha=0.00056`; Elastic Net `alpha=0.0031623`, `l1_ratio=0.25`.
  - Mean CV MSE (log scale): TPI 0.02281; Ridge 0.02252; Lasso 0.02277; Elastic Net 0.02245.
  - Test MSE/R²: TPI 691.7M / 0.9009; Ridge 632.6M / 0.9093; Lasso 483.2M / 0.9308; Elastic Net 614.6M / 0.9119.
- Questions 3.c–3.e: completed concise explanations, coefficient/direction counts, and final recommendation.
  - Lasso has 122 non-zero feature coefficients; Elastic Net has 111; Ridge has all 311 (L2 norm 0.36 vs 3.97 for OLS); truncated pseudoinverse retains 150 singular directions (all 311 implied coefficients non-zero).
  - Elastic Net is recommended because it has the lowest mean CV MSE (0.02245). The test set is not used for the choice. All four regularized models are practically tied: CV MSE gap ≤ 0.00035; fold by fold Elastic Net is better in only 3–5 of 8 folds; every mean difference is ≤ 0.44 standard errors. Elastic Net: test RMSE ≈ $24,791, test $R^2 = 0.9119$.

## Validation notes

- Lasso/Elastic Net use `max_iter=100,000` and alpha grids from $10^{-4}$ to $1$. Every fit converges, so no warnings are suppressed.
- The final model-comparison table and all printed Question 3 conclusions are saved in the notebook output.
- Question 2.b requires `statsmodels` in the notebook kernel.
- Missing descriptions added (4 Oct 2026) as printed text, without structural changes:
  - 1.b: why to transform;
  - 1.d: shapes and the corrected one-hot comment;
  - 2.a: comparison of the two scales, with the outlier houses Id 1299 and Id 524;
  - Q3: intercept, grid ranges, 3.c, 3.d, and 3.e chosen by CV.
- NA fix and finer grids applied (4 Oct 2026). Backup of the notebook from before this change: `helpers/1-main-old-copy.ipynb`.
- Small polish applied (4 Oct 2026): see `TO_FIX.md`, round 3. Q3 results are unchanged; numerical coefficients are rescaled slightly by the std-after-imputation change.
- Charts added (4 Oct 2026, `TO_FIX.md` round 4): transformation comparison in 1.b (Box-Cox λ = −0.077 ≈ log), predicted vs actual in USD in 2.a, CV curves in 3.b(iv). No results changed.
- Q3 template restored and `%pip` cell removed (4 Oct 2026): a fresh Restart & Run All works (tested).
- Project Report rewritten (4 Oct 2026, `TO_FIX.md` round 6): six sections in simple English, Elastic Net recommended by CV, numbers from the latest run.
- Round 7 (4 Oct 2026, `TO_FIX.md`): 16 fixes after a second full check (Id 811 diagnostic, finer Lasso/EN grid, paired CV comparison, OLS comparison, statsmodels summary, report update). Backup before: `helpers/1-main-before-round7.ipynb`. Fresh-kernel run saved (counts 1–4, no warnings). Remaining: group names only.
- Round 8 (4 Oct 2026): Project Report shortened to about 800 words (same sections, table and numbers); Markdown only.
- Round 9 (4 Oct 2026, `TO_FIX.md`): 2.b part labels, residual/HC3 check in 2.b(vi), "at or near the edge" grid check with a finer l1_ratio grid, report fixes, small code polish. No results changed. Fresh-kernel run saved (counts 1–4, no warnings). Remaining: group names only.
