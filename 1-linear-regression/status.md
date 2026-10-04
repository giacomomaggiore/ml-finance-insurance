# Project 1 status

- Project: Linear Regression and Regularization.
- Dataset: Ames housing data; target is `SalePrice`.
- Notebook: `1-main.ipynb` (the only project notebook).
- Deadline: 14 October 2026.

## Completed

- Question 1.a: loaded 1,460 observations with 79 predictors (35 numerical, 44 categorical); `MSSubClass` is handled as categorical. Valid categorical `NA`/`None` levels are preserved.
- Question 1.b: `SalePrice` is strongly right-skewed (skewness 1.883; excess kurtosis 6.536), so the target is `log1p` transformed. The transformed skewness and excess kurtosis are 0.121 and 0.810.
- Question 1.c: used a reproducible 70/30 split (`random_state=42`): 1,022 training and 438 test observations. Raw splits are retained as `X_train_raw` and `X_test_raw`.
- Question 1.d: fitted numerical mean imputation and standardization, categorical modal imputation, and one-hot encoding using only training data; test dummy columns are aligned to the training columns. The prepared matrices are asserted to have identical columns. The final prepared shapes are not printed.
- Question 2.a: fitted sklearn OLS with numerical predictors. Metrics are reported on log and USD scales and saved to `output/q2a_ols_numerical_*.csv`.
  - Log scale: train MSE/R² = 0.02200 / 0.85810; test MSE/R² = 0.02187 / 0.87109.
  - USD scale: train MSE/R² = 1,433,790,871.40 / 0.76177; test MSE/R² = 829,116,179.68 / 0.88118.
- Question 2.b: completed matrix-algebra OLS, uncertainty, rank, pseudoinverse, and statsmodels checks; results are saved to `output/q2b_*`.
  - The training design matrix is 1,022 × 36 with rank 34. Two exact dependencies involve the basement-area and living-area totals; the two smallest singular values are about $10^{-14}$, while the next is 9.91.
  - `np.linalg.solve` gives arbitrary coefficients and huge/NaN standard errors for the eight collinear area variables. Fitted values and the remaining coefficients agree with sklearn.
  - In-sample matrix-algebra metrics match Question 2.a; $\hat\sigma^2 = 0.02279841$ using 986 residual degrees of freedom.
  - The Moore–Penrose pseudoinverse matches sklearn and statsmodels coefficients. Statsmodels uses rank-based residual df = 988, producing standard errors scaled by 0.99899 relative to the matrix-algebra convention.

- Question 3.a: OLS with all features was evaluated. Categories increase test MSE by $46,429,693 and reduce test $R^2$ by 0.0067; numerical-only OLS generalizes slightly better.
- Question 3.b: completed leakage-safe 8-fold CV from raw splits for truncated pseudoinverse, Ridge, Lasso, and Elastic Net. Every selected hyperparameter is inside its final grid, so no grid extension is required.
  - Selected settings: TPI `n_components=125`; Ridge `alpha=10`; Lasso `alpha=0.001`; Elastic Net `alpha=0.0031623`, `l1_ratio=0.25`.
  - Test MSE/R²: TPI 713,780,121 / 0.89771; Ridge 627,306,166 / 0.91010; Lasso 584,831,618 / 0.91619; Elastic Net 614,607,752 / 0.91192.
- Questions 3.c–3.e: completed concise explanations, coefficient/direction counts, and final recommendation.
  - Lasso has 94 non-zero feature coefficients; Elastic Net has 111; Ridge has 313; truncated pseudoinverse retains 125 singular directions.
  - Lasso is recommended: test MSE $584,831,618 and test $R^2 = 0.9162$, the best held-out result.

## Validation notes

- Question 3 completed successfully after increasing Lasso/Elastic Net `max_iter` to 100,000, narrowing their alpha grids to $10^{-4}$–$1$, and suppressing expected convergence warnings.
- The final model-comparison table and all printed Question 3 conclusions are saved in the notebook output.
- The Question 1 and 2 cells retain outputs, but only the Question 2 cell has a current successful execution count. Re-run the notebook in order before submission.
- Question 2.b requires `statsmodels` in the notebook kernel.
