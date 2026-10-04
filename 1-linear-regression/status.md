# Project 1 status

- Project: Linear Regression and Regularization.
- Dataset: Ames housing data; target is `SalePrice`.
- Notebook: `1-main.ipynb`.
- The duplicate template notebook `data/solution.ipynb` was removed; `1-main.ipynb` is the only project notebook.
- Current state: Questions 1 and 2 (a, b) solved in `1-main.ipynb`.
- Next step: Question 3.a (OLS on all prepared features).
- Validation: re-run the edited cells in VS Code (`helpers/Analysis/` no longer contains the old notes or `run_notebook.py`).
- Deadline: 14 October 2026.
- Completed Question 2.a: sklearn OLS on numerical predictors, with log- and USD-scale metrics exported to `output/q2a_ols_numerical_*.csv`.
- Completed Question 2.b (same cell as 2.a), outputs in `output/q2b_*`:
  - A (1022 x 36) has rank 34: `TotalBsmtSF = BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF` and `GrLivArea = 1stFlrSF + 2ndFlrSF + LowQualFinSF` hold exactly; two singular values ~1e-14, next smallest 9.91.
  - `np.linalg.solve` gives arbitrary, machine-dependent coefficients and huge/NaN SEs for these 8 features; intercept and other 27 slopes match sklearn.
  - Fitted values, sigma^2 = 0.02280 (df 986), in-sample MSE/R^2 match 2.a exactly.
  - `pinv` = sklearn = statsmodels (minimum-norm solution). statsmodels SEs are x0.99899 because it uses df = m - rank = 988.
  - Note: the 2.b code needs `statsmodels` in the notebook kernel.
