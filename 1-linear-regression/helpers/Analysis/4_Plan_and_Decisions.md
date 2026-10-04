# Project 1 — Plan, decisions and checks

This file turns the analysis into a work plan for the notebook. Each "step" has the same label as a cell (and a `PREP NOTE`) in `1-main.ipynb`.

- **How** = how to do it · **Show** = what must be visible · **Check** = numbers from the prototype runs (split `random_state=42`, from `scripts/numeric_checks.py`) · **Answer points** = the arguments the written answer should contain. The points are material for the answers, not the final wording.
- The **Check** numbers are only there to verify the code. They are **never** a reason to choose a model: the 8-fold CV chooses, and the test set only confirms at the end.

---

## 1. Decisions to confirm next week

| # | Decision | Recommendation | Why | Alternative |
| --- | --- | --- | --- | --- |
| D1 | Target transformation | **natural log**, `np.log(SalePrice)` | skew 1.88 → 0.12; Box-Cox λ ≈ −0.06 ≈ 0; no parameter to fit (no leakage); coefficients read as % effects | Box-Cox with λ fitted on the training set |
| D2 | Random seed | **`SEED = 42`**, fixed before any result, used for the split, the folds and TruncatedSVD | reproducible; not tuned | any value; the robustness check covers split dependence |
| D3 | `NA` / `None` | levels as listed in the description; **Ids 949 and 333** (basement present) → missing | faithful to the description and to the houses | all 14 variables' `NA` = level (differs on 2 cells only) |
| D4 | Where the `NA` rule lives | one **stateless function** `recode_missing(X)`, used in 1.d and as the **first pipeline step** (`FunctionTransformer`) | one rule, reused; satisfies 3.b.i ("including the special treatment of NA/None") | recode once at loading (then the pipeline does not contain it) |
| D5 | Standardization details | impute **first**, then z-score with **`ddof=0`** | the course standardizes with 1/m (Notes 2.7) = `ddof=0` = `StandardScaler`, so Q3 reproduces 1.d **exactly** | pandas default `ddof=1` (0.05% different, breaks the exact check) |
| D6 | Encoding | **full one-hot** (no reference level dropped), `dtype=float`, categories learned on training data | the assignment says one-hot; penalized models need no reference level | dummy encoding (`drop_first`) for OLS only |
| D7 | OLS solver | `LinearRegression` (sklearn) for 2.a and 3.a; numpy `solve`/`pinv` only in 2.b as asked | sklearn's `lstsq` detects rank deficiency (`rank_`) | `np.linalg.pinv` with default cutoff — **fragile on the full design** (see 3.a) |
| D8 | R² | `r2_score` in all tables; mention the course's R²_os (training mean) once | template uses `r2_score`; difference < 0.001 here | extra column with the course version |
| D9 | TruncatedSVD | `TruncatedSVD(algorithm="arpack", random_state=SEED)` | exact singular vectors, reproducible | default `"randomized"` is approximate (then at least fix `random_state`) |
| D10 | One-hot output | `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` | dense = same matrix as `get_dummies`; data is small | sparse output (then the exact check needs `.toarray()`) |
| D11 | Outliers (Ids 524, 1299) | **keep** them; discuss; optional sensitivity check | not asked; removing test points is never allowed | drop houses > 4000 sq ft from the *training* set as a robustness check only |
| D12 | Back-transformation | plain `np.exp(prediction)` | the assignment says "inverse-transform" | Duan smearing (mention in Limitations) |

---

## 2. Helper functions to write (the "code quality" marks)

Write them once, near the top of the notebook (after Setup), each with a one-line docstring:

| Helper | Purpose | Used in |
| --- | --- | --- |
| `parse_description(path)` | 35 numerical / 44 categorical names and the listed levels, read from `data_description.txt` | 1.a |
| `load_housing(path)` | read the CSV safely (`keep_default_na=False`, numerical `NA` → NaN, `Id` as index, MSSubClass → str); return X, y | 1.a |
| `recode_missing(X)` | stateless `NA` rule (D3) | 1.d, 3.b.i |
| `prepare_features(X_train_raw, X_test_raw)` | 1.d by hand: impute → standardize → one-hot → align | 1.d, 2, 3.a |
| `regression_metrics(model, X_train, X_test)` | MSE and R² in/out of sample, **log and USD** scale, as one row | 2.a, 3.a, 3.b.v, table |
| `ols_matrix(A, y, method)` | β̂, σ̂², SE with `method="solve"` or `"pinv"` | 2.b |
| `make_preprocessor()` | the 3.b.i component (recode → ColumnTransformer) | 3.b |
| `tune(estimator, grid, svd=False)` | build the pipeline, run `GridSearchCV` with the shared `cv`, return the search | 3.b.iii |
| `grid_edges(search, grid)` | which selected values sit on the first/last grid point | 3.b.iv |
| `cv_summary(search)` | best params, mean CV MSE (sign flipped) and std at `best_index_` | table |

---

## 3. Step by step

### Setup

- **How:** imports and constants in one cell (already in the scaffold). Print the package versions — this helps reproducibility.

### 1.a — Import

- **How:** `load_housing` as in D4/`2_Data_Findings.md` §2; the lists from `parse_description`; `X = df.drop(columns="SalePrice")`, `y = df["SalePrice"]`.
- **Show:** `X.shape == (1460, 79)`, `y.shape == (1460,)`; `len(NUMERICAL) == 35`, `len(CATEGORICAL) == 44` (use `assert`); a short line that MSSubClass is categorical.
- **Check:** 872 NaN in MasVnrType with pandas defaults vs 8 genuinely missing with the safe read.

### 1.b — Is SalePrice Gaussian?

- **How:** 2 × 2 figure: histogram with the fitted normal density + normal Q-Q plot (`scipy.stats.probplot`), for SalePrice and for log(SalePrice). Optional: skewness and Box-Cox λ on the training data.
- **Show:** the figure (see `figures/target_distribution.png`); skewness numbers.
- **Check:** skew 1.881 → 0.121; excess kurtosis 6.51 → 0.80; Box-Cox λ −0.077 (full) / −0.063 (training).
- **Answer points:**
  - not Gaussian: long right tail, mean > median, the Q-Q plot bends upwards
  - apply the natural log; after it the histogram is almost symmetric and the Q-Q plot almost straight (a slightly heavy lower tail: a few very cheap houses)
  - why it matters: (1) the OLS inference (SEs, t-tests, CIs in 2.b) assumes Gaussian errors with constant variance; price errors grow with the price level → the log stabilizes the variance; (2) the MSE on raw prices is dominated by a few expensive houses; (3) effects become multiplicative (% per unit), which is economically sensible; (4) predictions after `exp` are always positive
  - the log is parameter-free → no information from the test set can leak into the transformation
  - honest note: with n = 1460 a formal test still rejects exact normality; the question is graphical, and the log is approximately Gaussian

### 1.c — Split

- **How:** `X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)`; then `y_train_log = np.log(y_train)`, `y_test_log = np.log(y_test)`. Keep `X_train_raw`/`X_test_raw` untouched (make copies inside the helpers).
- **Show:** the shapes (1022 / 438).
- **Check:** Ids 524, 1299, 1183 → training; Id 692 → test.

### 1.d — Prepare X (training statistics only)

- **How:** `prepare_features` = `recode_missing` → numerical: fill with the training mean, then z-score with the training mean and `std(ddof=0)` → categorical: fill with the training mode → `pd.get_dummies(..., dtype=float)` on train and test separately → `X_test = X_test.reindex(columns=X_train.columns, fill_value=0)`.
- **Show:** final shapes; `assert list(X_train.columns) == list(X_test.columns)`; the missing values after preparation = 0; which test levels were unseen in training (5 levels).
- **Check:** train (1022, 311), test (438, 311); unseen: Condition2_PosA, Condition2_RRNn, Electrical_Mix, Exterior1st_ImStucc, RoofMatl_Membran; training means LotFrontage 70.4, MasVnrArea 105.3, GarageYrBlt 1978.7.
- **Answer points (short note):** why training-only statistics (leakage); GarageYrBlt's NaN means "no garage" but the assignment's rule (mean) is applied — the "no garage" dummy carries that information anyway.

### 2.a — OLS on the 35 numerical features

- **How:** `LinearRegression()` on `X_train[NUMERICAL]`, `y_train_log`; coefficient table (feature, coefficient; optionally also `exp(coef) − 1` = % change of the price per +1 standard deviation); `regression_metrics` for both scales.
- **Show:** the coefficient table, the 2 × 2 × 2 metrics (in/out × MSE/R² × log/USD).
- **Check:**

| | log MSE | log R² | USD MSE | USD R² |
| --- | ---: | ---: | ---: | ---: |
| in-sample | 0.02200 | 0.8581 | 1.434e9 | 0.7618 |
| out-of-sample | 0.02187 | 0.8711 | 8.291e8 | 0.8812 |

  sklearn `rank_` = 33 (35 centred columns, 2 dependencies). Out-of-sample R² with the course definition: 0.88121 (USD) vs `r2_score` 0.88118.

- **Answer points:**
  - log scale = relative errors (RMSE 0.148 ≈ a typical error of 15%); USD scale = dollar errors, driven by expensive houses (out-of-sample RMSE ≈ $28.8k)
  - the model minimizes squared error **on the log scale** → the log metrics are its "own" criterion; the USD metrics answer the practical question
  - **the in-sample USD MSE is larger than the out-of-sample one** (R² 0.76 < 0.88) — unusual. Reason: the two cheap giants (Ids 524, 1299) are in the training set. The model predicts about $854k (Id 1299, sold for $160k) and $668k (Id 524, sold for $185k); these two squared USD errors dominate the in-sample MSE. On the log scale the effect is much smaller → a direct illustration of point (2) in 1.b
  - the 8 area coefficients (basement pieces, floor pieces, and the two totals) are not unique → do not interpret them one by one (see 2.b.iv)

### 2.b — OLS with matrix algebra (training set only)

- **How:** `A = np.column_stack([np.ones(m), X_train[NUMERICAL].to_numpy()])`, `y = y_train_log.to_numpy()`.
  - (i) `beta = np.linalg.solve(A.T @ A, A.T @ y)`
  - (ii) `sigma2 = resid @ resid / (m - (d + 1))`; `se = np.sqrt(sigma2 * np.diag(np.linalg.solve(A.T @ A, np.eye(d + 1))))` (the diagonal of the inverse is needed here; computing it with `solve` against I is fine)
  - (iii) MSE and R² from `A @ beta`
  - (iv) `np.linalg.matrix_rank(A)`, `np.linalg.svd(A, compute_uv=False)`; a log-scale plot of the 36 singular values; the right singular vectors of the two smallest (`Vt[-2:]`) to name the dependent columns
  - (v) `beta_p = np.linalg.pinv(A) @ y`, and the SEs with `np.linalg.pinv(A.T @ A)`
  - (vi) `sm.OLS(y, A).fit()` with the feature names; `.params`, `.bse`, `.summary()`
- **Check:**
  - rank(A) = **34** < 36; singular values: largest 84.9, then … 11.8, 11.0, 9.92, then **1.4e-14 and 6.3e-15**
  - cond(A) ≈ 1e16; cond(AᵀA) = cond(A)² ≈ 2e32 ≫ 1/eps ≈ 4.5e15; without the two zero directions cond ≈ 8.6 (the rest of A is fine)
  - the two null vectors load **only** on BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, 1stFlrSF, 2ndFlrSF, LowQualFinSF, GrLivArea
  - `solve` raises **no error and no warning**
  - the 28 other coefficients and their SEs are identical in (i) and (v) (differences ~1e-14)
  - the 8 area coefficients differ; their `solve` SEs are **~1e4–1e5** and change between computers (they are rounding noise)
  - in-sample MSE identical in (i), (v) and 2.a: 0.021995641108
  - σ̂² = 0.022799 (divisor 986); statsmodels `scale` = 0.022753 (divisor `df_resid` = 988 = m − rank)
  - statsmodels coefficients = pinv coefficients exactly; statsmodels SEs = pinv SEs × √(986/988) = × 0.998987
  - statsmodels summary note: "The smallest eigenvalue is … (about 1e-28; the exact value depends on the computer) … the design matrix is singular"
  - `pinv` default cutoff = 1e-15 × 84.9 = 8.5e-14 > both tiny singular values → both are treated as 0 (safe here; margin about ×6)
- **Answer points:**
  - (i)/(iii): the numbers match sklearn up to rounding — **but** only because every least-squares solution gives the same fitted values. The coefficients themselves are not unique
  - (iv): A is **not** full column rank: 34 < 36. Reason: two exact identities (`2_Data_Findings.md` §6). The two singular values ~1e-14 are zero up to rounding. AᵀA is singular (Case 2 of Notes 2.1), so (AᵀA)⁻¹ does not exist; `solve` returns one arbitrary solution determined by rounding. For the 8 area coefficients the formula SEs are meaningless. Apart from these two exact dependencies, A is well conditioned (cond ≈ 8.6): the problem is **exact redundancy**, not near-collinearity
  - (v): pinv drops the two zero directions → the **minimum-norm** least-squares solution (Notes 2.1 Lemma / 2.2 (vi)). Same fitted values → same σ̂²; same β̂ and SEs for the 28 non-affected coefficients; different β̂ and finite SEs for the 8 affected ones. The pinv SEs describe the minimum-norm estimator, i.e. the identifiable part of β, not the separate effect of each area variable. Identical when A has full column rank and is well conditioned (then A⁺ = (AᵀA)⁻¹Aᵀ); different when A is rank-deficient or so ill-conditioned that singular values fall below the cutoff
  - (vi): coefficients = pinv exactly (statsmodels uses the pseudoinverse); SEs smaller by the constant factor √(986/988), because statsmodels divides by m − rank(A) = 988, the correct residual degrees of freedom when the rank is 34; the assignment's formula uses m − (d + 1) = 986. Its warning is the same diagnosis as (iv)
  - optional remedy (1 cell): drop TotalBsmtSF and GrLivArea → rank 34 of 34, cond 9.4, solve = pinv, the same fitted values, and meaningful SEs

### Minimal sklearn workflow example

- Keep it as in the template. It runs as is.

### 3.a — OLS on all prepared features

- **How:** `LinearRegression().fit(X_train, y_train_log)`; `regression_metrics`; print `rank_` and the rank of `[1, X_train]`.
- **Check:**

| | log MSE | log R² | USD MSE | USD R² |
| --- | ---: | ---: | ---: | ---: |
| in-sample | 0.00775 | 0.9500 | 2.914e8 | 0.9516 |
| out-of-sample | 0.02220 | 0.8692 | 8.811e8 | 0.8737 |

  rank of `[1, X_train]` = 254 of 312 (58 exact dependencies); sklearn `rank_` = 253 of 311 centred columns.

- **Answer points:**
  - in-sample fit jumps (log R² 0.858 → 0.950), but out of sample it is slightly **worse** than 2.a (0.871 → 0.869 log, 0.881 → 0.874 USD) → the categorical features do **not** help plain OLS; the in/out gap widens from ~0 to a factor 3 in log MSE (0.0078 vs 0.0222) → **overfitting** (Notes 1.8: training error ≪ test error)
  - why: ~250 effective parameters for 1022 houses; one-hot creates many **rare** levels (35 levels have ≤ 2 houses in the full data) and OLS fits them almost exactly. The largest coefficients belong to levels with 1–3 training houses (the PoolQC block with only 5 pools, RoofMatl_ClyTile −0.89 with 1 house, Condition2_PosN −0.67 with 2)
  - concrete example: Id 811 (test) has a 648 sq ft pool; only 5 training houses have a pool; OLS predicts $575k for a $181k house (standardized PoolArea alone adds +4.3 in log)
  - the design is rank-deficient (dummy trap + duplicate "no garage/no basement" dummies + the 2 area identities) → the OLS coefficients are not unique; sklearn returns a minimum-norm solution. For **11 test houses** even the **prediction** depends on which least-squares solution is chosen (up to 0.20 in log ≈ 22% in price): the 6 houses with a level never seen in training (Ids 30, 549, 584, 399, 272, 1188) and 5 whose combination of dummies lies outside the row space of the training design (Ids 352, 683, 721, 347, 811). The in-sample fitted values of all least-squares solutions are identical → another sign of an ill-posed problem that regularization fixes (ridge has a unique solution)
- **Numerical warning (learned the hard way):** `np.linalg.pinv(A_full)` with its **default** cutoff gives **garbage** on this Mac. With Apple's Accelerate LAPACK, one of the 58 zero singular values is 5.5e-13, above the default cutoff 1.7e-13, and pinv inverts pure rounding noise. Use sklearn (as planned) or `np.linalg.pinv(A, rtol=max(A.shape) * np.finfo(float).eps)`. The real singular values start at 0.034, so the gap is huge.

### 3.b — Regularization with 8-fold CV

**3.b.i — preprocessor.** `Pipeline([("na_levels", FunctionTransformer(recode_missing, feature_names_out="one-to-one")), ("columns", ColumnTransformer([("num", mean-imputer → StandardScaler, NUMERICAL), ("cat", most_frequent-imputer → OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)], verbose_feature_names_out=False))])`.
- **Show (strong robustness evidence):** fit it on `X_train_raw` and compare with the 1.d matrices → **max difference 0.0** on train and test; the same 311 column names in the same order.

**3.b.ii — pipelines.** `("preprocess", make_preprocessor())` → `("model", Ridge()/Lasso(max_iter=…)/ElasticNet(max_iter=…))`; TPI: `("preprocess", …)` → `("svd", TruncatedSVD(algorithm="arpack", random_state=SEED))` → `("model", LinearRegression())`.
- **Check:** the 8 folds have 305–310 features after one-hot and rank 246–251 → `n_components` must stay **below ~246** (above the rank TPI turns back into OLS and becomes unstable; above the number of features arpack fails).

**3.b.iii — grids and tuning.** Create `cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)` **once** and pass the same object to all four searches; `scoring="neg_mean_squared_error"`, `refit=True`, `n_jobs=-1`.

| Method | Scan range that worked | Scan optimum | CV MSE (log) ± std | Scan runtime (4 cores) |
| --- | --- | --- | --- | --- |
| Ridge | `alpha`: 29 values in 1e-3 … 1e4 | ≈ 17.8 | 0.0225 ± 0.0112 | 5 s |
| Lasso | `alpha`: 21 values in 1e-6 … 1e-1 | ≈ 5.6e-4 | 0.0228 ± 0.0127 | 42 s |
| Elastic Net | `alpha`: 21 values in 1e-5 … 1 × `l1_ratio` ∈ {0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1} | α ≈ 3.2e-3, l1_ratio 0.25 | 0.0225 ± 0.0115 | **≈ 300 s** |
| TPI | `n_components`: 10, 15, …, 245 | 150 | 0.0228 ± 0.0091 | 12 s |

- For the final notebook: keep these ranges but make the Elastic Net grid smaller (e.g. 15 α values × `l1_ratio` ∈ {0.05, 0.1, 0.25, 0.5, 0.75, 1.0}), or it dominates the run time. Use `max_iter` ≥ 50,000 for Lasso/EN; the scans had **no** convergence warnings with 100,000.
- The scans are in `scripts/outputs/cv_scan_*.csv` and `figures/cv_curves.png`.

**3.b.iv — boundary check.**
- **How:** for each search, look at the position of each selected value in its grid; flag "first/last point" (at the edge) and "second/second-to-last" (near the edge). Plot mean CV MSE vs each hyperparameter.
- **Show:** a small **grid record** table (method · parameter · grid min · grid max · number of points · selected value · at/near edge?) — the TA asked to "keep a concise record of grids and selected values". If a grid was extended, show both the first and the final range.
- **Answer points:**
  - all scan optima are **interior**, and the CV curves rise clearly on both sides (U-shape) → the ranges are wide enough
  - justify the ranges: from "almost no penalty" (close to OLS) to "so much penalty that the model is almost the constant ȳ" (CV MSE → 0.155 = the variance of log price); for TPI from 10 directions up to just below the rank of every fold
  - curiosities to mention: at very small α, Ridge and Lasso show a **hump** (CV MSE 0.027 at α = 1e-3, 0.032 near α = 0.03, minimum 0.0225 at 18), so the CV curve is not monotone near OLS; the Elastic Net surface has a long flat **valley** (l1_ratio 0.01–0.25 all reach ≈ 0.0225) → l1_ratio is only weakly determined; the TPI curve is **jagged** (many local minima, 100–155 all ≈ 0.023)
  - if a new grid puts the optimum at an edge → extend by 1–2 decades on that side and run again (that is exactly what the assignment wants to see)

**3.b.v — evaluation.** `search.best_estimator_` (already refitted) → `regression_metrics` on USD scale; compare with 2.a and 3.a.

**Summary table** (one DataFrame, one row per model):

| Model | Selected hyperparameter(s) | Mean CV MSE (log) | Std CV MSE (log) | In-sample MSE (USD) | In-sample R² (USD) | Out-of-sample MSE (USD) | Out-of-sample R² (USD) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OLS, numerical (2.a) | – | – | – | … | … | … | … |
| OLS, all features (3.a) | – | – | – | … | … | … | … |
| Truncated PI | n_components = … | … | … | … | … | … | … |
| Ridge | alpha = … | … | … | … | … | … | … |
| Lasso | alpha = … | … | … | … | … | … | … |
| Elastic Net | alpha = …, l1_ratio = … | … | … | … | … | … | … |

- `mean CV MSE = -cv_results_["mean_test_score"][best_index_]`, `std = cv_results_["std_test_score"][best_index_]` (no sign flip)
- **Check:** the CV std is large (~half the mean) because **one fold holds Id 1299** (fold MSE ≈ 0.05 vs 0.013–0.027 for the others). Say this when you report the std

**Intercept — answer points:** see `3_Course_Theory.md` §7: the intercept only sets the average level (b₀ = ȳ with centred features, Notes 2.7); penalizing it would shrink predictions towards log-price 0 = $1, an arbitrary point; it would break shift invariance and the zero mean of the residuals; the penalty is meant to control complexity (slopes), not the level. sklearn: `fit_intercept=True` centres the data and leaves the intercept unpenalized; for TPI the intercept is fitted by `LinearRegression` after the SVD, so it is never truncated.

### 3.c — `model__alpha`, `refit=True`

- **Answer points:** `step__parameter` addresses a parameter of a named pipeline step (`model__alpha` → `alpha` of the step `"model"`; nesting: `preprocess__columns__num__...`). `refit=True`: after the search, the best configuration is fitted **once more on the whole training set** (all 8 folds, including the preprocessing statistics) → `best_estimator_`, used by `search.predict`. Useful because it uses all training data (each CV fit saw only 7/8), gives one final model, and keeps the test set untouched until this single final evaluation.

### 3.d — Sparsity

- **How:** `(np.abs(search.best_estimator_.named_steps["model"].coef_) > 0).sum()` for Lasso, EN and Ridge (the intercept is separate in `intercept_`). TPI: `search.best_params_["svd__n_components"]`; optionally show that `V_k @ gamma` gives a dense β.
- **Answer points:** Lasso/EN set many coefficients exactly to 0 → feature selection (the L1 corners, Notes 2.5); Ridge keeps all 311 non-zero (only shrinks); EN keeps more than Lasso (the L2 part keeps correlated features together); TPI keeps k directions = linear combinations of **all** features, chosen from X only (unsupervised), and its β is dense → dimension reduction, not feature selection ("Lasso sparsifies feature coefficients; truncated SVD removes directions in the design space").

### 3.e — Recommendation (write it after seeing the final table)

- **Choose by the CV MSE first** (the criterion that was not fitted on the test set), then check that the test metrics agree. In the scans, Ridge and Elastic Net had the lowest CV MSE, but all four were within 0.0004 of each other — much less than the CV std. Honest conclusion: several regularized models are **statistically indistinguishable**; decide with the problem structure.
- **Arguments from the problem structure:** 311 columns for 1022 houses; many rare one-hot levels; exact and strong collinearity (area identities, garage cars/area, the duplicate "no garage" dummies) → OLS is unstable/unidentified → shrinkage is needed. Ridge handles collinearity smoothly (unique solution, keeps all dummies, groups correlated effects) but does not select; Lasso gives a sparse, more interpretable model but picks one feature of a correlated group somewhat arbitrarily; Elastic Net combines both (sparsity + grouping). TPI works on directions that are not tied to single features → harder to interpret, jagged CV curve.
- **Limits to mention:** all four are **linear** in the (standardized, one-hot) features → no interactions (neighbourhood × quality), no nonlinearity except through the log target; outliers; the back-transform gives a median-type prediction.
- Add the uncertainty of the test error (Notes 1.8): with n = 438, the standard error of a test MSE is sd(squared errors)/√438. Compute it; differences between the regularized models will most likely be inside it.

### Robustness checks (optional section, evidence for the report)

| Check | How | Cost |
| --- | --- | --- |
| R.1 no leakage + exact reproduction of 1.d | the max-difference check of 3.b.i (0.0) | seconds |
| R.2 grid adequacy | CV curves as in `figures/cv_curves.png` + the edge report | from the searches |
| R.3 split dependence | repeat split + tuning for 5 seeds with slimmer grids around the optima; mean ± sd of the test metrics and how often each model is best | ~3–5 min with `n_jobs=-1` |
| R.4 unusual houses | fold-wise CV MSE (fold with Id 1299 ≈ 0.05); largest test errors; optional: refit the chosen model without Ids 524, 1299 in the training set | seconds |
| R.5 test-error uncertainty | 95% CI of each test MSE: mean ± 1.96·sd/√n (Notes 1.8) | seconds |

### Project Report — outline

- **Approach:** data (1460 houses, 35 + 44 features), log target, the `NA`/`None` rule, 70/30 split, preprocessing with training statistics only (the same code inside the pipelines), the model sequence (OLS numerical → OLS all → TPI/Ridge/Lasso/EN).
- **Model selection:** 8-fold CV on the training set with the same folds, MSE on the log scale, grids and the edge check, refit on the full training set, one final test evaluation.
- **Key results:** the summary table in compact form + 3 numbers (best CV MSE, out-of-sample USD R² of the chosen model vs the two OLS, sparsity count, TPI's k).
- **Interpretation:** what drives prices (largest standardized coefficients: overall quality, living area, age, neighbourhood); why OLS with all features overfits; the rank deficiency found in 2.b and what it means for inference; log vs USD metrics. Use the TA's framing: Question 2 is the **inference** view (coefficients, standard errors, rank — statsmodels), Question 3 the **prediction** view (CV, test error — scikit-learn).
- **Robustness:** leakage prevention (exact reproduction, pipeline inside CV), numerical stability (rank, pinv vs solve, statsmodels df, the pinv-cutoff lesson), grid adequacy (interior optima, curves), fold variability (outlier fold), optional R.3–R.5.
- **Limitations:** linear model (no interactions/nonlinearity; feature engineering such as log of skewed areas would help); single split and a small test set (438); outliers kept; simple imputation (GarageYrBlt); exp(prediction) estimates a median, not a mean (smearing factor ≈ 1.01); OLS inference assumes homoscedastic Gaussian errors.

---

## 4. Presentation standards (what makes the grader's job easy)

- **One number format everywhere:** log-scale MSE with 5 decimals (0.02187); USD MSE in scientific notation (8.29e+08) **plus** RMSE in dollars ($28.8k), which is easier to read; R² with 4 decimals; CV MSE as mean ± std.
- **The same model names in every table and plot:** "OLS (numerical)", "OLS (all features)", "Truncated PI", "Ridge", "Lasso", "Elastic Net".
- **Tables as pandas DataFrames**, not long `print` blocks. At most one screen of output per cell.
- **Every figure** has a title, axis labels (with units: USD, log USD) and a legend when there are several lines.
- **No warnings in the final outputs** (e.g. `ConvergenceWarning`): fix the cause (`max_iter`). Do not hide warnings globally.
- **Checks as `assert`s** (shapes, identical columns, the 1.d = pipeline match): short and convincing.
- **Every answer next to its code**, 3–8 short sentences, with the numbers that support it. The report then only **synthesizes**.
- **Total run time** below about 10 minutes (keep the Elastic Net grid slim). Check with `helpers/Analysis/scripts/run_notebook.py`.

## 5. Pitfalls that fail silently

- reading the CSV with pandas defaults (`NA`/`None` → NaN)
- selecting numerical columns by dtype (36 instead of 35; misses MSSubClass)
- `select_dtypes("object")` finds no columns under pandas 3 (the new `str` dtype)
- computing any mean/std/mode on the full data or on the test set
- `std()` with the pandas default `ddof=1` → the pipeline check then shows differences of about 5e-4
- forgetting `reindex` → test columns in a different order (the model then silently uses the wrong coefficients)
- reporting `mean_test_score` without flipping the sign, or flipping the sign of the std
- reporting USD metrics computed on log predictions (forgetting `np.exp`), or R² on mixed scales
- a new `KFold` object per method with a different seed → not the same folds
- `TruncatedSVD` default algorithm without `random_state` → results change at every run
- `n_components` ≥ the number of features of a fold → arpack error
- `np.linalg.pinv` default cutoff on the full one-hot design (machine-dependent garbage)
- Lasso/EN `ConvergenceWarning` ignored (raise `max_iter`)
- using the test set to pick a hyperparameter or to "try again with a better grid"

## 6. Questions for the TA (Moodle, by 7 October) — optional

None of them blocks the work. For each, the plan above already has a defensible default.

1. Out-of-sample R²: the lecture defines SST with the training mean (MLFI 1.6), `r2_score` uses the test mean. Is `r2_score` fine? (Default: `r2_score`, mention the course version.)
2. Two basement `NA`s belong to houses that have a basement (Ids 333, 949). Is it fine to treat them as missing rather than as "No Basement"? (Default: yes, as missing.)
3. The two "Partial" sales over 4000 sq ft (Ids 524, 1299): keep them in the training set? (Default: keep, discuss, optional sensitivity check.)

Draft post (only if you decide to ask; posting is your call):

> Dear Zhexin, three short questions on Project 1. (1) For the out-of-sample R², is `sklearn.metrics.r2_score` fine, or should the total sum of squares use the training mean as in lecture 1.6? (2) Two houses (Ids 333 and 949) have a basement but `NA` in BsmtExposure / BsmtFinType2. Is it fine to treat these two values as missing instead of "No Basement"? (3) Should the two large "Partial" sales (Ids 524 and 1299, over 4000 sq ft) stay in the data? Thank you!

## 7. Agenda for the working session next week (about 3–4 hours)

1. 15 min — group names; confirm the decisions D1–D12; decide whether to post the TA questions.
2. 40 min — helpers + Question 1 (1.a–1.d), with the checks above.
3. 60 min — Question 2 (2.a, 2.b.i–vi) and the written answers.
4. 75 min — Question 3 (3.a, 3.b with the final grids, the table, 3.c–3.e).
5. 30 min — robustness cells R.1, R.2, R.4, R.5 (R.3 if time allows) and the Project Report.
6. 15 min — Restart & Run All in a fresh kernel, go through the checklist in `1_Requirements.md`, remove the `PREP NOTE` lines, export.
