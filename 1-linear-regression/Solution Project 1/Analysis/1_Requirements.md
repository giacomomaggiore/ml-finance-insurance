# Project 1 — Requirements

Sources: `Pr1_26.pdf` (the assignment), `MLFI_Project1_Introduction_2026.pdf` (TA session of 23 Sept 2026), `Project1 - Template.ipynb`, and `Slides/Info Projects.pdf` (general rules).

## 1. The facts

| Item | Value |
| --- | --- |
| Title | Project 1: Linear Regression and Regularization |
| In charge | Zhexin Wu (TA). Questions go to the Moodle Q&A forum. |
| First discussion | 23 September 2026 |
| **Last day for questions** | **7 October 2026** |
| **Deadline** | **14 October 2026**, at the latest by midnight |
| Team | up to 5 students; one submission per group; you may change team for each project |
| What you submit | **one fully executed Jupyter notebook** (Python), which should **closely follow the template** |
| Report | Markdown cells at the end, under the heading **Project Report** |
| Grading | **80%** implementation and results, **20%** Project Report |
| Weight in the course | the project average counts **30%** of the final grade (4 projects → this one is about 7.5%) |

> The `Info Projects.pdf` slides are from 2024 and show older dates (deadline 16 Oct). The 2026 assignment PDF is the one that counts: **14 October**.

**Code-quality rule (from the assignment).** The code must be readable and reproducible. Avoid duplicated code and use small reusable helper functions. Classes are not needed.

**Workflow rules the TA stressed in the intro session:**

- keep the test set special: use it only for the final evaluation, **never to choose settings**
- keep the original train/test split, because Question 3 goes back to the raw features
- use **the same 8 folds** for every method, and compare the mean CV MSE **and its variability**
- check whether the best hyperparameter lies **at the edge of the grid**
- keep a short record of the grids and the selected values
- make the notebook understandable to someone who did not write it
- "Outsource coding, not understanding and judgement." You must be able to explain every choice.

## 2. Every sub-question, what it asks, and where it is easy to lose points

Legend: **Show** = output that must be visible in the executed notebook · **Write** = text answer · **Watch** = trap.

### Question 1 — Import and prepare the data

**1.a) Import.**
- Show: `Housing.csv` as a pandas DataFrame; X = the 79 features, y = SalePrice.
- Watch: `Id` (first column) is an identifier, not a feature. Keep the original variable names.
- Watch: pandas turns the strings `NA` and `None` into missing values by default. Several of them are real categories. See `2_Data_Findings.md`, section 2.

**1.b) Is SalePrice Gaussian?**
- Show: a **graphical** check (histogram and a normal Q-Q plot), before and after the transformation.
- Write: (1) is it Gaussian? (2) which transformation you suggest and apply; (3) **why it is important** to consider a transformation.
- Watch: keep track of the transformation. Later you need metrics on the transformed scale **and** on the USD scale.

**1.c) Split 70/30.**
- Show: a random 70% / 30% split into (X, y) train and (X, y) test.
- Watch: keep an **unprocessed copy** (`X_train_raw`, `X_test_raw`), because Question 3.b starts again from it. Fix the random seed.

**1.d) Prepare X with training statistics only.**
- Show: numerical missing values → training mean; categorical missing values → training mode; `NA`/`None` kept as a level where the description says so; z-score standardization of the numerical features with the training mean and standard deviation (the same numbers applied to the test set); one-hot encoding (e.g. `pd.get_dummies`); test columns aligned to the training columns (`reindex(columns=..., fill_value=0)`).
- Show (template "checks"): the final shapes; proof that the column names and their order are identical; no test statistic used.
- Watch: data leakage (any statistic computed on the test set or on the full data).

### Question 2 — OLS on the 35 numerical features

**2.a) sklearn OLS.**
- Show: `LinearRegression` with intercept on the training set; **a table with the coefficient of every feature**; in-sample and out-of-sample **MSE and R²**.
- Show: because y was transformed, metrics **on the USD scale** (predictions inverse-transformed first) **and** on the transformed scale.
- Write: comment on the differences between the two scales. What does each set of metrics tell about the model?

**2.b) The same OLS with matrix algebra (numpy, training set only).** A = design matrix with a column of ones, size m × (d + 1).
- (i) Show: β̂ = (AᵀA)⁻¹Aᵀy computed **without forming the inverse** (`np.linalg.solve(A.T @ A, A.T @ y)`), and β̂₀ = the intercept. Compare with 2.a.
- (ii) Show: σ̂² = SSR / (m − (d + 1)) and SE(β̂ⱼ) = √(σ̂² [(AᵀA)⁻¹]ⱼⱼ), including the intercept; say clearly which row is the intercept.
- (iii) Show: the in-sample MSE and R² agree with 2.a up to rounding.
- (iv) Show: rank of A and its singular values. Write: is A full column rank? Do the smallest singular values signal numerical instability? Relate this to (i)–(iii).
- (v) Show: β̂, σ̂² and the SEs with `np.linalg.pinv(A)` (default cutoff). Write: do they change? **When** are the two approaches identical and when can they differ (use (iv))?
- (vi) Show: `statsmodels` OLS coefficient table and SEs; check that they match. Write: if they differ, explain the difference with the rank / stability findings.
- Watch: this part is **designed** around a problem in the data. A is not full rank (two exact identities, see `2_Data_Findings.md`, section 6). `np.linalg.solve` gives no error but its answer is not unique. The marks are for **noticing and explaining** this.

### Question 3 — Regularization

**3.a) OLS on all prepared features.**
- Show: OLS of the (transformed) target on all features of the prepared data from 1.d; in-sample and out-of-sample MSE and R² **on the USD scale**.
- Write: compare with 2.a. Do the categorical features improve out-of-sample prediction? How does the gap between in-sample and out-of-sample performance change?

**3.b) Truncated pseudoinverse (TPI), Ridge, Lasso, Elastic Net.**
- Start from the **raw** splits of 1.c. Use **8-fold CV on the training set**, **MSE** as the criterion, **the same folds** for every method.
- Build the workflow with `ColumnTransformer`, `Pipeline` and `GridSearchCV`.
- (i) One preprocessing component that reproduces 1.d (imputation, standardization, one-hot, `NA`/`None` levels), fitted **inside** each CV training fold. Categories absent from a training fold must not crash the validation fold.
- (ii) One pipeline per method with the **same** preprocessor: Ridge / Lasso / Elastic Net directly after the preprocessor; TPI = preprocessor → `TruncatedSVD` → `LinearRegression`.
- (iii) Reasonable grids: the strength for Ridge and Lasso; strength **and** mixing parameter for Elastic Net; the number of kept singular directions (`svd__n_components`) for TPI.
- (iv) Write: are the selected values at or near the edge of their grid? If yes, extend and search again. **Justify the final ranges briefly.**
- (v) Show: in-sample and out-of-sample MSE and R² on the USD scale of the refitted models; compare with the OLS models of 2.a and 3.a.
- **Summary table** (rows: the 4 regularized models + OLS 2.a + OLS 3.a; "–" where a value does not apply). Columns: selected hyperparameter(s); **mean CV MSE across the 8 folds for the selected values ± its standard deviation**; in-sample MSE and R² (USD); out-of-sample MSE and R² (USD). The CV MSE is on the **transformed** scale.
- Watch: with `scoring="neg_mean_squared_error"`, flip the sign of `mean_test_score` but **not** of `std_test_score`.
- Every model must have an intercept, and the intercept must **not** be penalized. Write: **why is it important not to penalize the intercept?**

**3.c) Explain** the pipeline parameter notation (e.g. `model__alpha`), what `refit=True` does after the best values are chosen, and why this refit is useful before the final test evaluation.

**3.d) Sparsity.**
- Show: the number of non-zero feature coefficients (intercept excluded) for Lasso and Elastic Net; compare with Ridge.
- Show: for TPI, the selected number of singular directions (instead of a count of non-zero coefficients).
- Write: why is truncating singular directions **conceptually different** from coefficient sparsity (Lasso / Elastic Net)?

**3.e) Recommendation.**
- Write: which model do you recommend for predicting house prices? Justify it with the metrics **and** with the nature of the problem (number of features, categorical variables, collinearity, sparsity, nonlinearity). Explain how the strengths and limits of the method fit the structure of the problem.

### Project Report (20%)

Concise. It must **synthesize**, not repeat every cell. Six parts, in this order: **Approach · Model selection · Key results · Interpretation · Robustness · Limitations**. Detailed computations stay in the main body.

## 3. Where the marks most likely are

- **Correct preparation without leakage.** The `NA`/`None` levels, training-only statistics, aligned columns, and a pipeline that really reproduces 1.d.
- **Both scales, correctly.** Metrics on the USD scale after inverse-transforming the predictions, plus the transformed scale where asked. Correct signs from `neg_mean_squared_error`.
- **The rank problem in 2.b.** Rank, singular values, solve vs pinv vs statsmodels, and a clear explanation of why they agree or differ.
- **A clean CV workflow.** The same folds everywhere, sensible grids, an explicit boundary check, and a complete summary table.
- **Interpretation.** Each "why" question answered with the course's ideas (bias-variance, overfitting, regularization, SVD).
- **Report quality.** Short, specific, with the key numbers, honest about limits.

## 4. Final submission checklist

- [ ] Group names in the first cell.
- [ ] Kernel → Restart & Run All, in a fresh kernel, top to bottom, no errors. All outputs visible.
- [ ] No `PREP NOTE` comments left and no unused scaffold text.
- [ ] Every "Write" item above answered in a Markdown cell next to its code.
- [ ] 2.a metrics on **both** scales; every later metric on the USD scale; CV MSE on the log scale.
- [ ] Summary table: 6 rows, all columns, "–" for OLS where needed, CV mean **and** std.
- [ ] Grid-boundary check shown and ranges justified.
- [ ] The test set used only for the final evaluation (say so in the report).
- [ ] Project Report with the six headings, after all questions.
- [ ] Notebook opens and reads well; figures have titles and axis labels.
- [ ] Only the notebook is submitted (plus whatever Moodle asks for). It reads `Housing.csv` from its own folder.
