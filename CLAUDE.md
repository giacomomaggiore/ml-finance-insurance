# CLAUDE.md — handoff for solving MLFI Project 1 with Ugo

**Project:** Machine Learning in Finance and Insurance (ETH), coding Project 1 "Linear Regression and Regularization" (Ames housing, 1460 houses, predict SalePrice). TA: Zhexin Wu. **Last questions to the TA: 7 Oct 2026 · deadline: 14 Oct 2026 (midnight).** Grading: 80% implementation and results, 20% Project Report.

**Status: prepared on 28 Sep 2026, not solved — on purpose.** Ugo wants to solve it together, step by step, and to understand every choice. Your job in the working session: fill in `Project1_Notebook.ipynb` with him, one sub-question at a time, following `Analysis/4_Plan_and_Decisions.md`.

## Read in this order

1. This file.
2. `Analysis/4_Plan_and_Decisions.md`: §1 decisions D1–D12 (confirm them with Ugo first), §2 helpers, §3 the steps with **Check** numbers and **Answer points**, §4 presentation standards, §7 agenda.
3. On demand: `Analysis/1_Requirements.md` (every requirement + the final checklist), `Analysis/2_Data_Findings.md` (data facts), `Analysis/3_Course_Theory.md` (the theory in the course's notation, with the notes sections).
4. The course context pack: `../../../Claude Machine Learning/README.md` and `course-state.md` (Ugo's conventions and the course state).

## How to work with Ugo — per step (1.a, 1.b, …)

1. Explain in 2–3 simple sentences what the step asks and where the marks are.
2. Write the code in the matching notebook cell. Keep the template's guidance comments. Adapt the **tested** reference code in `Analysis/scripts/helpers.py` where it exists (loading, `recode_missing`, 1.d preparation, the preprocessor). The notebook must not import from `Analysis/`.
3. Run it and compare with the plan's **Check** numbers (split `random_state=42`). If a number differs, stop and find the reason before going on.
4. Draft the written answer from the plan's **Answer points**: simple English, short sentences, with the supporting numbers. Ugo reads it and changes it; it is his submission.
5. Delete that step's `PREP NOTE` and answer placeholder, and tick the step in the checklist below.

**Ugo's preferences:** simple English (he is not a native speaker); explain every new notation (he has no strong maths background); no idioms; show the reasoning, not only the code. He decides. Ask before any choice that changes results.

## Rules that protect the grade

- The **test set is used once**, for the final evaluation. Never choose a grid, a model or a transformation by looking at test numbers.
- **One** `KFold(n_splits=8, shuffle=True, random_state=SEED)` object for all four searches. CV MSE on the **log** scale. `neg_mean_squared_error` → flip the sign of the mean, **not** of the std.
- All metrics except 2.a's log-scale ones on the **USD scale**: `np.exp(prediction)` first.
- Preprocessing statistics from training data only. `assert` that the 3.b.i preprocessor reproduces the 1.d matrices exactly (it does: max difference 0.0).
- OLS on the full design: sklearn `LinearRegression`, **never** `np.linalg.pinv` with the default cutoff (it breaks on this Mac, see below).
- Finish with `run_notebook.py --save` (a clean run, outputs saved), then the checklist in `1_Requirements.md` §4.

## Verified facts — do not re-derive (details and scripts in `Analysis/`)

- Read the CSV with `keep_default_na=False` and `na_values` only for the 35 numerical columns. The pandas defaults turn the valid `None` level of MasVnrType (864 houses) and the 14 variables' valid `NA` levels into NaN.
- The types come from `data_description.txt`: 35 numerical, 44 categorical. MSSubClass is categorical (stored as integers).
- Genuinely missing after recoding: LotFrontage 259, GarageYrBlt 81, MasVnrArea 8, MasVnrType 8, Electrical 1, BsmtExposure 1 (Id 949), BsmtFinType2 1 (Id 333).
- Target: skew 1.88 → 0.12 after `np.log`; Box-Cox λ ≈ −0.06 (training data).
- Split 1022/438. Ids 524 and 1299 (cheap "Partial" giants) are in training, and Id 692 is in test. After 1.d: 1022 × 311 and 438 × 311. Five test levels are unseen in training.
- 2.a (log/USD): in-sample MSE 0.02200 / 1.434e9, R² 0.858 / 0.762; out-of-sample 0.02187 / 8.29e8, R² 0.871 / 0.881.
- 2.b: rank(A) = 34 < 36, because TotalBsmtSF and GrLivArea are exact sums of other columns. Singular values 84.9 … 9.92, 1.4e-14, 6.3e-15. `solve` gives no warning and arbitrary values for the 8 area coefficients, with standard errors around 1e5. `pinv` gives the minimum-norm solution. statsmodels = pinv; its SEs are × √(986/988) because it uses df_resid = m − rank.
- 3.a: rank 254 of 312. In-sample log R² 0.950; out-of-sample USD MSE 8.81e8, R² 0.874 (worse than 2.a) → overfitting via rare levels (e.g. Id 811, a pool house).
- CV scans (log MSE): Ridge α ≈ 17.8 → 0.0225 ± 0.0112; Lasso α ≈ 5.6e-4 → 0.0228 ± 0.0127; Elastic Net α ≈ 3.2e-3, l1_ratio 0.25 → 0.0225 ± 0.0115; TPI k = 150 → 0.0228 ± 0.0091. All interior. The fold that holds Id 1299 has MSE ≈ 0.05.

## Environment and commands (run from this folder)

- Kernel: **"Python 3.13 (venv)"** = `~/jupyter-env/bin/python` (numpy with Apple Accelerate). `~/miniconda3` also has all packages. scikit-learn 1.9.1 was installed in both on 28 Sep 2026.
- Full "Restart & Run All" check: `~/miniconda3/bin/python Analysis/scripts/run_notebook.py` (runs a copy, reports errors, warnings, slow cells and leftover `PREP NOTE`s/placeholders). Add `--save` to write the outputs into the notebook.
- Do **not** use `jupyter nbconvert`: `~/.jupyter/jupyter_nbconvert_config.json` points to a missing extension.
- Do **not** regenerate the scaffold: `build_notebook_scaffold.py` would wipe the work, so it refuses without `--force`.
- Analysis: `~/jupyter-env/bin/python Analysis/scripts/data_audit.py` (seconds); `numeric_checks.py` (about 7 minutes).
- PDFs: no poppler on this Mac. Read short PDFs whole; use PyMuPDF (`import pymupdf`, in miniconda) for page text and images.
- **Numerical trap found here:** `np.linalg.pinv(A)` with the default cutoff (1e-15·σ_max) inverts a rounding-level singular value (5.5e-13) of the full one-hot design under Accelerate → garbage predictions. OpenBLAS (miniconda) happens to be fine. Use sklearn, or `pinv(A, rtol=max(A.shape) * np.finfo(float).eps)`.

## Progress checklist (update it during the sessions)

- [ ] Decisions D1–D12 confirmed (note any change here)
- [ ] Group names · [ ] TA questions posted or skipped (deadline 7 Oct)
- [ ] Helper functions
- [ ] 1.a · [ ] 1.b (+ answer) · [ ] 1.c · [ ] 1.d
- [ ] 2.a (+ answer) · [ ] 2.b.i · [ ] 2.b.ii · [ ] 2.b.iii · [ ] 2.b.iv (+ answer) · [ ] 2.b.v (+ answer) · [ ] 2.b.vi (+ answer)
- [ ] 3.a (+ answer) · [ ] 3.b.i · [ ] 3.b.ii · [ ] 3.b.iii · [ ] 3.b.iv (+ answer) · [ ] 3.b.v + table (+ answer) · [ ] intercept (+ answer)
- [ ] 3.c · [ ] 3.d · [ ] 3.e
- [ ] Robustness cells (R.1, R.2, R.4, R.5; R.3 optional)
- [ ] Project Report (6 headings)
- [ ] `run_notebook.py --save` clean · [ ] checklist in `1_Requirements.md` §4 · [ ] submitted on Moodle

## At the end of every session

Update the checklist above and the "Projects" section of `../../../Claude Machine Learning/course-state.md`. Then tell Ugo in one line what changed.
