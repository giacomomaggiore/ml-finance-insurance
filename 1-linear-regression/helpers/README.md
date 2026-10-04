# Project 1 — Linear Regression and Regularization

**Status (28 September 2026): prepared, not solved.** The analysis is done and the notebook is set up from the template. The questions themselves are answered in the working session next week.

**Deadline: 14 October 2026** (midnight) · **last day for questions to the TA: 7 October** · TA in charge: Zhexin Wu (Moodle Q&A forum).

## What is in this folder

| File | What it is |
| --- | --- |
| `../status.md` | Current project status, next steps, and validation commands. |
| `1-main.ipynb` | **The notebook to submit**, built from the official template. One cell per sub-question, the TA's guidance kept word for word, a setup cell, an empty helper-functions section, answer placeholders where text is required, and `PREP NOTE` lines with the key traps (delete them before submitting). It already runs top to bottom. |
| `data/Housing.csv`, `data/data_description.txt` | Course input data and its description. These files are source material and should not be modified. |
| `requirements.txt` | Package versions used. |
| `helpers/Analysis/1_Requirements.md` | Every requirement of the assignment, the grading, where points are easy to lose, and the **submission checklist**. |
| `helpers/Analysis/2_Data_Findings.md` | Verified facts about the data and its traps (the `NA`/`None` problem, missing values, rare levels, the two exact identities, the target, outliers, the split). |
| `helpers/Analysis/3_Course_Theory.md` | The theory from the lecture notes that each question needs, with formulas in the course's notation. |
| `helpers/Analysis/4_Plan_and_Decisions.md` | **The work plan.** Decisions to confirm, helper functions, step-by-step instructions with sanity-check numbers, answer points, pitfalls, optional TA questions, and the agenda for next week. |
| `helpers/Analysis/figures/` | `target_distribution.png` (for 1.b) and `cv_curves.png` (the CV scans behind the grid ranges). |
| `helpers/Analysis/scripts/` | `data_audit.py` and `numeric_checks.py` (with `helpers.py`) reproduce every number in the documents; their results are in `outputs/`. `run_notebook.py` = "Restart & Run All" + a pre-submission check. `build_notebook_scaffold.py` = how the notebook was built (it refuses to overwrite it). These are tools, not the submission. |

## Start here next week

Paste this into a new chat (Claude Code, in the Machine Learning folder):

> Let's solve MLFI Project 1 together, step by step. First read `status.md` and `helpers/Analysis/4_Plan_and_Decisions.md`, then we do Question 1.a.

If you work alone:

1. Read `helpers/Analysis/4_Plan_and_Decisions.md`: section 1 (decisions D1–D12) and section 7 (agenda).
2. Open `1-main.ipynb` with the kernel **"Python 3.13 (venv)"** (`~/jupyter-env`).
3. Work through the notebook from top to bottom. Each cell's `PREP NOTE` points to its step in the plan.
4. Check the whole notebook with `~/miniconda3/bin/python helpers/Analysis/scripts/run_notebook.py` (add `--save` for the final run).
5. Finish with the checklist in `helpers/Analysis/1_Requirements.md`, section 4.

## The analysis in five points

1. **`NA` is not always missing.** For 14 variables it is a real category (no basement, no garage, …), and in MasVnrType `None` is a real category. pandas' default reader turns both into NaN, and imputation would then give 872 houses a brick veneer. Read the file with `keep_default_na=False`. Two basement `NA`s (Ids 333, 949) are genuinely missing, because those houses do have a basement.
2. **Take the variable types from `data_description.txt`** (35 numerical, 44 categorical). MSSubClass is stored as numbers but is a category.
3. **Question 2.b is built around a rank problem.** TotalBsmtSF and GrLivArea are exact sums of other columns, so A has rank 34 < 36. `np.linalg.solve` silently returns one of infinitely many solutions. `pinv` returns the minimum-norm one. statsmodels matches `pinv` but divides by m − rank = 988 instead of 986, so its standard errors are √(986/988) smaller.
4. **Plain OLS with all features overfits.** In-sample R² goes up to 0.95, but out of sample it does no better than the numerical-only model. The cause is rare one-hot levels fitted to 1–3 houses. The full design has rank 254 of 312. On this Mac, `np.linalg.pinv` with its default cutoff breaks on it; use sklearn.
5. **With 8-fold CV, all four regularized models end up close** (log-scale CV MSE 0.0225–0.0228; Ridge α ≈ 18, Lasso α ≈ 5.6e-4, Elastic Net α ≈ 3e-3 with l1_ratio 0.25, TPI k = 150). The fold standard deviation is large because one fold contains a very unusual house (Id 1299). All optima are inside the scanned ranges.

## Environment

- `scikit-learn` was missing in both Python environments. Version 1.9.1 is now installed in `~/jupyter-env` (the Jupyter kernel) and in `miniconda3`.
- Known issue, not needed for the submission: `jupyter nbconvert` from miniconda fails. `~/.jupyter/jupyter_nbconvert_config.json` refers to `jupyter_contrib_nbextensions`, which is not installed. Removing that entry fixes it.
- Re-run the analysis (from this folder):

```bash
~/jupyter-env/bin/python helpers/Analysis/scripts/data_audit.py
```

```bash
~/jupyter-env/bin/python helpers/Analysis/scripts/numeric_checks.py
```

The second one takes about 7 minutes.
