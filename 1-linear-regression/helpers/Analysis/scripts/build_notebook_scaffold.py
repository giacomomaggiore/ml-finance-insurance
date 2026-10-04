"""Build the Project 1 notebook scaffold from the official template (run once on 28 Sep 2026).

Kept for reference only. It refuses to overwrite an existing notebook unless called with --force.
Run with miniconda's python (it needs nbformat):  ~/miniconda3/bin/python build_notebook_scaffold.py

The TA's guidance comments are copied VERBATIM from 'Project1 - Template.ipynb' and split
into one code cell per sub-question. Added: a setup cell, short headings, answer
placeholders where the assignment asks for text, and 'PREP NOTE' comments (to be deleted
before submission) that point to helpers/Analysis/4_Plan_and_Decisions.md.
"""
import re
import sys
from pathlib import Path

import nbformat as nbf

PROJ = Path(__file__).resolve().parents[3]
TEMPLATE = PROJ / "data" / "solution.ipynb"
OUT = PROJ / "1-main.ipynb"
if OUT.exists() and "--force" not in sys.argv:
    sys.exit(f"{OUT.name} already exists and may contain your work. Re-run with --force to overwrite it.")

tpl = nbf.read(TEMPLATE, as_version=4)
src = ["".join(c.source) if isinstance(c.source, list) else c.source for c in tpl.cells]
assert src[2].startswith("### Question 1") and src[4].startswith("### Question 2")
assert src[6].startswith("### Minimal") and src[8].startswith("### Question 3") and src[10].startswith("### Project Report")


def chunks(text, starts):
    """Split a template code cell at the lines that begin with one of `starts` (in order)."""
    lines = text.splitlines()
    idx = []
    for s in starts:
        hit = next(i for i, ln in enumerate(lines) if ln.startswith(s) and (not idx or i > idx[-1]))
        idx.append(hit)
    idx.append(len(lines))
    return ["\n".join(lines[a:b]).strip("\n") for a, b in zip(idx[:-1], idx[1:])]


q1 = dict(zip("abcd", chunks(src[3], ["# 1.a)", "# 1.b)", "# 1.c)", "# 1.d)"])))
q2 = dict(zip(["a", "i", "ii", "iii", "iv", "v", "vi"],
              chunks(src[5], ["# 2.a)", "# 2.b)", "# (ii)", "# (iii)", "# (iv)", "# (v)", "# (vi)"])))
q3_keys = ["a", "b", "bi", "bii", "biii", "biv", "bv", "table", "intercept", "c", "d", "e"]
q3 = dict(zip(q3_keys, chunks(src[9], ["# 3.a)", "# 3.b) Implement", "# 3.b.i)", "# 3.b.ii)", "# 3.b.iii)",
                                       "# 3.b.iv)", "# 3.b.v)", "# 3.b) REQUIRED SUMMARY TABLE",
                                       "# Why is it important", "# 3.c)", "# 3.d)", "# 3.e)"])))

PLAN = "helpers/Analysis/4_Plan_and_Decisions.md"


def note(step, *lines):
    body = "\n".join(f"#   {ln}" for ln in lines)
    return f"\n\n# PREP NOTE (delete before submission) - {PLAN}, step {step}:\n{body}"


def answer(label):
    return nbf.v4.new_markdown_cell(
        f"**Answer {label}**\n\n*To be written. The key points and the numbers to check are in "
        f"`{PLAN}`, step {label}.*")


cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# --- title and group (template cells 0 and 1, verbatim) ---------------------------------
md(src[0])
md(src[1].replace("**Please write the names of all group members here:**\n",
                  "**Please write the names of all group members here:**\n\n- *(name 1)*\n- *(name 2)*\n", 1))

# --- setup --------------------------------------------------------------------------------
md("### Setup\n\nAll packages and all fixed constants of the analysis, in one place.")
code('''import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import sklearn
import statsmodels
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 42          # fixed once, before looking at any result: used for the split, the CV folds and TruncatedSVD
TEST_SIZE = 0.30   # 70% training / 30% test (Question 1.c)
N_FOLDS = 8        # 8-fold cross-validation with the same folds for every method (Question 3.b)
DATA_FILE = "data/Housing.csv"
DESCRIPTION_FILE = "data/data_description.txt"

pd.set_option("display.max_columns", 60)
print(f"pandas {pd.__version__} | numpy {np.__version__} | scipy {scipy.__version__} | "
      f"scikit-learn {sklearn.__version__} | statsmodels {statsmodels.__version__}")''')

# --- helper functions (empty on purpose: written together in the working session) ---------
md("### Helper functions\n\nSmall reusable functions used in several questions "
   "(the assignment asks for them under *Code quality*).")
code("# Helper functions, each with a one-line docstring (plan: section 2 of " + PLAN + "):\n"
     "#   parse_description(path)      -> 35 numerical / 44 categorical names + listed levels\n"
     "#   load_housing(path)           -> X, y read safely ('NA'/'None' kept as text, Id as index)\n"
     "#   recode_missing(X)            -> 'NA' that is not a valid level -> NaN (stateless)\n"
     "#   prepare_features(Xtr, Xte)   -> Question 1.d by hand (training statistics only)\n"
     "#   regression_metrics(model, Xtr, Xte) -> MSE and R2, in/out of sample, log and USD scale\n"
     "#   ols_matrix(A, y, method)     -> beta, sigma^2, SE with 'solve' or 'pinv' (Question 2.b)\n"
     "#   make_preprocessor()          -> the Question 3.b.i component\n"
     "#   tune(estimator, grid, svd)   -> GridSearchCV with the shared 8 folds\n"
     "#   grid_edges(search, grid)     -> selected values on the edge of the grid?\n"
     "#   cv_summary(search)           -> best params, mean CV MSE (sign flipped) and its std"
     + note("2", "Tested reference versions of parse_description, load_raw (= load_housing), recode_missing,",
            "prepare_manual (= prepare_features) and make_preprocessor are in helpers/Analysis/scripts/helpers.py.",
            "Adapt them here; the submitted notebook must not import from the Analysis folder."))

# --- Question 1 ---------------------------------------------------------------------------
md(src[2])
md("#### 1.a) Import the data and separate X and y")
code(q1["a"] + note("1.a",
     "pandas' default read_csv turns the strings 'NA' AND 'None' into NaN: the 864 valid 'None' levels of",
     "MasVnrType and every valid 'NA' level would be lost. Read with keep_default_na=False and set",
     "na_values only for the 35 numerical columns.",
     "Take the 35 numerical / 44 categorical lists from data_description.txt, not from the dtypes:",
     "MSSubClass is stored as integers but is categorical (dtype selection gives 36 'numerical' columns)."))
md("#### 1.b) Is SalePrice approximately Gaussian?")
code(q1["b"] + note("1.b",
     "Raw SalePrice: skew 1.88, convex Q-Q plot. log(SalePrice): skew 0.12, almost straight Q-Q line.",
     "Box-Cox lambda is about -0.06 on the training data, so the plain log is the natural choice (no parameter to fit).",
     "Keep the USD target too: every final metric must be reported in USD after np.exp(prediction)."))
cells.append(answer("1.b"))
md("#### 1.c) Train/test split (70% / 30%)")
code(q1["c"] + note("1.c",
     "train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED) -> 1022 training and 438 test houses.",
     "With SEED = 42 the two 'Partial' sale outliers (Ids 524, 1299) fall in the TRAINING set, Id 692 in the test set."))
md("#### 1.d) Missing values, standardization, one-hot encoding")
code(q1["d"] + note("1.d",
     "'NA' is a real level for 14 variables; for the 5 basement variables only if TotalBsmtSF == 0",
     "(Ids 949 and 333 have a basement, so their 'NA' is genuinely missing). All other 'NA' are missing.",
     "Impute first, then standardize with ddof=0 (the same as StandardScaler, so Question 3 can reproduce it exactly).",
     "Expected with SEED = 42: train (1022, 311), test (438, 311); 5 test levels do not occur in training."))

# --- Question 2 ---------------------------------------------------------------------------
md(src[4])
md("#### 2.a) OLS on the 35 numerical features (sklearn)")
code(q2["a"] + note("2.a",
     "Write ONE metrics helper (MSE and R2, in/out of sample, log scale and USD scale) and reuse it for every model.",
     "The coefficients of the 8 area variables in the two exact identities are not unique (see 2.b.iv).",
     "With SEED = 42 the in-sample USD MSE is LARGER than the out-of-sample one: explain it (outliers 524, 1299)."))
cells.append(answer("2.a"))
md("#### 2.b) OLS with matrix algebra (training set only)\n\n##### (i) Coefficients from the normal equation")
code(q2["i"] + note("2.b.i",
     "np.linalg.solve raises no error here, but A has rank 34 < 36, so A'A is singular: it returns ONE of",
     "infinitely many solutions. Check np.linalg.matrix_rank(A) and the condition number and say so."))
md("##### (ii) Standard errors")
code(q2["ii"] + note("2.b.ii",
     "sigma^2 = SSR / (m - (d+1)) = SSR / 986. The SEs of the 8 area variables come out around 1e5: meaningless."))
md("##### (iii) Check against Question 2.a")
code(q2["iii"])
md("##### (iv) Rank and singular values of A")
code(q2["iv"] + note("2.b.iv",
     "rank 34 of 36; two singular values about 1e-14 (largest about 85); the next one is about 10, so the rest",
     "of A is well conditioned. The right singular vectors of the two zero singular values show WHICH columns",
     "are dependent: TotalBsmtSF = BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF and GrLivArea = 1stFlrSF + 2ndFlrSF + LowQualFinSF."))
cells.append(answer("2.b.iv"))
md("##### (v) Moore-Penrose pseudoinverse")
code(q2["v"] + note("2.b.v",
     "pinv cuts singular values below 1e-15 * s_max, so it returns the minimum-norm solution.",
     "Same fitted values, same sigma^2, same SEs for the 28 other coefficients; different, finite values for the 8 area ones."))
cells.append(answer("2.b.v"))
md("##### (vi) statsmodels")
code(q2["vi"] + note("2.b.vi",
     "statsmodels uses pinv (same coefficients as (v)) but df_resid = m - rank(A) = 988 instead of 986:",
     "every SE is smaller by the factor sqrt(986/988) = 0.99899. Its summary also warns about a tiny eigenvalue."))
cells.append(answer("2.b.vi"))

# --- minimal example (template cells 6 and 7, verbatim) ----------------------------------------
md(src[6])
code(src[7])

# --- Question 3 ---------------------------------------------------------------------------
md(src[8])
md("#### 3.a) OLS on all prepared features")
code(q3["a"] + note("3.a",
     "The full design (1022 x 312 with intercept) has rank 254: 58 exact dependencies (one-hot blocks, identical",
     "'no garage'/'no basement' dummies, the 2 area identities). Expect a much better in-sample fit but NO",
     "out-of-sample gain over 2.a: rare levels seen in 1-3 training houses get huge coefficients (overfitting)."))
cells.append(answer("3.a"))
md("#### 3.b) Truncated pseudoinverse, Ridge, Lasso and Elastic Net with 8-fold CV")
code(q3["b"])
md("##### 3.b.i) One reusable preprocessing component")
code(q3["bi"] + note("3.b.i",
     "FunctionTransformer(recode_missing, feature_names_out='one-to-one') -> ColumnTransformer(num: mean imputer +",
     "StandardScaler; cat: most_frequent imputer + OneHotEncoder(handle_unknown='ignore', sparse_output=False)).",
     "Check that it reproduces the 1.d matrices exactly (it does: max difference 0.0 in the preparation check)."))
md("##### 3.b.ii) One pipeline per method")
code(q3["bii"] + note("3.b.ii",
     "TruncatedSVD(algorithm='arpack', random_state=SEED): exact and reproducible (the default 'randomized' is approximate).",
     "The folds have 305-310 features after one-hot and rank 246-251, so n_components must stay below about 245."))
md("##### 3.b.iii) Hyperparameter grids and tuning")
code(q3["biii"] + note("3.b.iii",
     "Grid ranges that worked in the scans: Ridge alpha 1e-3..1e4; Lasso alpha 1e-6..1e-1;",
     "Elastic Net alpha 1e-5..1 x l1_ratio 0.01..1; TPI n_components 10..245. Create the KFold object ONCE."))
md("##### 3.b.iv) Are the selected values at the edge of a grid?")
code(q3["biv"] + note("3.b.iv",
     "Write a small helper that reports, for each parameter, whether the best value is the first or last grid",
     "point, and plot the mean CV MSE against each hyperparameter (a U-shape shows the minimum is inside)."))
cells.append(answer("3.b.iv"))
md("##### 3.b.v) Final evaluation of the refitted models")
code(q3["bv"])
md("##### Summary table")
code(q3["table"])
cells.append(answer("3.b.v"))
md("##### Why is the intercept not penalized?")
code(q3["intercept"])
cells.append(answer("3.b (intercept)"))
md("#### 3.c) Pipeline parameter names and `refit=True`")
code(q3["c"])
cells.append(answer("3.c"))
md("#### 3.d) Sparsity versus truncation of singular directions")
code(q3["d"])
cells.append(answer("3.d"))
md("#### 3.e) Final model recommendation")
code(q3["e"])
cells.append(answer("3.e"))

# --- optional robustness section ------------------------------------------------------------
md("### Additional robustness checks\n\nOptional. These give the evidence for the *Robustness* part of the report.")
code("# R.1) The pipeline preprocessor reproduces Question 1.d exactly (no hidden differences).\n"
     "# R.2) Mean CV MSE against each hyperparameter (grid adequacy).\n"
     "# R.3) Repeat split + tuning for several random seeds: is the ranking of the models stable?\n"
     "# R.4) Influence of the unusual houses (Ids 524, 1299, 692): largest errors, fold-wise CV MSE."
     + note("R", "Keep this section short: a few lines of code and one sentence per check."))

# --- report (template cell 10, verbatim) ---------------------------------------------------------
md(src[10])

nb = nbf.v4.new_notebook(cells=cells, metadata=tpl.metadata)
nbf.validate(nb)
OUT.write_text(nbf.writes(nb), encoding="utf-8")
print(f"written {OUT.name}: {len(cells)} cells "
      f"({sum(c.cell_type == 'code' for c in cells)} code, {sum(c.cell_type == 'markdown' for c in cells)} markdown)")
