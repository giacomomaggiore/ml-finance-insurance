"""Numerical checks for Project 1 — backs the statements in Analysis/4_Plan_and_Decisions.md.

These are PROTOTYPE CHECKS on the split with random_state=42, used to choose the
approach and the grid ranges. They are not the submission.

Run:     python "Analysis/scripts/numeric_checks.py"          (about 7 minutes on 4 CPU cores; the Elastic Net scan is the slow part)
Writes:  Analysis/scripts/outputs/numeric_checks.txt
         Analysis/scripts/outputs/cv_scan_<method>.csv
         Analysis/figures/cv_curves.png
"""
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error as mse, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from helpers import (FIG_DIR, N_FOLDS, NUM, OUT_DIR, SEED, Tee, make_preprocessor, prepare_manual,
                     recode_missing, split)

say = Tee(OUT_DIR / "numeric_checks.txt")
pd.set_option("display.width", 170)
X_tr_raw, X_te_raw, y_tr, y_te = split()
l_tr, l_te = np.log(y_tr), np.log(y_te)
P_tr, P_te, _ = prepare_manual(X_tr_raw, X_te_raw)


def metrics(model, X_a, X_b):
    """MSE and R2 in/out of sample, on the log scale and on the USD scale (exp of the prediction)."""
    out = {}
    for tag, Xs, ls, ys in [("in", X_a, l_tr, y_tr), ("out", X_b, l_te, y_te)]:
        p = model.predict(Xs)
        out[f"{tag} log MSE"], out[f"{tag} log R2"] = mse(ls, p), r2_score(ls, p)
        out[f"{tag} USD MSE"], out[f"{tag} USD R2"] = mse(ys, np.exp(p)), r2_score(ys, np.exp(p))
    return out


def course_r2_os(y_true, y_pred, train_mean):
    """Out-of-sample R2 as defined in the lecture (MLFI 1.6): SST uses the TRAINING mean."""
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - train_mean) ** 2)


say("=" * 78, "\nA. QUESTION 2.a — OLS ON THE 35 NUMERICAL FEATURES (sklearn)\n" + "=" * 78)
ols_num = LinearRegression().fit(P_tr[NUM], l_tr)
res = metrics(ols_num, P_tr[NUM], P_te[NUM])
say(pd.Series(res).map(lambda v: f"{v:.5g}").to_string())
say(f"sklearn LinearRegression.rank_ = {ols_num.rank_} of 35 centred columns (2 exact dependencies found by lstsq)")
p_te = ols_num.predict(P_te[NUM])
say(f"out-of-sample R2, USD: sklearn (test mean) {r2_score(y_te, np.exp(p_te)):.5f} | course definition "
    f"(training mean) {course_r2_os(y_te, np.exp(p_te), y_tr.mean()):.5f}")
say(f"out-of-sample R2, log: sklearn {r2_score(l_te, p_te):.5f} | course {course_r2_os(l_te, p_te, l_tr.mean()):.5f}")
smear = np.mean(np.exp(l_tr - ols_num.predict(P_tr[NUM])))
say(f"Duan smearing factor mean(exp(residual)) = {smear:.4f}; out USD MSE plain exp {mse(y_te, np.exp(p_te)):.4e} "
    f"vs smeared {mse(y_te, smear * np.exp(p_te)):.4e}")
worst = (np.exp(ols_num.predict(P_tr[NUM])) - y_tr).abs().sort_values(ascending=False).head(3)
say(f"largest in-sample USD errors (Id: |error|): {worst.round(0).to_dict()} -> the 'Partial' sale outliers are in the TRAINING set")

say("\n" + "=" * 78, "\nB. QUESTION 2.b — MATRIX ALGEBRA ON THE TRAINING SET\n" + "=" * 78)
A = np.column_stack([np.ones(len(P_tr)), P_tr[NUM].to_numpy()])
yv = l_tr.to_numpy()
names = ["Intercept"] + NUM
m, p = A.shape
s = np.linalg.svd(A, compute_uv=False)
say(f"A is {m} x {p}; rank(A) = {np.linalg.matrix_rank(A)}  -> NOT full column rank")
say(f"singular values: largest {s[0]:.4g}; 5 smallest {np.array2string(s[-5:], precision=3)}")
say(f"cond(A) = {s[0] / s[-1]:.3g}; cond(A'A) = cond(A)^2 = {(s[0] / s[-1]) ** 2:.3g} >> 1/eps = {1 / np.finfo(float).eps:.2g}")
say(f"cond(A) ignoring the two zero directions = {s[0] / s[-3]:.3g}  -> the rest of A is well conditioned")
say(f"matrix_rank tolerance = {s[0] * max(A.shape) * np.finfo(float).eps:.2g}; np.linalg.pinv default cutoff = 1e-15 * s_max = {1e-15 * s[0]:.2g}")

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    b_solve = np.linalg.solve(A.T @ A, A.T @ yv)
say(f"np.linalg.solve(A'A, A'y): no error, {len(caught)} warnings -> it silently returns ONE of infinitely many solutions")
b_pinv = np.linalg.pinv(A) @ yv
res_s, res_p = yv - A @ b_solve, yv - A @ b_pinv
s2_s, s2_p = res_s @ res_s / (m - p), res_p @ res_p / (m - p)
se_s = np.sqrt(s2_s * np.diag(np.linalg.solve(A.T @ A, np.eye(p))))
se_p = np.sqrt(s2_p * np.diag(np.linalg.pinv(A.T @ A)))
sm_fit = sm.OLS(yv, A).fit()
table = pd.DataFrame({"b solve": b_solve, "b pinv": b_pinv, "b statsmodels": sm_fit.params,
                      "SE solve": se_s, "SE pinv": se_p, "SE statsmodels": sm_fit.bse}, index=names)
collinear = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea"]
say(table.loc[["Intercept", "OverallQual", "YearBuilt"] + collinear].map(lambda v: f"{v:.4g}").to_string())
others = [n for n in names if n not in collinear]
say(f"max |b solve - b pinv| on the {len(others)} other coefficients: {np.abs(b_solve - b_pinv)[[names.index(n) for n in others]].max():.2e}")
say(f"max |SE solve - SE pinv| on the other coefficients: {np.abs(se_s - se_p)[[names.index(n) for n in others]].max():.2e}")
say(f"in-sample MSE solve {np.mean(res_s ** 2):.12f} | pinv {np.mean(res_p ** 2):.12f} | sklearn {mse(l_tr, ols_num.predict(P_tr[NUM])):.12f}")
say(f"sigma^2 with m-(d+1) = {m - p}: {s2_p:.6f} | statsmodels scale with m-rank = {sm_fit.df_resid:.0f}: {sm_fit.scale:.6f}")
say(f"max |b statsmodels - b pinv| = {np.abs(sm_fit.params - b_pinv).max():.1e}; SE ratio statsmodels/pinv = "
    f"{np.median(sm_fit.bse / se_p):.6f} = sqrt({m - p}/{m - 34}) = {np.sqrt((m - p) / (m - 34)):.6f}")
say("statsmodels summary note: " + [ln.strip() for ln in str(sm_fit.summary()).splitlines() if "eigenvalue" in ln][0])
_, _, Vt = np.linalg.svd(A, full_matrices=False)
null = pd.DataFrame(Vt[-2:].T, index=names, columns=["v_35", "v_36"])
say("right singular vectors of the two ~0 singular values (entries with |value| > 1e-8):")
say(null[(null.abs() > 1e-8).any(axis=1)].round(4).to_string())
keep = [i for i, n in enumerate(names) if n not in ("TotalBsmtSF", "GrLivArea")]
A2 = A[:, keep]
s2v = np.linalg.svd(A2, compute_uv=False)
b2 = np.linalg.solve(A2.T @ A2, A2.T @ yv)
say(f"remedy check — drop TotalBsmtSF and GrLivArea: rank {np.linalg.matrix_rank(A2)} of {A2.shape[1]}, cond {s2v[0] / s2v[-1]:.3g}, "
    f"solve vs pinv max diff {np.abs(b2 - np.linalg.pinv(A2) @ yv).max():.1e}, same fitted values: "
    f"{np.allclose(A2 @ b2, A @ b_pinv)}")

say("\n" + "=" * 78, "\nC. QUESTION 1.d BY HAND == PIPELINE PREPROCESSOR (Question 3.b.i)?\n" + "=" * 78)
pre = make_preprocessor()
Q_tr, Q_te = pre.fit_transform(X_tr_raw), pre.transform(X_te_raw)
say(f"shapes {Q_tr.shape} / {Q_te.shape}; same column names and order: {list(pre.get_feature_names_out()) == list(P_tr.columns)}")
say(f"max |manual - pipeline|: train {np.abs(P_tr.to_numpy() - Q_tr).max():.1e}, test {np.abs(P_te.to_numpy() - Q_te).max():.1e}")

say("\n" + "=" * 78, "\nD. QUESTION 3.a — OLS ON ALL PREPARED FEATURES\n" + "=" * 78)
ols_all = LinearRegression().fit(P_tr, l_tr)
say(pd.Series(metrics(ols_all, P_tr, P_te)).map(lambda v: f"{v:.5g}").to_string())
A_full = np.column_stack([np.ones(len(P_tr)), P_tr.to_numpy()])
s_full = np.linalg.svd(A_full, compute_uv=False)
rank_tol = s_full[0] * max(A_full.shape) * np.finfo(float).eps          # the tolerance of np.linalg.matrix_rank
zero = s_full[s_full <= rank_tol]
say(f"design with intercept: {A_full.shape}, rank {int((s_full > rank_tol).sum())} -> {len(zero)} exact dependencies "
    f"(one-hot blocks sum to the intercept column, identical 'no garage'/'no basement' dummies, the 2 area identities)")
say(f"sklearn LinearRegression.rank_ = {ols_all.rank_} of {P_tr.shape[1]} centred columns (it finds the same dependencies)")
say(f"the {len(zero)} 'zero' singular values lie in [{zero.min():.1e}, {zero.max():.1e}]; the smallest real one is "
    f"{s_full[s_full > rank_tol].min():.3g} -> a clear gap; matrix_rank tolerance {rank_tol:.1e}")
default_cut = 1e-15 * s_full[0]
say(f"np.linalg.pinv DEFAULT cutoff 1e-15 * s_max = {default_cut:.1e}: {int((zero > default_cut).sum())} 'zero' singular value(s) lie above it "
    f"with this numpy/LAPACK build -> the default pinv would invert rounding noise (machine-dependent!). "
    f"Use sklearn, or pinv(A, rtol=max(m, n) * eps).")
coef = pd.Series(ols_all.coef_, index=P_tr.columns)
top = coef.abs().sort_values(ascending=False).head(8).index
say("largest |coefficients| and how many TRAINING houses have that level:")
say(pd.DataFrame({"coef": coef[top].round(3), "train count": [str(int(P_tr[c].sum())) if c not in NUM else "numerical" for c in top]}).to_string())
b_min = np.linalg.pinv(A_full, rtol=max(A_full.shape) * np.finfo(float).eps) @ yv
p_alt = np.column_stack([np.ones(len(P_te)), P_te.to_numpy()]) @ b_min
diff = pd.Series(np.abs(p_alt - ols_all.predict(P_te)), index=P_te.index)
say(f"in-sample fitted values of the two least-squares solutions agree: max diff "
    f"{np.abs(A_full @ b_min - ols_all.predict(P_tr)).max():.1e}")
say(f"two least-squares solutions (sklearn vs rank-aware pinv with explicit intercept) predict the TEST set differently on "
    f"{int((diff > 1e-6).sum())} houses (feature vectors outside the row space of the training design), by up to "
    f"{diff.max():.3f} log units: {diff[diff > 1e-6].round(3).to_dict()}")
e = pd.Series(np.exp(ols_all.predict(P_te)) - y_te, index=P_te.index)
say(f"worst test errors (USD): {e.abs().sort_values(ascending=False).head(3).round(0).to_dict()} "
    f"(Id 811 has a 648 sq ft pool; only {int((recode_missing(X_tr_raw).PoolArea > 0).sum())} training houses have a pool)")

say("\n" + "=" * 78, "\nE. QUESTION 3.b — 8-FOLD CV SCANS (to choose grid RANGES)\n" + "=" * 78)
cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for k, (a, b) in enumerate(cv.split(X_tr_raw)):
    F = make_preprocessor().fit_transform(X_tr_raw.iloc[a])
    flag = [i for i in (524, 1299) if i in X_tr_raw.index[b]]
    say(f"fold {k}: {len(a)} fitting rows, {F.shape[1]} features after one-hot, rank {np.linalg.matrix_rank(F)}"
        + (f" | validation part holds outlier(s) {flag}" if flag else ""))

scans = {
    "Ridge": (Ridge(), {"model__alpha": np.logspace(-3, 4, 29)}, False),
    "Lasso": (Lasso(max_iter=100_000), {"model__alpha": np.logspace(-6, -1, 21)}, False),
    "ElasticNet": (ElasticNet(max_iter=100_000), {"model__alpha": np.logspace(-5, 0, 21),
                                                  "model__l1_ratio": [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]}, False),
    "TruncatedPI": (LinearRegression(), {"svd__n_components": list(range(10, 246, 5))}, True),
}
results = {}
for name, (model, grid, use_svd) in scans.items():
    steps = [("preprocess", make_preprocessor())]
    if use_svd:
        steps.append(("svd", TruncatedSVD(algorithm="arpack", random_state=SEED)))
    steps.append(("model", model))
    search = GridSearchCV(Pipeline(steps), grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=4, refit=True)
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        search.fit(X_tr_raw, l_tr)
    cr = pd.DataFrame(search.cv_results_)
    cr["cv_mse"], cr["cv_std"] = -cr["mean_test_score"], cr["std_test_score"]
    params = [c for c in cr.columns if c.startswith("param_")]
    cr[params + ["cv_mse", "cv_std"] + [f"split{k}_test_score" for k in range(N_FOLDS)]].to_csv(OUT_DIR / f"cv_scan_{name}.csv", index=False)
    best = search.best_index_
    fold_mse = [-cr.loc[best, f"split{k}_test_score"] for k in range(N_FOLDS)]
    say(f"{name:11s} best {search.best_params_} | CV MSE {cr.cv_mse[best]:.5f} ± {cr.cv_std[best]:.5f} | "
        f"{time.time() - t0:.0f} s | convergence warnings {sum(issubclass(w.category, ConvergenceWarning) for w in caught)}")
    say(f"            fold MSEs at the best value: {np.round(fold_mse, 4).tolist()}")
    say(f"            refitted model: test log MSE {mse(l_te, search.predict(X_te_raw)):.5f}  (for orientation only — never used to choose)")
    results[name] = (cr, params, search.best_params_)

fig, axes = plt.subplots(1, 4, figsize=(17, 4), sharey=True)
for ax, (name, (cr, params, best)) in zip(axes, results.items()):
    if name == "ElasticNet":
        for l1, g in cr.groupby("param_model__l1_ratio"):
            ax.plot(g["param_model__alpha"].astype(float), g["cv_mse"], marker=".", lw=1, label=f"l1_ratio={l1}")
        ax.legend(fontsize=7)
    else:
        xcol = params[0]
        ax.plot(cr[xcol].astype(float), cr["cv_mse"], marker=".", color="#1b4a7a")
    if name != "TruncatedPI":
        ax.set_xscale("log")
    ax.set_title(f"{name}: best {', '.join(f'{k.split('__')[1]}={v:.3g}' for k, v in best.items())}", fontsize=10)
    ax.set_xlabel("n_components" if name == "TruncatedPI" else "alpha")
    ax.set_ylim(0.02, 0.045)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("mean 8-fold CV MSE (log scale target)")
fig.tight_layout()
fig.savefig(FIG_DIR / "cv_curves.png", dpi=110)
say(f"\nfigure written: {FIG_DIR / 'cv_curves.png'}")
