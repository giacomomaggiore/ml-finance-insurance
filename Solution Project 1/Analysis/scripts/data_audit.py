"""Data audit for Project 1 — reproduces every number in Analysis/2_Data_Findings.md.

Run from anywhere:  python "Analysis/scripts/data_audit.py"
Writes:  Analysis/scripts/outputs/data_audit.txt
         Analysis/figures/target_distribution.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from helpers import (BASEMENT, CAT, DATA_FILE, FIG_DIR, LEVELS, NA_LEVEL, NUM, OUT_DIR, SEED, TYPES,
                     Tee, load_raw, prepare_manual, recode_missing, split)

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
say = Tee(OUT_DIR / "data_audit.txt")
pd.set_option("display.width", 160)

say("=" * 78, "\n1. SHAPE AND VARIABLE TYPES\n" + "=" * 78)
raw = pd.read_csv(DATA_FILE)
say(f"file shape: {raw.shape} (rows x columns, incl. Id and SalePrice)")
say(f"Id unique: {raw['Id'].is_unique} | duplicated rows (ignoring Id): {raw.drop(columns='Id').duplicated().sum()}")
say(f"description: {len(TYPES)} variables = {len(NUM)} numerical + {len(CAT)} categorical")
say(f"csv columns (minus Id, SalePrice) == description variables: {set(raw.columns) - {'Id', 'SalePrice'} == set(TYPES)}")
numeric_by_dtype = [c for c in raw.select_dtypes("number").columns if c not in ("Id", "SalePrice")]
say(f"TRAP: selecting numeric columns by dtype gives {len(numeric_by_dtype)} columns, "
    f"extra = {sorted(set(numeric_by_dtype) - set(NUM))} (MSSubClass is an integer CODE for a category)")
say(f"pandas {pd.__version__}: default dtypes {raw.dtypes.astype(str).value_counts().to_dict()}")

say("\n" + "=" * 78, "\n2. THE 'NA' / 'None' TRAP\n" + "=" * 78)
default_nan = raw.isna().sum()
literal = pd.read_csv(DATA_FILE, keep_default_na=False, dtype=str)
rows = []
for c in raw.columns:
    n_na, n_none = int((literal[c] == "NA").sum()), int((literal[c] == "None").sum())
    if n_na or n_none:
        kind = TYPES.get(c, "?")
        rows.append((c, kind, n_na, n_none, int(default_nan[c]),
                     "NA = real level" if c in NA_LEVEL else ("None = real level" if "None" in LEVELS.get(c, []) else "missing")))
tab = pd.DataFrame(rows, columns=["variable", "type", "literal 'NA'", "literal 'None'", "NaN with pandas defaults", "meaning (description)"])
say(tab.to_string(index=False))
say(f"\nvariables where the description lists 'NA' as a level ({len(NA_LEVEL)}): {NA_LEVEL}")
say(f"variables where the description lists 'None' as a level: {[c for c in CAT if 'None' in LEVELS[c]]}")

say("\n" + "=" * 78, "\n3. IS EVERY 'NA' LEVEL CONSISTENT WITH THE HOUSE?\n" + "=" * 78)
X, y = load_raw()
no_bsmt = X["TotalBsmtSF"] == 0
say(f"houses with no basement (TotalBsmtSF == 0): {int(no_bsmt.sum())}")
for c in BASEMENT:
    bad = X.index[(X[c] == "NA") & ~no_bsmt].tolist()
    say(f"  {c:13s} 'NA' = {int((X[c] == 'NA').sum()):3d} | 'NA' although the house HAS a basement: {bad}")
say(f"no garage (GarageArea == 0): {int((X.GarageArea == 0).sum())} | GarageYrBlt NaN exactly on these rows: "
    f"{bool((X.GarageYrBlt.isna() == (X.GarageArea == 0)).all())}")
for c in ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]:
    say(f"  {c:13s} 'NA' rows differ from 'no garage' rows: {int(((X[c] == 'NA') != (X.GarageArea == 0)).sum())}")
say(f"FireplaceQu 'NA' vs Fireplaces == 0, mismatches: {int(((X.FireplaceQu == 'NA') != (X.Fireplaces == 0)).sum())}")
say(f"PoolQC 'NA' vs PoolArea == 0, mismatches: {int(((X.PoolQC == 'NA') != (X.PoolArea == 0)).sum())}")
say(f"MasVnrType: {X.MasVnrType.value_counts().to_dict()} | MasVnrArea NaN: {int(X.MasVnrArea.isna().sum())} "
    f"(same rows as type 'NA': {bool((X.MasVnrArea.isna() == (X.MasVnrType == 'NA')).all())})")
say(f"  type 'None' but area > 0: {int(((X.MasVnrType == 'None') & (X.MasVnrArea > 0)).sum())} | "
    f"real type but area == 0: {int((~X.MasVnrType.isin(['None', 'NA']) & (X.MasVnrArea == 0)).sum())}  (small data errors, left as they are)")
say(f"Electrical: {X.Electrical.value_counts().to_dict()}  ('NA' is not a level -> missing)")

say("\n" + "=" * 78, "\n4. GENUINELY MISSING VALUES AFTER THE RECODING\n" + "=" * 78)
R = recode_missing(X)
miss = R.isna().sum()
say(miss[miss > 0].rename("missing").to_frame().assign(type=lambda d: [TYPES[c] for c in d.index]).to_string())
say(f"total missing cells: {int(miss.sum())} of {R.size} ({miss.sum() / R.size:.2%})")

say("\n" + "=" * 78, "\n5. CATEGORICAL LEVELS\n" + "=" * 78)
nlev = R[CAT].nunique().sort_values(ascending=False)
say(f"levels (incl. the 'NA' levels) summed over the 44 variables: {int(nlev.sum())} -> one-hot columns on the full data")
say(f"most levels: {nlev.head(6).to_dict()}")
rare = {c: R[c].value_counts()[lambda s: s <= 2].to_dict() for c in CAT}
rare = {c: v for c, v in rare.items() if v}
say(f"levels seen at most twice in 1460 houses: {sum(len(v) for v in rare.values())}")
for c, v in rare.items():
    say(f"  {c}: {v}")
mismatch = {c: sorted(set(X[c]) - set(LEVELS[c]) - {"NA"}) for c in CAT}
say(f"labels in the data that the description spells differently (harmless for one-hot): "
    f"{ {c: v for c, v in mismatch.items() if v} }")

say("\n" + "=" * 78, "\n6. EXACT LINEAR IDENTITIES AND STRONG CORRELATIONS\n" + "=" * 78)
d1 = X.TotalBsmtSF - (X.BsmtFinSF1 + X.BsmtFinSF2 + X.BsmtUnfSF)
d2 = X.GrLivArea - (X["1stFlrSF"] + X["2ndFlrSF"] + X.LowQualFinSF)
say(f"TotalBsmtSF = BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF   holds on every row: {bool((d1 == 0).all())}")
say(f"GrLivArea   = 1stFlrSF + 2ndFlrSF + LowQualFinSF    holds on every row: {bool((d2 == 0).all())}")
corr = X[NUM].corr()
pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack()
         .rename("r").rename_axis(["feature 1", "feature 2"]).reset_index()
         .loc[lambda d: d.r.abs() >= 0.75].sort_values("r", ascending=False))
say("pairs of numerical features with |correlation| >= 0.75:")
say(pairs.round(3).to_string(index=False))

say("\n" + "=" * 78, "\n7. TARGET: SalePrice\n" + "=" * 78)
for name, v in [("SalePrice", y), ("log(SalePrice)", np.log(y))]:
    jb = stats.jarque_bera(v)
    say(f"{name:15s} mean={v.mean():,.4g} median={v.median():,.4g} sd={v.std():,.4g} "
        f"skew={stats.skew(v):.3f} excess kurtosis={stats.kurtosis(v):.3f} Jarque-Bera p={jb.pvalue:.1e}")
X_tr, X_te, y_tr, y_te = split()
say(f"Box-Cox lambda (MLE): full data {stats.boxcox_normmax(y, method='mle'):.3f} | training data only "
    f"{stats.boxcox_normmax(y_tr, method='mle'):.3f}  (0 would mean exactly log)")
skew = X[NUM].apply(lambda s: stats.skew(s.dropna())).sort_values(ascending=False)
say(f"most skewed numerical features: {skew.head(8).round(1).to_dict()}")

say("\n" + "=" * 78, "\n8. UNUSUAL HOUSES\n" + "=" * 78)
big = X.loc[X.GrLivArea > 4000, ["GrLivArea", "OverallQual", "Neighborhood", "SaleCondition"]].assign(SalePrice=y)
say(big.to_string())
say("Ids 524 and 1299: huge, top quality, but cheap 'Partial' sales in Edwards (the dataset author suggests "
    "removing houses > 4000 sq ft for teaching). Ids 692 and 1183: the two most expensive houses.")

say("\n" + "=" * 78, f"\n9. THE 70/30 SPLIT WITH random_state={SEED}\n" + "=" * 78)
say(f"train {X_tr.shape}, test {X_te.shape}")
for i in (524, 1299, 692, 1183):
    say(f"  Id {i}: in {'TRAINING' if i in X_tr.index else 'TEST'} set")
say(f"mean log price: train {np.log(y_tr).mean():.4f}, test {np.log(y_te).mean():.4f}")
P_tr, P_te, unseen = prepare_manual(X_tr, X_te)
say(f"after Question 1.d: train {P_tr.shape}, test {P_te.shape}; test levels never seen in training: {unseen}")
Rt = recode_missing(X_te)
for u in unseen:
    var, lvl = u.split("_", 1)
    say(f"  {u}: test Ids {Rt.index[Rt[var] == lvl].tolist()} -> their dummy block for {var} is all zeros")
modes = recode_missing(X_tr)[CAT].mode().iloc[0]
say(f"training modes used for the missing categorical values: "
    f"{ {c: modes[c] for c in ['MasVnrType', 'Electrical', 'BsmtExposure', 'BsmtFinType2']} }")
say(f"training means used for the missing numerical values: "
    f"{ {c: round(float(recode_missing(X_tr)[c].mean()), 1) for c in ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']} }")

# Figure: is SalePrice Gaussian? (histogram + normal Q-Q plot, raw and log)
fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
for row, (name, v) in enumerate([("SalePrice (USD)", y), ("log(SalePrice)", np.log(y))]):
    ax[row, 0].hist(v, bins=50, density=True, color="#8fb3d9", edgecolor="white")
    grid = np.linspace(v.min(), v.max(), 300)
    ax[row, 0].plot(grid, stats.norm.pdf(grid, v.mean(), v.std()), color="#b03024", lw=2, label="normal fit")
    ax[row, 0].set_title(f"{name}: skew {stats.skew(v):.2f}")
    ax[row, 0].legend()
    stats.probplot(v, dist="norm", plot=ax[row, 1])
    ax[row, 1].set_title(f"Normal Q-Q plot of {name}")
fig.tight_layout()
fig.savefig(FIG_DIR / "target_distribution.png", dpi=110)
say(f"\nfigure written: {FIG_DIR / 'target_distribution.png'}")
