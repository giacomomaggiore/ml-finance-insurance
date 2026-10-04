"""Shared helpers for the preparation analysis of Project 1.

These scripts only VERIFY the facts written in the helpers/Analysis/*.md files.
They are not the submission: the submitted notebook must be self-contained,
so the notebook will define its own (cleaned-up) versions of these helpers.
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[3] # .../1-linear-regression
DATA_FILE = ROOT / "data" / "Housing.csv"
DESCRIPTION_FILE = ROOT / "data" / "data_description.txt"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
FIG_DIR = ROOT / "helpers" / "Analysis" / "figures"

SEED, TEST_SIZE, N_FOLDS = 42, 0.30, 8


def parse_description(path=DESCRIPTION_FILE):
    """Read the variable types and the listed levels from data_description.txt.

    Header lines look like 'Alley (categorical): Type of alley access'.
    Level lines are indented and look like '       NA \tNo alley access'.
    """
    txt = path.read_text(encoding="utf-8", errors="replace")
    header = re.compile(r"^(\S+)\s+\((numerical|categorical)\):", re.M)
    matches = list(header.finditer(txt))
    bounds = [m.start() for m in matches] + [len(txt)]
    types, levels = {}, {}
    for m, start, end in zip(matches, bounds[:-1], bounds[1:]):
        name = m.group(1)
        types[name] = m.group(2)
        body = txt[start:end].splitlines()[1:]
        levels[name] = [line.strip().split("\t")[0].strip()
                        for line in body if line.strip() and line[0] in " \t"]
    return types, levels


TYPES, LEVELS = parse_description()
NUM = [c for c, t in TYPES.items() if t == "numerical"]      # 35 features
CAT = [c for c, t in TYPES.items() if t == "categorical"]    # 44 features
NA_LEVEL = [c for c in CAT if "NA" in LEVELS[c]]             # 14 features: 'NA' is a real level
BASEMENT = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]


def load_raw():
    """Housing.csv exactly as written in the file.

    - 'Id' becomes the index (it is an identifier, not a feature).
    - Numerical columns: the string 'NA' can only mean 'missing' -> NaN.
    - Categorical columns: keep 'NA' and 'None' as literal strings. pandas would
      otherwise turn BOTH into NaN (its default na_values contain 'NA' and 'None').
    - MSSubClass is stored as integers but is categorical -> cast to str.
    """
    df = pd.read_csv(DATA_FILE, index_col="Id", keep_default_na=False,
                     na_values={c: ["NA"] for c in NUM})
    df[CAT] = df[CAT].astype(str)
    return df.drop(columns="SalePrice"), df["SalePrice"]


def recode_missing(X):
    """Turn every 'NA' that is NOT a valid level into a real missing value (NaN).

    'NA' is a valid level only for the 14 variables whose description lists it,
    and for the 5 basement variables only if the house really has no basement
    (TotalBsmtSF == 0). 'None' (MasVnrType) is always a valid level.
    Stateless: it learns nothing from the data, so it cannot leak information.
    """
    X = X.copy()
    for c in CAT:
        is_na = X[c].eq("NA")
        if c in BASEMENT:
            missing = is_na & (X["TotalBsmtSF"] > 0)
        elif c in NA_LEVEL:
            missing = is_na & False
        else:
            missing = is_na
        X[c] = X[c].mask(missing)
    return X


def split(seed=SEED):
    X, y = load_raw()
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)


def prepare_manual(X_train_raw, X_test_raw):
    """Question 1.d done 'by hand' with pandas (training statistics only)."""
    tr, te = recode_missing(X_train_raw), recode_missing(X_test_raw)
    mean = tr[NUM].mean()
    tr[NUM], te[NUM] = tr[NUM].fillna(mean), te[NUM].fillna(mean)
    mu, sd = tr[NUM].mean(), tr[NUM].std(ddof=0)       # ddof=0 = StandardScaler
    tr[NUM], te[NUM] = (tr[NUM] - mu) / sd, (te[NUM] - mu) / sd
    mode = tr[CAT].mode().iloc[0]
    tr[CAT], te[CAT] = tr[CAT].fillna(mode), te[CAT].fillna(mode)
    tr = pd.get_dummies(tr, columns=CAT, dtype=float)
    te = pd.get_dummies(te, columns=CAT, dtype=float)
    unseen = sorted(set(te.columns) - set(tr.columns))
    te = te.reindex(columns=tr.columns, fill_value=0.0)
    return tr, te, unseen


def make_preprocessor():
    """The same preparation as a scikit-learn object (for Question 3.b)."""
    numeric = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())
    categorical = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    columns = ColumnTransformer([("num", numeric, NUM), ("cat", categorical, CAT)],
                                verbose_feature_names_out=False)
    return Pipeline([("na_levels", FunctionTransformer(recode_missing, feature_names_out="one-to-one")),
                     ("columns", columns)])


class Tee:
    """Print to the screen and to a text file at the same time."""
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, *args):
        text = " ".join(str(a) for a in args)
        print(text)
        self.f.write(text + "\n")
        self.f.flush()
