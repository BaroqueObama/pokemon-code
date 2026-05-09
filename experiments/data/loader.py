"""Real dataset loaders -- Adult, COMPAS, ACS PUMS."""

import hashlib
import io
import time

import numpy as np
import pandas as pd
import requests

from experiments.config import DATA_RAW_DIR, ensure_dirs
from experiments.data import FairnessDataset

_ADULT_TRAIN_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)
_ADULT_TEST_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
)
_COMPAS_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/"
    "master/compas-scores-two-years.csv"
)

_ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


def _download_with_cache(
    url: str, filename: str, retries: int = 3,
    expected_sha256: str | None = None,
) -> str:
    """Download *url* to DATA_RAW_DIR/filename and return its text."""
    ensure_dirs()
    path = DATA_RAW_DIR / filename
    if path.exists():
        return path.read_text()

    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            text = resp.text
            if expected_sha256 is not None:
                actual = _sha256(text)
                assert actual == expected_sha256, (
                    f"SHA-256 mismatch for {filename}: "
                    f"expected {expected_sha256[:16]}..., got {actual[:16]}..."
                )
            path.write_text(text)
            return text
        except (requests.RequestException, IOError) as e:
            if attempt == retries - 1:
                raise RuntimeError(
                    f"Failed to download {url} after {retries} attempts: {e}"
                ) from e
            time.sleep(2**attempt)

    raise RuntimeError("Unreachable")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _standardize_continuous(
    X: np.ndarray, continuous_mask: np.ndarray
) -> np.ndarray:
    """Standardize continuous columns to zero mean and unit variance."""
    X = X.copy()
    cols = np.where(continuous_mask)[0]
    if len(cols) == 0:
        return X
    mean = X[:, cols].mean(axis=0)
    std = X[:, cols].std(axis=0)
    std[std == 0] = 1.0
    X[:, cols] = (X[:, cols] - mean) / std
    return X


def load_adult(seed: int = 42) -> FairnessDataset:
    """Load the Adult Income dataset (UCI)."""
    train_text = _download_with_cache(_ADULT_TRAIN_URL, "adult.data")
    test_text = _download_with_cache(_ADULT_TEST_URL, "adult.test")

    # skip first line of test file ("|1x3 Cross validator" header)
    df_train = pd.read_csv(
        io.StringIO(train_text),
        header=None,
        names=_ADULT_COLUMNS,
        skipinitialspace=True,
        na_values="?",
    )
    test_lines = test_text.strip().split("\n")
    if test_lines[0].startswith("|"):
        test_lines = test_lines[1:]
    df_test = pd.read_csv(
        io.StringIO("\n".join(test_lines)),
        header=None,
        names=_ADULT_COLUMNS,
        skipinitialspace=True,
        na_values="?",
    )

    df_test["income"] = df_test["income"].str.rstrip(".")
    df = pd.concat([df_train, df_test], ignore_index=True)

    df = df.drop(columns=["fnlwgt"])
    df = df.dropna().reset_index(drop=True)

    y = (df["income"].str.strip() == ">50K").astype(np.int64).values
    group = (df["sex"].str.strip() == "Male").astype(np.int64).values

    df_features = df.drop(columns=["income", "sex"])
    continuous_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_features.select_dtypes(exclude=[np.number]).columns.tolist()

    df_encoded = pd.get_dummies(df_features, columns=categorical_cols, drop_first=True)
    feature_names = df_encoded.columns.tolist()
    X = df_encoded.values.astype(np.float64)

    continuous_mask = np.array(
        [name in continuous_cols for name in feature_names], dtype=bool
    )
    X = _standardize_continuous(X, continuous_mask)

    assert np.all(np.isfinite(X)), f"Found {np.sum(~np.isfinite(X))} non-finite values"
    assert X.shape[0] > 40_000, f"Expected >40K rows after cleaning, got {X.shape[0]}"

    emp_p_a = float(y[group == 1].mean())
    emp_p_b = float(y[group == 0].mean())
    assert abs(emp_p_a - 0.306) < 0.03, (
        f"Male base rate {emp_p_a:.3f} too far from expected ~0.306"
    )
    assert abs(emp_p_b - 0.109) < 0.03, (
        f"Female base rate {emp_p_b:.3f} too far from expected ~0.109"
    )

    return FairnessDataset(
        X=X,
        y=y,
        group=group,
        feature_names=feature_names,
        name="adult",
        base_rates={"a": emp_p_a, "b": emp_p_b},
    )


_COMPAS_SHA256 = "d180e410066da845eceb452417fbf8b119633f05526bcb29ef2c5b546d8946c5"


def load_compas(seed: int = 42) -> FairnessDataset:
    """Load the ProPublica COMPAS recidivism dataset."""
    text = _download_with_cache(
        _COMPAS_URL, "compas-scores-two-years.csv",
        expected_sha256=_COMPAS_SHA256,
    )
    df = pd.read_csv(io.StringIO(text))

    # ProPublica's cohort selection
    df = df[
        (df["days_b_screening_arrest"] >= -30)
        & (df["days_b_screening_arrest"] <= 30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ].copy()

    df = df[df["race"].isin(["African-American", "Caucasian"])].reset_index(drop=True)

    y = df["two_year_recid"].astype(np.int64).values
    group = (df["race"] == "African-American").astype(np.int64).values

    df["sex_encoded"] = (df["sex"] == "Male").astype(int)
    age_cat_dummies = pd.get_dummies(df["age_cat"], prefix="age_cat", drop_first=True)

    feature_cols = ["age", "juv_fel_count", "juv_misd_count", "juv_other_count",
                    "priors_count", "sex_encoded"]
    df_features = pd.concat(
        [df[feature_cols], age_cat_dummies], axis=1
    )
    feature_names = df_features.columns.tolist()
    X = df_features.values.astype(np.float64)

    continuous_cols = {"age", "juv_fel_count", "juv_misd_count",
                       "juv_other_count", "priors_count"}
    continuous_mask = np.array(
        [name in continuous_cols for name in feature_names], dtype=bool
    )
    X = _standardize_continuous(X, continuous_mask)

    assert np.all(np.isfinite(X)), f"Found non-finite values in COMPAS features"

    emp_p_a = float(y[group == 1].mean())
    emp_p_b = float(y[group == 0].mean())
    assert abs(emp_p_a - 0.52) < 0.03, (
        f"Black base rate {emp_p_a:.3f} too far from expected ~0.52"
    )
    assert abs(emp_p_b - 0.39) < 0.03, (
        f"White base rate {emp_p_b:.3f} too far from expected ~0.39"
    )

    return FairnessDataset(
        X=X,
        y=y,
        group=group,
        feature_names=feature_names,
        name="compas",
        base_rates={"a": emp_p_a, "b": emp_p_b},
    )


def load_acs_pums(
    states: list[str] | None = None,
    subsample_n: int = 20_000,
    seed: int = 42,
) -> FairnessDataset:
    """Load ACS PUMS income data via folktables."""
    from folktables import ACSDataSource, ACSIncome

    if states is None:
        states = ["CA"]

    ensure_dirs()
    data_source = ACSDataSource(
        survey_year="2018", horizon="1-Year", survey="person",
        root_dir=str(DATA_RAW_DIR),
    )
    acs_data = data_source.get_data(states=states, download=True)
    X_raw, y_raw, group_raw = ACSIncome.df_to_numpy(acs_data)

    y = y_raw.astype(np.int64)
    group = (group_raw == 1).astype(np.int64)  # RAC1P=1 is White

    rng = np.random.default_rng(seed)
    n_total = len(y)
    if n_total > subsample_n:
        strata = y * 2 + group
        selected = []
        for s in np.unique(strata):
            mask = np.where(strata == s)[0]
            frac = len(mask) / n_total
            n_s = max(1, int(subsample_n * frac))
            chosen = rng.choice(mask, size=min(n_s, len(mask)), replace=False)
            selected.extend(chosen)
        selected = np.array(selected)
        rng.shuffle(selected)
        selected = selected[:subsample_n]
        X_raw = X_raw[selected]
        y = y[selected]
        group = group[selected]

    X = X_raw.astype(np.float64)
    feature_names = [f"acs_feat_{i}" for i in range(X.shape[1])]

    continuous_mask = np.ones(X.shape[1], dtype=bool)
    X = _standardize_continuous(X, continuous_mask)

    assert np.all(np.isfinite(X)), f"Found non-finite values in ACS PUMS features"

    emp_p_a = float(y[group == 1].mean())
    emp_p_b = float(y[group == 0].mean())

    return FairnessDataset(
        X=X,
        y=y,
        group=group,
        feature_names=feature_names,
        name="acs_pums",
        base_rates={"a": emp_p_a, "b": emp_p_b},
    )


def train_val_test_split(
    dataset: FairnessDataset,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[FairnessDataset, FairnessDataset, FairnessDataset]:
    """Return (train, val, test) splits stratified by (y, group)."""
    assert 0 < train_frac < 1 and 0 < val_frac < 1
    assert train_frac + val_frac < 1

    rng = np.random.default_rng(seed)
    strata = dataset.y * 2 + dataset.group  # 4 strata: {0, 1, 2, 3}

    train_idx, val_idx, test_idx = [], [], []
    for s in np.unique(strata):
        mask = np.where(strata == s)[0]
        rng.shuffle(mask)
        n_s = len(mask)
        n_train = int(n_s * train_frac)
        n_val = int(n_s * val_frac)
        train_idx.extend(mask[:n_train])
        val_idx.extend(mask[n_train : n_train + n_val])
        test_idx.extend(mask[n_train + n_val :])

    def _make_subset(indices: list[int]) -> FairnessDataset:
        idx = np.array(indices)
        y_sub = dataset.y[idx]
        g_sub = dataset.group[idx]
        emp_p_a = float(y_sub[g_sub == 1].mean()) if (g_sub == 1).any() else 0.0
        emp_p_b = float(y_sub[g_sub == 0].mean()) if (g_sub == 0).any() else 0.0
        return FairnessDataset(
            X=dataset.X[idx],
            y=y_sub,
            group=g_sub,
            feature_names=dataset.feature_names,
            name=dataset.name,
            base_rates={"a": emp_p_a, "b": emp_p_b},
        )

    train_ds = _make_subset(train_idx)
    val_ds = _make_subset(val_idx)
    test_ds = _make_subset(test_idx)

    for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        for g in [0, 1]:
            for label in [0, 1]:
                count = ((ds.group == g) & (ds.y == label)).sum()
                assert count >= 10, (
                    f"{split_name} split has only {count} samples in "
                    f"(group={g}, y={label}) -- need at least 10"
                )

    return train_ds, val_ds, test_ds
