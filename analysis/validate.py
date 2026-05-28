import os
import numpy as np
import pandas as pd

DATA_DIR = "data"
FEATURES_FILE = os.path.join(DATA_DIR, "features.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "validation_stress_summary.csv")

STRESS_PERIODS = [
    ("GFC", "2008-09-01", "2009-06-30"),
    ("Flash Crash", "2010-05-01", "2010-06-30"),
    ("Q4 Selloff", "2018-10-01", "2018-12-31"),
    ("COVID Crash", "2020-02-20", "2020-04-30"),
    ("Rate Hikes", "2022-01-01", "2022-10-31"),
]


def stress_mask(index):
    mask = pd.Series(False, index=index)
    for _, start, end in STRESS_PERIODS:
        mask |= (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
    return mask


def stress_summary(features):
    mask = stress_mask(features.index)
    rows = []
    for feat in features.columns:
        inside = features.loc[mask, feat].dropna()
        outside = features.loc[~mask, feat].dropna()
        rows.append({
            "feature": feat,
            "mean_stress": float(inside.mean()) if len(inside) else np.nan,
            "mean_normal": float(outside.mean()) if len(outside) else np.nan,
            "ratio": float(inside.mean() / outside.mean())
                     if len(outside) and outside.mean() != 0 else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    features = pd.read_csv(FEATURES_FILE, index_col=0, parse_dates=True).sort_index()
    summary = stress_summary(features)
    os.makedirs(DATA_DIR, exist_ok=True)
    summary.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()
