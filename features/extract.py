import os
import pickle
import numpy as np
import pandas as pd

DATA_DIR = "data"
INPUT_FILE = os.path.join(DATA_DIR, "persistence_diagrams.pkl")
OUTPUT_FILE = os.path.join(DATA_DIR, "features.csv")


def load_diagrams(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def persistence_entropy(lifetimes):
    if len(lifetimes) == 0:
        return 0.0
    total = lifetimes.sum()
    if total == 0.0:
        return 0.0
    p = lifetimes / total
    return float(-np.sum(p * np.log(p + 1e-12)))


def extract_features_one(dgms, explained_variance):
    h0 = dgms[0]
    h1 = dgms[1]
    h0_finite = h0[h0[:, 1] < np.inf]
    h0_life = h0_finite[:, 1] - h0_finite[:, 0]
    h1_life = h1[:, 1] - h1[:, 0] if len(h1) > 0 else np.array([])
    return {
        "beta0": int(len(h0_finite)),
        "beta1": int(len(h1)),
        "h0_total_persistence": float(h0_life.sum()) if len(h0_life) > 0 else 0.0,
        "h1_total_persistence": float(h1_life.sum()) if len(h1_life) > 0 else 0.0,
        "h0_max_persistence": float(h0_life.max()) if len(h0_life) > 0 else 0.0,
        "h1_max_persistence": float(h1_life.max()) if len(h1_life) > 0 else 0.0,
        "h0_mean_persistence": float(h0_life.mean()) if len(h0_life) > 0 else 0.0,
        "h1_mean_persistence": float(h1_life.mean()) if len(h1_life) > 0 else 0.0,
        "h0_entropy": persistence_entropy(h0_life),
        "h1_entropy": persistence_entropy(h1_life),
        "explained_variance": float(explained_variance),
    }


def extract_all_features(records):
    rows = []
    for record in records:
        features = extract_features_one(record["dgms"], record["explained_variance"])
        features["date"] = record["date"]
        rows.append(features)
    df = pd.DataFrame(rows).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def main():
    records = load_diagrams(INPUT_FILE)
    features = extract_all_features(records)
    os.makedirs(DATA_DIR, exist_ok=True)
    features.to_csv(OUTPUT_FILE)


if __name__ == "__main__":
    main()
