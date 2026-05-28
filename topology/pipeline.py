import os
import pickle
import numpy as np
import pandas as pd
from ripser import ripser as rips
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

WINDOW_SIZE = 30
N_COMPONENTS = 10
MAXDIM = 1
STEP = 1
DATA_DIR = "data"
INPUT_FILE = os.path.join(DATA_DIR, "returns.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "persistence_diagrams.pkl")


def load_returns(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.dropna(axis=1, how="any")


def build_point_cloud(window, n_components):
    scaler = StandardScaler()
    window_scaled = scaler.fit_transform(window)
    n_comp = min(n_components, window.shape[0] - 1, window.shape[1])
    pca = PCA(n_components=n_comp)
    cloud = pca.fit_transform(window_scaled)
    return cloud, float(pca.explained_variance_ratio_.sum())


def compute_persistence(cloud, maxdim):
    return rips(cloud, maxdim=maxdim)["dgms"]


def run_pipeline(returns, window_size, n_components, maxdim, step):
    values = returns.values
    dates = returns.index
    n_days, n_stocks = values.shape
    end_indices = range(window_size, n_days + 1, step)
    records = []
    for end_idx in end_indices:
        start_idx = end_idx - window_size
        window = values[start_idx:end_idx]
        cloud, explained = build_point_cloud(window, n_components)
        dgms = compute_persistence(cloud, maxdim)
        records.append({
            "date": dates[end_idx - 1],
            "window_start": dates[start_idx],
            "dgms": dgms,
            "explained_variance": explained,
            "n_stocks": n_stocks,
            "window_size": window_size,
        })
    return records


def main():
    if N_COMPONENTS >= WINDOW_SIZE:
        raise ValueError("N_COMPONENTS must be less than WINDOW_SIZE")
    returns = load_returns(INPUT_FILE)
    records = run_pipeline(returns, WINDOW_SIZE, N_COMPONENTS, MAXDIM, STEP)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
