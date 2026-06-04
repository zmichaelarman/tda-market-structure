import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

DATA_DIR = "data"
FEATURES_FILE = os.path.join(DATA_DIR, "features.csv")
ALPHA = 0.05
GRANGER_MAXLAG = 21
LEAD_HORIZONS = [0, 1, 5, 10, 21]

FEATURES_TO_TEST = [
    "beta1",
    "h1_total_persistence",
    "h1_max_persistence",
    "h1_entropy",
    "h0_total_persistence",
]

STRESS_PERIODS = [
    ("GFC", "2008-09-01", "2009-06-30"),
    ("Flash Crash", "2010-05-01", "2010-06-30"),
    ("Q4 Selloff", "2018-10-01", "2018-12-31"),
    ("COVID Crash", "2020-02-20", "2020-04-30"),
    ("Rate Hikes", "2022-01-01", "2022-10-31"),
]


def load_features(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df[FEATURES_TO_TEST].dropna()


def download_market_data(start, end):
    targets = pd.DataFrame()
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    spy_close = spy["Close"].squeeze() if isinstance(spy.columns, pd.MultiIndex) else spy["Close"]
    spy_ret = np.log(spy_close / spy_close.shift(1))
    targets["spy_return"] = spy_ret
    targets["realized_vol"] = spy_ret.rolling(21, min_periods=15).std() * np.sqrt(252)
    for H in [5, 21]:
        fwd = (spy_ret.shift(-H)
                       .rolling(H, min_periods=max(3, H // 2))
                       .std() * np.sqrt(252))
        targets[f"fwd_vol_{H}"] = fwd
    fwd_min = spy_close.rolling(21).min().shift(-21)
    targets["fwd_drawdown_21"] = (fwd_min - spy_close) / spy_close
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
    vix_close = vix["Close"].squeeze() if isinstance(vix.columns, pd.MultiIndex) else vix["Close"]
    targets["vix"] = vix_close
    return targets


def build_stress_mask(index):
    mask = pd.Series(False, index=index)
    for _, start, end in STRESS_PERIODS:
        mask |= (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
    return mask


def run_adf_tests(features):
    rows = []
    for feat in features.columns:
        series = features[feat].dropna()
        try:
            stat, p, used_lag, nobs, crit_vals, _ = adfuller(series, autolag="AIC", regression="ct")
            rows.append({
                "feature": feat,
                "adf_stat": round(stat, 4),
                "p_value": round(p, 4),
                "used_lag": used_lag,
                "stationary": p < ALPHA,
            })
        except Exception:
            pass
    return pd.DataFrame(rows)


def run_spearman_tests(features, targets, horizons):
    target_cols = [c for c in ["vix", "realized_vol", "fwd_vol_5", "fwd_vol_21", "fwd_drawdown_21"]
                   if c in targets.columns]
    rows = []
    for feat in features.columns:
        for target in target_cols:
            for h in horizons:
                feat_series = features[feat]
                target_series = targets[target].shift(-h)
                aligned = pd.concat([feat_series, target_series], axis=1).dropna()
                if len(aligned) < 50:
                    continue
                rho, p = stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
                rows.append({
                    "feature": feat,
                    "target": target,
                    "horizon": h,
                    "n_obs": len(aligned),
                    "spearman_rho": round(rho, 4),
                    "p_raw": round(p, 6),
                })
    df = pd.DataFrame(rows)
    if len(df) > 0:
        reject, p_corr, _, _ = multipletests(df["p_raw"], method="fdr_bh")
        df["p_fdr"] = np.round(p_corr, 6)
        df["significant"] = reject
    return df


def run_mannwhitney_tests(features, stress_mask):
    stress_aligned = stress_mask.reindex(features.index).fillna(False)
    rows = []
    for feat in features.columns:
        series = features[feat]
        inside = series[stress_aligned].dropna()
        outside = series[~stress_aligned].dropna()
        if len(inside) < 10 or len(outside) < 10:
            continue
        u_stat, p = stats.mannwhitneyu(inside, outside, alternative="two-sided")
        n1, n2 = len(inside), len(outside)
        r_rb = 1 - (2 * u_stat) / (n1 * n2)
        rows.append({
            "feature": feat,
            "n_stress": n1,
            "n_normal": n2,
            "median_stress": round(inside.median(), 4),
            "median_normal": round(outside.median(), 4),
            "direction": "higher" if inside.median() > outside.median() else "lower",
            "u_stat": round(u_stat, 2),
            "effect_size_r": round(r_rb, 4),
            "p_raw": round(p, 6),
        })
    df = pd.DataFrame(rows)
    if len(df) > 0:
        reject, p_corr, _, _ = multipletests(df["p_raw"], method="fdr_bh")
        df["p_fdr"] = np.round(p_corr, 6)
        df["significant"] = reject
    return df


def run_granger_tests(features, targets, adf_results, maxlag):
    non_stationary = set(adf_results.loc[~adf_results["stationary"], "feature"].tolist())
    granger_targets = [t for t in ["fwd_vol_21", "vix"] if t in targets.columns]
    rows = []
    for feat in features.columns:
        feat_series = features[feat].copy()
        used_diff = feat in non_stationary
        if used_diff:
            feat_series = feat_series.diff().dropna()
        for target in granger_targets:
            target_series = targets[target].copy()
            combined = pd.concat([target_series, feat_series], axis=1).dropna()
            combined.columns = ["target", "feature"]
            if len(combined) < maxlag * 3 + 10:
                continue
            try:
                result = grangercausalitytests(combined[["target", "feature"]].values,
                                               maxlag=maxlag, verbose=False)
                lag_pvals = {lag: result[lag][0]["ssr_ftest"][1]
                             for lag in range(1, maxlag + 1)}
                best_lag = min(lag_pvals, key=lag_pvals.get)
                rows.append({
                    "feature": feat,
                    "target": target,
                    "differenced": used_diff,
                    "best_lag_days": best_lag,
                    "best_p_raw": round(lag_pvals[best_lag], 6),
                    "n_obs": len(combined),
                })
            except Exception:
                pass
    df = pd.DataFrame(rows)
    if len(df) > 0:
        reject, p_corr, _, _ = multipletests(df["best_p_raw"], method="fdr_bh")
        df["p_fdr"] = np.round(p_corr, 6)
        df["significant"] = reject
    return df


def main():
    features = load_features(FEATURES_FILE)
    targets = download_market_data(str(features.index[0].date()),
                                   str(features.index[-1].date()))
    targets = targets.reindex(features.index, method="ffill")
    stress_mask = build_stress_mask(features.index)
    adf_results = run_adf_tests(features)
    spearman_results = run_spearman_tests(features, targets, LEAD_HORIZONS)
    mw_results = run_mannwhitney_tests(features, stress_mask)
    granger_results = run_granger_tests(features, targets, adf_results, GRANGER_MAXLAG)
    os.makedirs(DATA_DIR, exist_ok=True)
    adf_results.to_csv(os.path.join(DATA_DIR, "results_adf.csv"), index=False)
    spearman_results.to_csv(os.path.join(DATA_DIR, "results_spearman.csv"), index=False)
    mw_results.to_csv(os.path.join(DATA_DIR, "results_mannwhitney.csv"), index=False)
    granger_results.to_csv(os.path.join(DATA_DIR, "results_granger.csv"), index=False)


if __name__ == "__main__":
    main()
