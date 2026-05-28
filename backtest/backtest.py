import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

DATA_DIR = "data"
FEATURES_FILE = os.path.join(DATA_DIR, "features.csv")
TRAIN_FRACTION = 0.70
ZSCORE_WINDOW = 252
COST_BPS = 2
THRESHOLD_GRID = np.arange(0.5, 3.1, 0.25)
RISK_FREE_RATE = 0.0
FALLBACK_FEATURE = "h1_total_persistence"


def annualized_return(returns, periods=252):
    if len(returns) == 0:
        return 0.0
    return float((1 + returns).prod() ** (periods / len(returns)) - 1)


def sharpe_ratio(returns, rf=RISK_FREE_RATE, periods=252):
    excess = returns - rf / periods
    std = excess.std()
    return float(np.sqrt(periods) * excess.mean() / std) if std > 1e-10 else 0.0


def sortino_ratio(returns, rf=RISK_FREE_RATE, periods=252):
    excess = returns - rf / periods
    downside = excess[excess < 0].std()
    return float(np.sqrt(periods) * excess.mean() / downside) if downside > 1e-10 else 0.0


def max_drawdown(returns):
    if len(returns) == 0:
        return 0.0
    cum = (1 + returns).cumprod()
    roll_max = cum.expanding().max()
    dd = (cum - roll_max) / roll_max
    return float(dd.min())


def calmar_ratio(returns, periods=252):
    mdd = abs(max_drawdown(returns))
    ann = annualized_return(returns, periods)
    return float(ann / mdd) if mdd > 1e-10 else 0.0


def win_rate_monthly(returns):
    if len(returns) < 21:
        return float(np.nan)
    monthly = (1 + returns).resample("ME").prod() - 1
    return float((monthly > 0).mean())


def count_trades(signal):
    return int(signal.diff().abs().fillna(0).sum())


def metrics_dict(returns, label):
    return {
        "period": label,
        "ann_return": round(annualized_return(returns), 4),
        "sharpe": round(sharpe_ratio(returns), 4),
        "sortino": round(sortino_ratio(returns), 4),
        "max_drawdown": round(max_drawdown(returns), 4),
        "calmar": round(calmar_ratio(returns), 4),
        "win_rate_monthly": round(win_rate_monthly(returns), 4),
        "n_trading_days": len(returns),
    }


def load_features(path):
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def identify_best_feature():
    granger_path = os.path.join(DATA_DIR, "results_granger.csv")
    if os.path.exists(granger_path):
        gr = pd.read_csv(granger_path)
        sig = gr[gr.get("significant", pd.Series(False, index=gr.index))]
        if len(sig) > 0:
            return str(sig.sort_values("p_fdr").iloc[0]["feature"])
    spearman_path = os.path.join(DATA_DIR, "results_spearman.csv")
    if os.path.exists(spearman_path):
        sp = pd.read_csv(spearman_path)
        if "significant" in sp.columns:
            sig = sp[sp["significant"]]
            if len(sig) > 0:
                return str(sig.sort_values("p_fdr").iloc[0]["feature"])
    return FALLBACK_FEATURE


def download_spy(start, end):
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    return spy["Close"].squeeze() if isinstance(spy.columns, pd.MultiIndex) else spy["Close"]


def build_signal(feature, threshold, zscore_window=ZSCORE_WINDOW):
    roll_mean = feature.rolling(zscore_window, min_periods=zscore_window // 4).mean()
    roll_std = feature.rolling(zscore_window, min_periods=zscore_window // 4).std()
    z_score = (feature - roll_mean) / (roll_std + 1e-10)
    raw_signal = pd.Series(np.where(z_score > threshold, 0, 1), index=feature.index)
    signal = raw_signal.shift(1).fillna(1)
    return signal, z_score


def apply_transaction_costs(strategy_returns, signal, cost_bps=COST_BPS):
    cost_per_trade = cost_bps / 10_000
    trades = signal.diff().abs().fillna(0)
    return strategy_returns - (trades * cost_per_trade)


def calibrate_threshold(feature, spy_returns, train_end_idx):
    feat_train = feature.iloc[:train_end_idx]
    ret_train = spy_returns.iloc[:train_end_idx]
    rows = []
    for thresh in THRESHOLD_GRID:
        sig, _ = build_signal(feat_train, thresh)
        strat = sig * ret_train
        strat = apply_transaction_costs(strat, sig)
        rows.append({"threshold": thresh, "sharpe": sharpe_ratio(strat.dropna())})
    results = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    return float(results.iloc[0]["threshold"]), results


def run_backtest(feature, spy_prices, threshold):
    spy_ret = np.log(spy_prices / spy_prices.shift(1)).dropna()
    aligned = pd.concat([feature, spy_ret], axis=1).dropna()
    aligned.columns = ["feature", "spy_return"]
    n_total = len(aligned)
    n_train = int(n_total * TRAIN_FRACTION)
    split_date = aligned.index[n_train]
    signal, z_score = build_signal(aligned["feature"], threshold)
    warm_mask = z_score.notna()
    strat_ret = signal * aligned["spy_return"]
    strat_ret = apply_transaction_costs(strat_ret, signal)
    train_mask = (aligned.index < split_date) & warm_mask
    test_mask = (aligned.index >= split_date) & warm_mask
    return {
        "threshold": threshold,
        "split_date": split_date,
        "n_trades_train": count_trades(signal[train_mask]),
        "n_trades_test": count_trades(signal[test_mask]),
        "train_strategy": metrics_dict(strat_ret[train_mask], "train_strategy"),
        "train_benchmark": metrics_dict(aligned["spy_return"][train_mask], "train_benchmark"),
        "test_strategy": metrics_dict(strat_ret[test_mask], "test_strategy"),
        "test_benchmark": metrics_dict(aligned["spy_return"][test_mask], "test_benchmark"),
    }


def save_metrics(results):
    rows = [results["train_strategy"], results["train_benchmark"],
            results["test_strategy"], results["test_benchmark"]]
    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "backtest_results.csv"), index=False)


def main():
    features = load_features(FEATURES_FILE)
    feature_name = identify_best_feature()
    if feature_name not in features.columns:
        feature_name = FALLBACK_FEATURE
    feature = features[feature_name]
    spy_prices = download_spy(str(features.index[0].date()),
                              str(features.index[-1].date()))
    combined = pd.concat([feature, spy_prices], axis=1).dropna()
    feature_aln = combined.iloc[:, 0]
    spy_aln = combined.iloc[:, 1]
    n_train = int(len(combined) * TRAIN_FRACTION)
    best_threshold, _ = calibrate_threshold(
        feature_aln,
        np.log(spy_aln / spy_aln.shift(1)).dropna(),
        n_train,
    )
    results = run_backtest(feature_aln, spy_aln, best_threshold)
    os.makedirs(DATA_DIR, exist_ok=True)
    save_metrics(results)


if __name__ == "__main__":
    main()
