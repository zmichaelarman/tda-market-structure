import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf

START_DATE = "2004-01-01"
END_DATE = "2024-12-31"
MAX_MISSING_PCT = 0.05
DATA_DIR = "data"


def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        tickers = tables[0]["Symbol"].tolist()
        tickers = [t.replace(".", "-") for t in tickers]
        return sorted(set(tickers))
    except Exception:
        return _fallback_tickers()


def _fallback_tickers():
    return sorted([
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "ACN", "IBM",
        "JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP",
        "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR",
        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX",
        "GOOGL", "META", "NFLX", "DIS", "VZ", "T",
        "CAT", "HON", "UPS", "BA", "GE",
        "PG", "KO", "PEP", "COST", "WMT",
        "XOM", "CVX", "COP",
        "LIN", "APD",
        "PLD", "AMT",
        "NEE", "DUK",
    ])


def download_prices(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False, threads=True)
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        elif "Adj Close" in raw.columns.get_level_values(0):
            prices = raw["Adj Close"].copy()
        else:
            raise ValueError("Unexpected column structure from yfinance")
    else:
        prices = raw[["Close"]].copy()
    prices.index.name = "date"
    prices.index = pd.to_datetime(prices.index)
    return prices


def clean_prices(prices, max_missing_pct):
    n_dates = len(prices)
    missing_frac = prices.isna().sum() / n_dates
    dropped = missing_frac[missing_frac > max_missing_pct].index.tolist()
    kept = prices.drop(columns=dropped).ffill().dropna(how="all")
    return kept


def compute_log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna(how="all")


def main():
    tickers = get_sp500_tickers()
    prices_raw = download_prices(tickers, START_DATE, END_DATE)
    prices_clean = clean_prices(prices_raw, MAX_MISSING_PCT)
    returns = compute_log_returns(prices_clean)
    os.makedirs(DATA_DIR, exist_ok=True)
    prices_clean.to_csv(os.path.join(DATA_DIR, "prices.csv"))
    returns.to_csv(os.path.join(DATA_DIR, "returns.csv"))


if __name__ == "__main__":
    main()
