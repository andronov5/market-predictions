import datetime as dt
from pathlib import Path

import pandas as pd
import yfinance as yf
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
import ta
from ta.volatility import BollingerBands


__all__ = [
    "download_or_load_prices",
    "compute_features",
    "data_prep_and_feature_engineering",
]


# ---------------------------------------------------------------------------
# Price download/cache helper
# ---------------------------------------------------------------------------

def download_or_load_prices(tickers, cache_file: Path, start_date: dt.date, end_date: dt.date):
    """Download price history or load from ``cache_file``."""

    def _download(start, end):
        raw = yf.download(
            " ".join(tickers),
            start=start,
            end=end,
            progress=False,
            group_by="ticker",
        )
        return raw.stack(level=0, future_stack=True).swaplevel().sort_index()

    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            lvl0 = pd.to_datetime(df.index.get_level_values(0), errors="coerce")
            if lvl0.isna().any():
                raise ValueError("corrupt index detected")
            last_cached = lvl0.max().date()
            if last_cached < end_date - dt.timedelta(days=1):
                df = pd.concat([df, _download(last_cached + dt.timedelta(days=1), end_date)])
        except Exception:
            cache_file.unlink(missing_ok=True)
            df = _download(start_date, end_date)
    else:
        df = _download(start_date, end_date)

    df.to_parquet(cache_file)
    return df


# ---------------------------------------------------------------------------
# Feature engineering utilities
# ---------------------------------------------------------------------------

def compute_features(df, market, lookahead_days: int = 5, target_pct: float = 0.02):
    """Compute technical indicator features and the target column."""
    df = df.dropna(subset=["Close", "High", "Low", "Open", "Volume"]).copy()
    df["Target"] = (
        df["Close"].shift(-lookahead_days) >= df["Close"] * (1 + target_pct)
    ).astype(int)
    df["RSI"] = ta.momentum.rsi(df["Close"], 14)
    df["MACD"] = ta.trend.macd(df["Close"], 26, 12)
    df["MACD_SIGNAL"] = ta.trend.macd_signal(df["Close"], 26, 12, 9)
    df["SMA_10"] = ta.trend.sma_indicator(df["Close"], 10)
    df["SMA_50"] = ta.trend.sma_indicator(df["Close"], 50)
    df["EMA_10"] = ta.trend.ema_indicator(df["Close"], 10)
    df["EMA_50"] = ta.trend.ema_indicator(df["Close"], 50)
    df["SMA_ratio"] = df["SMA_10"] / df["SMA_50"]
    df["MFI"] = ta.volume.money_flow_index(df["High"], df["Low"], df["Close"], df["Volume"], 14)
    df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], 14)
    bb = BollingerBands(df["Close"], 20, 2)
    df["BOLL_HBAND"] = bb.bollinger_hband()
    df["BOLL_LBAND"] = bb.bollinger_lband()
    df["BOLL_WIDTH"] = (df["BOLL_HBAND"] - df["BOLL_LBAND"]) / df["Close"]
    df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
    df["SMA_200"] = ta.trend.sma_indicator(df["Close"], 200)
    df["Close_SMA_200"] = df["Close"] / df["SMA_200"]
    df["Price_Pctl_90"] = df["Close"].rolling(90).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    for w in (1, 5, 10, 20):
        df[f"Return_{w}d"] = df["Close"].pct_change(w)
    df["Volatility_10d"] = df["Close"].pct_change().rolling(10).std()
    df["Volatility_20d"] = df["Close"].pct_change().rolling(20).std()
    spy = market["SPY"].reindex(df.index).ffill()
    vix = market["^VIX"].reindex(df.index).ffill()
    df["SPY_Trend"] = (spy > spy.rolling(20).mean()).astype(int)
    df["VIX_Level"] = vix
    df["VIX_Change"] = vix.pct_change(5)
    df["RS_Market"] = (df["Close"] / df["Close"].shift(20)) / (spy / spy.shift(20))
    return df.dropna()


def data_prep_and_feature_engineering(
    tickers,
    features,
    cache_file: Path,
    start_date: dt.date,
    end_date: dt.date,
    n_jobs: int = -1,
    threshold: float = 0.8,
    lookahead_days: int = 5,
    target_pct: float = 0.02,
):
    """Download prices, compute features and return selected training data."""

    all_tickers = tickers + ["SPY", "^VIX"]
    prices = download_or_load_prices(all_tickers, cache_file, start_date, end_date)

    price_dict = {
        tkr: grp.droplevel("Ticker")
        for tkr, grp in prices.groupby(level="Ticker")
        if tkr in all_tickers
    }
    market_dict = {k: price_dict[k]["Close"] for k in ["SPY", "^VIX"]}

    feat_list = Parallel(n_jobs=n_jobs)(
        delayed(compute_features)(price_dict[tkr], market_dict, lookahead_days, target_pct)
        for tkr in tickers
    )
    all_data = pd.concat(feat_list, keys=tickers)
    all_data.index.names = ["Date", "Symbol"]

    cut = int(len(all_data) * 0.85)
    train_df = all_data.iloc[:cut]
    X_train, y_train = train_df[features], train_df["Target"]

    rf_sel = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42).fit(X_train, y_train)
    imp = pd.Series(rf_sel.feature_importances_, index=features).sort_values(ascending=False)
    selected_features = imp.loc[(imp.cumsum() / imp.sum()) <= threshold].index.tolist()

    X_train_sel = X_train[selected_features]
    return X_train_sel, y_train
