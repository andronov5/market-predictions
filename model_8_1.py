"""Model 8.1 command line interface."""
import sys
from pathlib import Path
import argparse
import datetime as dt
import json
from typing import List

try:
    repo_root = Path(__file__).resolve().parent
except NameError:
    repo_root = Path.cwd()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import backoff
import gspread
from google.auth import default as gauth_default

from model import (
    download_or_load_prices,
    compute_features,
    data_prep_and_feature_engineering,
    run_grid_search,
    run_backtest as _run_backtest,
)

# ---------------------------- CONFIG ------------------------------------
CACHE_FILE = Path("prices_5y.parquet")
TICKERS = [
    "AAPL","MSFT","TSLA","GOOGL","AMZN","NVDA","META","NFLX","AMD","BABA",
    "V","JPM","BAC","KO","DIS","XOM","CVX","INTC","IBM","ORCL",
]
YEARS = 5
END_DATE = dt.date.today()
START_DATE = END_DATE - dt.timedelta(days=YEARS * 365)

FEATURES = [
    "RSI","MACD","MACD_SIGNAL","SMA_10","SMA_50","EMA_10","EMA_50","SMA_ratio",
    "MFI","ATR","BOLL_HBAND","BOLL_LBAND","BOLL_WIDTH","Return_1d","Return_5d",
    "Return_10d","Return_20d","Volatility_10d","Volatility_20d","SPY_Trend",
    "VIX_Level","VIX_Change","RS_Market","OBV","SMA_200","Close_SMA_200",
    "Price_Pctl_90",
]
N_JOBS = -1

# --------------------------- ARGPARSE -----------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run Model 8.1 pipeline steps")
    parser.add_argument("--tickers", type=str, default=",".join(TICKERS), help="Comma separated list of tickers")
    parser.add_argument(
        "--start", type=lambda s: dt.datetime.strptime(s, "%Y-%m-%d").date(), default=START_DATE, help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", type=lambda s: dt.datetime.strptime(s, "%Y-%m-%d").date(), default=END_DATE, help="End date (YYYY-MM-DD)"
    )
    parser.add_argument("--grid-search", action="store_true", help="Run the Optuna grid search step")
    parser.add_argument("--backtest", action="store_true", help="Run the vectorbt backtest step")
    parser.add_argument("--update-sheets", action="store_true", help="Update Google Sheets with portfolio values")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials for grid search")
    parser.add_argument(
        "--feat-threshold", type=float, default=0.8, help="Cumulative importance threshold for feature selection"
    )
    parser.add_argument(
        "--lookahead-days", type=int, default=5, help="Days to look ahead when generating the Target column"
    )
    parser.add_argument(
        "--target-pct", type=float, default=0.02, help="Percent rise threshold for classifying Target=1"
    )
    args, _ = parser.parse_known_args()
    return args

# --------------------------- GOOGLE SHEETS -------------------------------
SPREADSHEET_NAME = "TradingLog"
EQUITY_SHEET = "Equity Tracker"
TRADES_SHEET = "Trades"
PORTFOLIO_FILE = Path("portfolio_values.json")
try:
    ARTEFACT_FILE = Path(__file__).with_name("ml_pipeline.joblib")
except NameError:
    ARTEFACT_FILE = Path.cwd() / "ml_pipeline.joblib"


def get_gs_client():
    creds, _ = gauth_default(
        scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    )
    return gspread.authorize(creds)


def get_sheet():
    client = get_gs_client()
    return client.open(SPREADSHEET_NAME)


# ------------------------------ HELPERS ---------------------------------

def _ensure_worksheet(sheet, title: str, cols: int = 3):
    try:
        return sheet.worksheet(title)
    except gspread.WorksheetNotFound:
        return sheet.add_worksheet(title, rows="100", cols=str(cols))


def _last_row(ws):
    str_vals = ws.col_values(1)
    return len(str_vals)


@backoff.on_exception(backoff.expo, (gspread.exceptions.APIError,), max_tries=5)
def _append_rows(ws, rows: List[list]):
    ws.append_rows(rows, value_input_option="USER_ENTERED")


# -------------------------- EQUITY TRACKER -------------------------------

def update_equity_tracker(json_path: Path = PORTFOLIO_FILE):
    if not json_path.exists():
        raise FileNotFoundError(json_path)

    with open(json_path) as f:
        values = json.load(f)

    sheet = get_sheet()
    ws = _ensure_worksheet(sheet, EQUITY_SHEET)
    start_idx = _last_row(ws)

    header = ["Date", "Equity", "Daily\u00a0PnL", "Cum\u00a0PnL"]
    if start_idx == 0:
        _append_rows(ws, [header])
        start_idx = 1

    rows = []
    init_equity = values[0]
    prev_equity = init_equity if start_idx <= 1 else float(ws.cell(start_idx, 2).value)

    for i, val in enumerate(values[start_idx - 1 :], start=start_idx - 1):
        date = (dt.date.today() - dt.timedelta(days=len(values) - 1 - i)).isoformat()
        daily_pnl = val - prev_equity
        cum_pnl = val - init_equity
        rows.append([date, f"{val:.2f}", f"{daily_pnl:.2f}", f"{cum_pnl:.2f}"])
        prev_equity = val

    if rows:
        for j in range(0, len(rows), 500):
            _append_rows(ws, rows[j : j + 500])
        print(f"✔ Added {len(rows)} equity rows (through {rows[-1][0]}).")
    else:
        print("ℹ Equity sheet already up‑to‑date.")


def run_backtest(artefact_file: Path = ARTEFACT_FILE):
    """Wrapper calling the backtest module with ``artefact_file``."""
    _run_backtest(artefact_file)


# ------------------------------ MAIN ------------------------------------

def main():
    args = parse_args()

    global TICKERS, START_DATE, END_DATE, YEARS
    TICKERS = [t.strip() for t in args.tickers.split(',') if t.strip()]
    START_DATE = args.start
    END_DATE = args.end
    YEARS = (END_DATE - START_DATE).days // 365

    run_all = not (args.grid_search or args.backtest or args.update_sheets)

    if args.grid_search or run_all:
        X_train_sel, y_train = data_prep_and_feature_engineering(
            TICKERS,
            FEATURES,
            CACHE_FILE,
            START_DATE,
            END_DATE,
            n_jobs=N_JOBS,
            threshold=args.feat_threshold,
            lookahead_days=args.lookahead_days,
            target_pct=args.target_pct,
        )
        run_grid_search(X_train_sel, y_train, args.n_trials)

    if args.backtest or run_all:
        artefact_file = Path(__file__).with_name("ml_pipeline.joblib")
        if not artefact_file.exists():
            artefact_file = Path.cwd() / "ml_pipeline.joblib"
        run_backtest(artefact_file)

    if args.update_sheets or run_all:
        update_equity_tracker()


if __name__ == "__main__":
    main()
