import json
from pathlib import Path

import matplotlib.pyplot as plt
import joblib
import pandas as pd
import vectorbt as vbt

__all__ = ["run_backtest"]


INIT_CASH = 100_000
MAX_POS_PCT = 0.10
STOP_LOSS_PCT = 0.05
TRAIL_PCT = 0.03
TAKE_PROFIT_PCT = 0.08
MAX_CONCURRENT = 5


def _generate_sl_tp(price: pd.DataFrame, entries: pd.DataFrame, *, stop_loss: float, take_profit: float):
    """Generate stop loss and take profit exits."""
    exits_sl = pd.DataFrame(False, index=price.index, columns=price.columns)
    exits_tp = pd.DataFrame(False, index=price.index, columns=price.columns)
    entry_price = {col: None for col in price.columns}
    for i in range(len(price)):
        for col in price.columns:
            if entries.iloc[i][col] and entry_price[col] is None:
                entry_price[col] = price.iloc[i][col]
            if entry_price[col] is not None:
                p = price.iloc[i][col]
                if stop_loss is not None and p <= entry_price[col] * (1 - stop_loss):
                    exits_sl.iloc[i, exits_sl.columns.get_loc(col)] = True
                    entry_price[col] = None
                elif take_profit is not None and p >= entry_price[col] * (1 + take_profit):
                    exits_tp.iloc[i, exits_tp.columns.get_loc(col)] = True
                    entry_price[col] = None
    return exits_sl, exits_tp


def _generate_trailing(price: pd.DataFrame, entries: pd.DataFrame, *, trail_percent: float):
    """Generate trailing stop exits."""
    exits = pd.DataFrame(False, index=price.index, columns=price.columns)
    entry_price = {col: None for col in price.columns}
    high_water = {col: None for col in price.columns}
    for i in range(len(price)):
        for col in price.columns:
            if entries.iloc[i][col] and entry_price[col] is None:
                entry_price[col] = price.iloc[i][col]
                high_water[col] = price.iloc[i][col]
            if entry_price[col] is not None:
                p = price.iloc[i][col]
                if p > high_water[col]:
                    high_water[col] = p
                elif p <= high_water[col] * (1 - trail_percent):
                    exits.iloc[i, exits.columns.get_loc(col)] = True
                    entry_price[col] = None
                    high_water[col] = None
    return exits


def run_backtest(artefact_file: Path):
    """Run the vectorbt backtest and export results."""
    print("📂  Loading artefacts …")
    if not artefact_file.exists():
        raise FileNotFoundError(f"{artefact_file} not found. Run the training pipeline to create it")
    art = joblib.load(artefact_file)
    df_bt = art["bt_data"]

    df_bt["Date"] = pd.to_datetime(df_bt["Date"])
    df_bt.set_index("Date", inplace=True)

    price_close = df_bt.pivot_table(index=df_bt.index, columns="Symbol", values="Close").ffill()
    entries_raw = df_bt.pivot_table(index=df_bt.index, columns="Symbol", values="Predicted").fillna(0).astype(bool)
    probs = df_bt.pivot_table(index=df_bt.index, columns="Symbol", values="Probability").fillna(0.0)

    conf = ((probs - 0.5).clip(lower=0) / 0.5).rolling(1).mean()
    size_pct = MAX_POS_PCT * (0.5 + 0.5 * conf)

    exit_after = 10
    exit_timeout = entries_raw.shift(exit_after).fillna(False)

    if hasattr(vbt, "tsignals"):
        exits_sl, exits_tp = vbt.tsignals.generate_sl_tp(
            price_close, entries_raw, stop_loss=STOP_LOSS_PCT, take_profit=TAKE_PROFIT_PCT
        )
        exits_trail = vbt.tsignals.generate_trailing(
            price_close, entries_raw, trail_percent=TRAIL_PCT
        )
    else:
        exits_sl, exits_tp = _generate_sl_tp(
            price_close, entries_raw, stop_loss=STOP_LOSS_PCT, take_profit=TAKE_PROFIT_PCT
        )
        exits_trail = _generate_trailing(
            price_close, entries_raw, trail_percent=TRAIL_PCT
        )
    exits_combined = exits_sl | exits_tp | exits_trail | exit_timeout

    print("🚀  Running vectorbt portfolio …")
    pf = vbt.Portfolio.from_signals(
        price_close,
        entries_raw,
        exits_combined,
        size=size_pct * INIT_CASH,
        init_cash=INIT_CASH,
        freq="1D",
        max_orders=MAX_CONCURRENT,
    )

    print("✅  Done.   Final NAV = ${:,.2f}".format(pf.final_value()))
    print("Sharpe Ratio : {:.2f}".format(pf.sharpe_ratio()))
    print("Max Drawdown : {:.2%}".format(pf.max_drawdown()))
    print("Total Trades : {}".format(pf.trades.count()))
    print("Win Rate     : {:.2%}".format(pf.trades.win_rate()))

    port_vals = pf.value()
    with open("portfolio_values.json", "w") as f:
        json.dump(port_vals.round(2).to_list(), f)

    plt.figure(figsize=(12, 7))
    plt.subplot(2, 1, 1)
    plt.plot(port_vals.index, port_vals.values, label="Equity")
    plt.title("Equity Curve")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    max_curve = port_vals.cummax()
    plt.fill_between(port_vals.index, 0, (port_vals / max_curve - 1).values, color="red", alpha=0.3)
    plt.title("Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("backtest_results.png", dpi=150)
    plt.close()

    print("📊  Outputs: portfolio_values.json  +  backtest_results.png")
