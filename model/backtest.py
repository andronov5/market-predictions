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

    exits_sl, exits_tp = vbt.tsignals.generate_sl_tp(
        price_close, entries_raw, stop_loss=STOP_LOSS_PCT, take_profit=TAKE_PROFIT_PCT
    )
    exits_trail = vbt.tsignals.generate_trailing(price_close, entries_raw, trail_percent=TRAIL_PCT)
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
