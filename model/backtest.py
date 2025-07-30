"""Plain‑Python + vectorbt back‑test helper (no vectorbt.pro required).

Exits are generated with pandas only – stop‑loss, take‑profit, trailing stop,
and time‑out.  The function expects the artefact produced by the training
notebook / pipeline and returns a vectorbt Portfolio.

Save as model/backtest.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import vectorbt as vbt

__all__ = ["run_backtest"]

# ──────────────────────────────────────────────────────────────────────────────
#  Strategy / back‑test parameters
# ──────────────────────────────────────────────────────────────────────────────
INIT_CASH = 100_000          # starting capital
MAX_POS_PCT = 0.10           # max position size as % of equity
STOP_LOSS_PCT = 0.05         # 5 % stop‑loss
TAKE_PROFIT_PCT = 0.08       # 8 % take‑profit
TRAIL_PCT = 0.03             # 3 % trailing stop
EXIT_AFTER = 10              # bars after which we force an exit
MAX_CONCURRENT = 5           # concurrent positions (position sizing handles this)


# ──────────────────────────────────────────────────────────────────────────────
#  Exit helpers – pure pandas
# ──────────────────────────────────────────────────────────────────────────────
def _generate_sl_tp(
    price: pd.DataFrame,
    entries: pd.DataFrame,
    *,
    stop_loss: float,
    take_profit: float,
) -> pd.DataFrame:
    """Boolean DF: exit when price moves beyond stop‑loss / take‑profit."""
    exits = pd.DataFrame(False, index=price.index, columns=price.columns)
    entry_px: dict[str, float | None] = {c: None for c in price.columns}

    for ts, row in price.iterrows():
        for col, px in row.items():
            if entries.at[ts, col]:
                entry_px[col] = px
                continue
            ep = entry_px[col]
            if ep is None:
                continue
            change = (px - ep) / ep
            if change <= -stop_loss or change >= take_profit:
                exits.at[ts, col] = True
                entry_px[col] = None
    return exits


def _generate_trailing(
    price: pd.DataFrame,
    entries: pd.DataFrame,
    *,
    trail_percent: float,
) -> pd.DataFrame:
    """Exit when price falls `trail_percent` from running high after entry."""
    exits = pd.DataFrame(False, index=price.index, columns=price.columns)
    entry_px: dict[str, float | None] = {c: None for c in price.columns}
    high_run: dict[str, float | None] = {c: None for c in price.columns}

    for ts, row in price.iterrows():
        for col, px in row.items():
            if entries.at[ts, col]:
                entry_px[col] = px
                high_run[col] = px
                continue
            ep = entry_px[col]
            if ep is None:
                continue
            if px > high_run[col]:
                high_run[col] = px
            elif px <= high_run[col] * (1 - trail_percent):
                exits.at[ts, col] = True
                entry_px[col] = None
                high_run[col] = None
    return exits


# ──────────────────────────────────────────────────────────────────────────────
#  Main function
# ──────────────────────────────────────────────────────────────────────────────
def run_backtest(artefact_file: Path) -> vbt.Portfolio:
    """Run vectorbt back‑test and save equity / plots to ./results/."""

    if not artefact_file.exists():
        raise FileNotFoundError(
            f"{artefact_file} not found – run the training notebook first."
        )

    print("📂  Loading artefact …")
    art = joblib.load(artefact_file)
    df_bt: pd.DataFrame = art["bt_data"].copy()

    # -- tidy into wide tables ------------------------------------------------
    df_bt["Date"] = pd.to_datetime(df_bt["Date"])
    df_bt.set_index("Date", inplace=True)

    price_close = df_bt.pivot_table(
        index=df_bt.index, columns="Symbol", values="Close"
    ).ffill()
    entries_raw = (
        df_bt.pivot_table(index=df_bt.index, columns="Symbol", values="Predicted")
        .fillna(0)
        .astype(bool)
    )
    probs = (
        df_bt.pivot_table(index=df_bt.index, columns="Symbol", values="Probability")
        .fillna(0.0)
    )

    # -- position sizing: confidence‑weighted --------------------------------
    conf = ((probs - 0.5).clip(lower=0) / 0.5)
    size_pct = MAX_POS_PCT * (0.5 + 0.5 * conf)

    # -- exits ----------------------------------------------------------------
    exit_timeout = entries_raw.shift(EXIT_AFTER).fillna(False)
    exits_sl_tp = _generate_sl_tp(
        price_close, entries_raw, stop_loss=STOP_LOSS_PCT, take_profit=TAKE_PROFIT_PCT
    )
    exits_trail = _generate_trailing(
        price_close, entries_raw, trail_percent=TRAIL_PCT
    )
    exits_combined = exits_sl_tp | exits_trail | exit_timeout

    # -- robust max_orders ----------------------------------------------------
    num_entries = int(entries_raw.to_numpy().sum())
    max_orders_dynamic = max(10_000, num_entries * 2 + 100)
    print(f"🚀  Running vectorbt portfolio (max_orders={max_orders_dynamic:,}) …")

    pf = vbt.Portfolio.from_signals(
        price_close,
        entries_raw,
        exits_combined,
        size=size_pct * INIT_CASH,
        init_cash=INIT_CASH,
        freq="1D",
        max_orders=max_orders_dynamic,
    )

    # -- summary --------------------------------------------------------------
    print(f"✅  Finished – final NAV  ${pf.final_value():,.2f}")
    print(f"Sharpe Ratio : {pf.sharpe_ratio():.2f}")
    print(f"Max Drawdown : {pf.max_drawdown():.2%}")
    print(f"Total Trades : {pf.trades.count()}")
    print(f"Win Rate     : {pf.trades.win_rate():.2%}")

    # -- save outputs ---------------------------------------------------------
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    port_vals = pf.value()
    (results_dir / "portfolio_values.json").write_text(
        json.dumps(port_vals.round(2).to_list())
    )

    plt.figure(figsize=(12, 7))
    # equity
    plt.subplot(2, 1, 1)
    plt.plot(port_vals.index, port_vals.values, label="Equity")
    plt.title("Equity Curve")
    plt.grid(True)
    plt.legend()
    # drawdown
    plt.subplot(2, 1, 2)
    max_curve = port_vals.cummax()
    drawdown = port_vals / max_curve - 1
    plt.fill_between(drawdown.index, 0, drawdown.values, color="red", alpha=0.3)
    plt.title("Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / "backtest_results.png", dpi=150)
    plt.close()

    print("📊  Saved: results/portfolio_values.json & results/backtest_results.png")

    return pf
