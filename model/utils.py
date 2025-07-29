import pandas as pd

__all__ = ["get_max_profit"]

def get_max_profit(prices: pd.Series) -> float:
    """Return max profit from single buy/sell transaction."""
    if prices.empty:
        return 0.0
    min_price = prices.iloc[0]
    max_profit = 0.0
    for price in prices.iloc[1:]:
        if price < min_price:
            min_price = price
        else:
            profit = price - min_price
            if profit > max_profit:
                max_profit = profit
    return float(max_profit)
