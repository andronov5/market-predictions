import sys
from pathlib import Path
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from model.utils import get_max_profit


def test_get_max_profit_normal_case():
    prices = pd.Series([100, 90, 95, 110, 105])
    assert get_max_profit(prices) == 20.0


def test_get_max_profit_increasing_prices():
    prices = pd.Series([1, 2, 3, 4, 5])
    assert get_max_profit(prices) == 4.0


def test_get_max_profit_decreasing_prices():
    prices = pd.Series([5, 4, 3, 2, 1])
    assert get_max_profit(prices) == 0.0
