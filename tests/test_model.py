import importlib.util
import importlib.machinery
import types
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import datetime as dt
import json
import pytest


def load_model():
    path = Path(__file__).resolve().parents[1] / 'Model_8.1'
    # stub heavy dependencies if missing
    modules = [
        'backoff', 'gspread', 'google', 'google.auth', 'joblib', 'lightgbm',
        'matplotlib', 'matplotlib.pyplot', 'optuna', 'vectorbt', 'xgboost',
        'yfinance', 'sklearn', 'sklearn.ensemble', 'sklearn.metrics',
        'sklearn.model_selection', 'ta', 'ta.momentum', 'ta.trend',
        'ta.volatility', 'ta.volume'
    ]
    for name in modules:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    # minimal stubs
    joblib = sys.modules['joblib']
    joblib.Parallel = lambda *a, **k: []
    joblib.delayed = lambda f: f
    gspread = sys.modules['gspread']
    gspread.WorksheetNotFound = type('WorksheetNotFound', (Exception,), {})
    gspread.authorize = lambda creds: None
    gspread.exceptions = types.SimpleNamespace(APIError=Exception)
    class DummyCreds:
        pass
    sys.modules['google.auth'].default = lambda scopes=None: (DummyCreds(), None)
    matplotlib_pyplot = types.ModuleType('matplotlib.pyplot')
    sys.modules['matplotlib.pyplot'] = matplotlib_pyplot
    backoff = sys.modules['backoff']
    def on_exception(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator
    backoff.on_exception = on_exception
    backoff.expo = lambda *a, **k: None
    sklearn_ensemble = sys.modules['sklearn.ensemble']
    sklearn_ensemble.RandomForestClassifier = object
    sklearn_ensemble.VotingClassifier = object
    sklearn_metrics = sys.modules['sklearn.metrics']
    sklearn_metrics.precision_score = lambda *a, **k: 0
    sklearn_metrics.make_scorer = lambda *a, **k: None
    sk_ms = sys.modules['sklearn.model_selection']
    sk_ms.TimeSeriesSplit = object
    sk_ms.cross_val_score = lambda *a, **k: [0]
    # ta library stubs
    ta = sys.modules.setdefault('ta', types.ModuleType('ta'))
    ta.momentum = types.ModuleType('ta.momentum')
    ta.trend = types.ModuleType('ta.trend')
    ta.volatility = types.ModuleType('ta.volatility')
    ta.volume = types.ModuleType('ta.volume')
    sys.modules['ta.momentum'] = ta.momentum
    sys.modules['ta.trend'] = ta.trend
    sys.modules['ta.volatility'] = ta.volatility
    sys.modules['ta.volume'] = ta.volume
    ta.momentum.rsi = lambda s, n=14: pd.Series(50, index=s.index)
    ta.trend.macd = lambda s, slow, fast: pd.Series(0, index=s.index)
    ta.trend.macd_signal = lambda s, slow, fast, signal: pd.Series(0, index=s.index)
    ta.trend.sma_indicator = lambda s, n: s.rolling(n).mean()
    ta.trend.ema_indicator = lambda s, n: s.ewm(span=n, adjust=False).mean()
    ta.volume.money_flow_index = lambda h, l, c, v, n: pd.Series(0, index=c.index)
    ta.volume.on_balance_volume = lambda c, v: v.cumsum()
    class DummyBB:
        def __init__(self, close, window, n):
            self.close = close
            self.window = window
            self.n = n
        def bollinger_hband(self):
            m = self.close.rolling(self.window).mean()
            s = self.close.rolling(self.window).std()
            return m + self.n * s
        def bollinger_lband(self):
            m = self.close.rolling(self.window).mean()
            s = self.close.rolling(self.window).std()
            return m - self.n * s
    ta.volatility.average_true_range = lambda h, l, c, n: pd.Series(0, index=c.index)
    ta.volatility.BollingerBands = DummyBB
    # load module
    loader = importlib.machinery.SourceFileLoader('model', str(path))
    spec = importlib.util.spec_from_loader('model', loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_compute_features_basic():
    model = load_model()
    dates = pd.date_range('2020-01-01', periods=250, freq='D')
    close = pd.Series(np.linspace(100, 349, len(dates)), index=dates)
    df = pd.DataFrame({
        'Open': close - 1,
        'High': close + 1,
        'Low': close - 2,
        'Close': close,
        'Volume': 1000,
    })
    market = {'SPY': close, '^VIX': pd.Series(10.0, index=dates)}

    result = model.compute_features(df, market, 5, 0.02)
    # Longest window (200‑day SMA) drops the first 199 rows
    assert len(result) == len(dates) - 199
    for col in model.FEATURES + ['Target']:
        assert col in result.columns
    assert not result.isna().any().any()
    assert (result['VIX_Level'] == 10).all()
    assert (result['SPY_Trend'] == 1).all()
    assert (result['RS_Market'].round(5) == 1).all()


def test_update_equity_tracker_appends(monkeypatch, tmp_path):
    model = load_model()
    values = [100, 102, 105]
    json_path = tmp_path / 'vals.json'
    json_path.write_text(json.dumps(values))

    appended = []

    class DummyWS:
        def __init__(self):
            self.rows = []
        def append_rows(self, rows, value_input_option=None):
            self.rows.extend(rows)
            appended.extend(rows)
        def col_values(self, idx):
            return [r[idx-1] for r in self.rows]
        def cell(self, r, c):
            class C:
                pass
            cell = C()
            cell.value = self.rows[r-1][c-1]
            return cell

    ws = DummyWS()

    class Sheet:
        def worksheet(self, title):
            return ws
        def add_worksheet(self, title, rows='100', cols='3'):
            return ws

    monkeypatch.setattr(model, 'get_sheet', lambda: Sheet())
    monkeypatch.setattr(model, '_ensure_worksheet', lambda sheet, title: ws)
    monkeypatch.setattr(model, '_last_row', lambda ws: len(ws.rows))
    monkeypatch.setattr(model, '_append_rows', lambda ws_, rows: ws_.append_rows(rows))

    class FixedDate(dt.date):
        @classmethod
        def today(cls):
            return cls(2020, 1, 10)
    monkeypatch.setattr(model.dt, 'date', FixedDate)

    model.update_equity_tracker(json_path)

    assert appended[0] == ['Date', 'Equity', 'Daily\u00a0PnL', 'Cum\u00a0PnL']
    assert appended[1:] == [
        ['2020-01-08', '100.00', '0.00', '0.00'],
        ['2020-01-09', '102.00', '2.00', '2.00'],
        ['2020-01-10', '105.00', '3.00', '5.00'],
    ]


def test_run_backtest_missing_file(tmp_path, monkeypatch):
    model = load_model()
    missing = tmp_path / 'ml_pipeline.joblib'
    monkeypatch.setattr(model, 'ARTEFACT_FILE', missing)
    with pytest.raises(FileNotFoundError):
        model.run_backtest()
