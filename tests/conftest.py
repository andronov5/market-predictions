import sys
import types


def pytest_configure(config):
    heavy_modules = [
        'yfinance', 'joblib', 'lightgbm', 'xgboost', 'xgboost.core', 'optuna',
        'sklearn', 'sklearn.ensemble', 'sklearn.metrics', 'sklearn.model_selection',
        'vectorbt', 'matplotlib', 'matplotlib.pyplot',
        'ta', 'ta.momentum', 'ta.trend', 'ta.volatility', 'ta.volume'
    ]
    for name in heavy_modules:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    joblib = sys.modules['joblib']
    joblib.Parallel = lambda *a, **k: []
    joblib.delayed = lambda f: f
    sk_ensemble = sys.modules['sklearn.ensemble']
    sk_ensemble.RandomForestClassifier = object
    sk_ensemble.VotingClassifier = object
    sk_metrics = sys.modules['sklearn.metrics']
    sk_metrics.precision_score = lambda *a, **k: 0

    ta_vol = sys.modules['ta.volatility']
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

    ta_vol.BollingerBands = DummyBB


