import importlib
import importlib.util
import importlib.machinery
import types
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Utility to load grid_search with heavy dependencies stubbed
# ---------------------------------------------------------------------------

def load_grid_search():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "model" / "grid_search.py"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    modules = [
        "optuna",
        "sklearn",
        "sklearn.ensemble",
        "sklearn.metrics",
        "xgboost",
        "xgboost.core",
        "lightgbm",
    ]
    for name in modules:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    # Dummy estimators -----------------------------------------------------
    class DummyEstimator:
        def __init__(self, *a, **k):
            pass

        def fit(self, *a, **k):
            return self

        def predict(self, X):
            return [0] * len(X)

    sk_ensemble = sys.modules["sklearn.ensemble"]
    sk_ensemble.RandomForestClassifier = DummyEstimator
    sk_ensemble.VotingClassifier = DummyEstimator

    sk_metrics = sys.modules["sklearn.metrics"]
    sk_metrics.precision_score = lambda *a, **k: 1.0

    xgb_mod = sys.modules["xgboost"]
    xgb_mod.core = types.ModuleType("xgboost.core")
    xgb_mod.core.XGBoostError = Exception
    xgb_mod.XGBClassifier = DummyEstimator

    lgbm_mod = sys.modules["lightgbm"]
    lgbm_mod.LGBMClassifier = DummyEstimator
    lgbm_mod.early_stopping = lambda *a, **k: None

    optuna_mod = sys.modules["optuna"]

    class DummyBestTrial:
        params = {"a": 1}

    class DummyStudy:
        def __init__(self):
            self.best_trial = DummyBestTrial()

        def optimize(self, fn, n_trials=None, timeout=None):
            pass

        def trials_dataframe(self):
            return pd.DataFrame({"value": [1]})

    optuna_mod.create_study = lambda direction=None: DummyStudy()

    # Avoid requiring pyarrow
    pd.DataFrame.to_parquet = lambda self, *a, **k: None

    loader = importlib.machinery.SourceFileLoader("model.grid_search", str(path))
    spec = importlib.util.spec_from_loader("model.grid_search", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_run_grid_search_basic():
    gs = load_grid_search()
    X = pd.DataFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7]})
    y = pd.Series([0, 1, 0, 1])
    params, model = gs.run_grid_search(X, y, n_trials=1)

    assert isinstance(params, dict)
    assert hasattr(model, "fit")
    assert len((params, model)) == 2

