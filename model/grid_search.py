from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import precision_score
import optuna
import xgboost as xgb
import lightgbm as lgbm


def _gpu_available() -> bool:
    """Return ``True`` if either XGBoost or LightGBM can access a GPU."""
    sample_X = [[0], [1]]
    sample_y = [0, 1]
    try:
        xgb.XGBClassifier(
            tree_method="gpu_hist", gpu_id=0, n_estimators=1
        ).fit(sample_X, sample_y)
        return True
    except Exception:
        pass
    try:
        lgbm.LGBMClassifier(device_type="gpu", n_estimators=1).fit(sample_X, sample_y)
        return True
    except Exception:
        return False

__all__ = ["run_grid_search"]


def run_grid_search(X_train_sel, y_train, n_trials: int = 50):
    """Run Optuna Bayesian search and print results.

    Parameters
    ----------
    X_train_sel : DataFrame
        Training feature matrix.
    y_train : Series
        Training labels.
    n_trials : int, optional
        Number of Optuna trials to run, by default ``50``.

    Returns
    -------
    tuple
        ``(best_params, best_model)`` where ``best_params`` are the hyper
        parameters from ``study.best_trial`` and ``best_model`` is the
        fitted :class:`VotingClassifier` using those parameters.
    """

    gpu = _gpu_available()

    def objective(trial):
        split = int(len(X_train_sel) * 0.8)
        X_tr, X_val = X_train_sel.iloc[:split], X_train_sel.iloc[split:]
        y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]

        rf = RandomForestClassifier(
            n_estimators=trial.suggest_int("n", 100, 500, 100),
            max_depth=trial.suggest_int("d", 5, 20),
            min_samples_leaf=trial.suggest_int("leaf", 1, 4),
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )

        params = dict(
            max_depth=6,
            n_estimators=trial.suggest_int("xgb_n", 100, 500, 50),
            learning_rate=trial.suggest_float("xgb_lr", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("xgb_subsample", 0.5, 1.0),
            random_state=42,
            eval_metric="logloss",
            early_stopping_rounds=20,
        )
        xgbc = xgb.XGBClassifier(
            tree_method="gpu_hist" if gpu else "hist",
            gpu_id=0 if gpu else None,
            **params,
        )
        xgbc.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        lgbc = lgbm.LGBMClassifier(
            device_type="gpu" if gpu else "cpu",
            n_estimators=trial.suggest_int("lgb_n", 100, 500, 50),
            learning_rate=trial.suggest_float("lgb_lr", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("lgb_subsample", 0.5, 1.0),
            random_state=42,
        )

        lgbc.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgbm.early_stopping(20)],
        )

        vote = VotingClassifier([("rf", rf), ("xgb", xgbc), ("lgb", lgbc)], voting="soft", n_jobs=-1)
        try:
            vote.fit(X_tr, y_tr)
            pred = vote.predict(X_val)
            score = precision_score(y_val, pred, pos_label=1, zero_division=0)
        except Exception as e:
            print(f"Trial failed due to {e}; returning 0")
            score = 0.0
        return score

    print(f"⏳  Running Optuna grid search ({n_trials} trials, CPU/GPU‑safe) …")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=600)

    results = study.trials_dataframe().sort_values("value", ascending=False)
    print("\n===== GridSearch Results (sorted by precision) =====")
    print(results[["value"] + [c for c in results.columns if c.startswith("params_")]].to_string(index=False))

    results.to_parquet("grid_search_results.parquet", index=False)
    print("\nGrid search completed. Results saved to grid_search_results.parquet")

    best_params = study.best_trial.params

    rf = RandomForestClassifier(
        n_estimators=best_params.get("n"),
        max_depth=best_params.get("d"),
        min_samples_leaf=best_params.get("leaf"),
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    xgb_params = dict(
        max_depth=6,
        n_estimators=best_params.get("xgb_n"),
        learning_rate=best_params.get("xgb_lr"),
        subsample=best_params.get("xgb_subsample"),
        random_state=42,
        eval_metric="logloss",
    )
    xgbc = xgb.XGBClassifier(
        tree_method="gpu_hist" if gpu else "hist",
        gpu_id=0 if gpu else None,
        **xgb_params,
    )

    lgbc = lgbm.LGBMClassifier(
        device_type="gpu" if gpu else "cpu",
        n_estimators=best_params.get("lgb_n"),
        learning_rate=best_params.get("lgb_lr"),
        subsample=best_params.get("lgb_subsample"),
        random_state=42,
    )

    rf.fit(X_train_sel, y_train)

    trained_estimators = [("rf", rf)]

    try:
        xgbc.fit(X_train_sel, y_train)
        trained_estimators.append(("xgb", xgbc))
    except xgb.core.XGBoostError as e:
        print(f"XGBoost failed due to {e}; falling back to CPU-only model")

    try:
        lgbc.fit(X_train_sel, y_train)
        trained_estimators.append(("lgb", lgbc))
    except Exception as e:
        print(f"LightGBM failed due to {e}; falling back to CPU-only model")

    best_model = VotingClassifier(trained_estimators, voting="soft", n_jobs=-1)
    best_model.fit(X_train_sel, y_train)

    return best_params, best_model
