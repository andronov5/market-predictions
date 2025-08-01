from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import precision_score
import optuna
import xgboost as xgb
import lightgbm as lgbm

__all__ = ["run_grid_search"]


def run_grid_search(X_train_sel, y_train, n_trials: int = 50):
    """Run Optuna Bayesian search and print results."""

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
        try:
            xgbc = xgb.XGBClassifier(tree_method="gpu_hist", gpu_id=0, **params)
            xgbc.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        except xgb.core.XGBoostError:
            xgbc = xgb.XGBClassifier(tree_method="hist", **params)
            xgbc.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        try:
            lgbc = lgbm.LGBMClassifier(
                device_type="gpu",
                n_estimators=trial.suggest_int("lgb_n", 100, 500, 50),
                learning_rate=trial.suggest_float("lgb_lr", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("lgb_subsample", 0.5, 1.0),
                random_state=42,
            )
        except Exception:
            lgbc = lgbm.LGBMClassifier(
                device_type="cpu",
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
