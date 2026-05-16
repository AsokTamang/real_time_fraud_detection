import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier, early_stopping
import numpy as np
import mlflow

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42


def optimize_model_cv(
    X_train, X_val, X_test, y_train, y_val, y_test, models, n_trials=25
):
    cv = TimeSeriesSplit(n_splits=5)
    hyperparameter_tuned_models = {}

    # convert to pandas before CV splitting
    X_train_pd = X_train.to_pandas() if hasattr(X_train, 'to_pandas') else X_train
    y_train_pd = y_train.to_pandas() if hasattr(y_train, 'to_pandas') else y_train

    neg = (y_train_pd == 0).sum()
    pos = (y_train_pd == 1).sum()
    scale_pos_weight = neg / pos

    mlflow.set_experiment("Optuna Hyperparameter Tuning")

    for model_name in models:
        with mlflow.start_run(run_name=model_name):

            def objective(trial, model_name=model_name):
                fold_scores = []

                for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_pd)):
                    X_tr = X_train_pd.iloc[train_idx]
                    X_fold_val = X_train_pd.iloc[val_idx]
                    y_tr = y_train_pd.iloc[train_idx]
                    y_fold_val = y_train_pd.iloc[val_idx]

                    if model_name == "Random Forest":
                        params = {
                            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
                            "max_depth": trial.suggest_int("max_depth", 3, 7),
                            "subsample": trial.suggest_float("subsample", 0.7, 1.0),        #  keeping high for time series
                            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.3, 1.0),  #  like max_features
                            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                            # fixed params
                            "scale_pos_weight": scale_pos_weight,
                            "random_state": SEED,
                            "device": "cuda",
                            "tree_method": "hist",
                        }
                        model = XGBRFClassifier(**params)
                        model.fit(X_tr, y_tr)

                    elif model_name == "XGBoost":
                        params = {
                            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                            "max_depth": trial.suggest_int("max_depth", 3, 10),
                            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                            # fixed params
                            "scale_pos_weight": scale_pos_weight,
                            "random_state": SEED,
                            "eval_metric": "aucpr",
                            "device": "cuda",
                            "tree_method": "hist",
                            "early_stopping_rounds": 15,
                        }
                        model = XGBClassifier(**params)
                        model.fit(
                            X_tr, y_tr,
                            eval_set=[(X_fold_val, y_fold_val)],
                            verbose=False,
                        )

                    elif model_name == "LightGBM":
                        params = {
                            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                            "max_depth": trial.suggest_int("max_depth", 3, 7),
                            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                            "min_child_samples": trial.suggest_int("min_child_samples", 2, 20),
                            "num_leaves": trial.suggest_int("num_leaves", 20, 50),
                            # fixed params
                            "scale_pos_weight": scale_pos_weight,
                            "random_state": SEED,
                            "device": "gpu",
                        }
                        model = LGBMClassifier(**params)
                        model.fit(
                            X_tr, y_tr,
                            eval_set=[(X_fold_val, y_fold_val)],
                            callbacks=[early_stopping(stopping_rounds=15, verbose=False)],
                        )

                    y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
                    score = average_precision_score(y_fold_val, y_pred_proba)
                    fold_scores.append(score)

                    trial.report(np.mean(fold_scores), fold)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                return np.mean(fold_scores)

            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=SEED),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
            )
            study.optimize(objective, n_trials=n_trials, n_jobs=1)

            hyperparameter_tuned_models[model_name] = study.best_params
            print(f"{model_name} | Best PR-AUC: {study.best_value:.4f} | Params: {study.best_params}")

            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("best_pr_auc", study.best_value)
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_dict(study.best_params, "best_params.json")

    return X_train, X_val, X_test, y_train, y_val, y_test, hyperparameter_tuned_models