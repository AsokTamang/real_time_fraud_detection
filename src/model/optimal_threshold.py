from src.utils import save_object
import mlflow
import mlflow.sklearn
import numpy as np
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier, XGBRFClassifier


class initiate_threshold_tuning:
    def model_with_optimal_threshold(
        X_train, X_val, X_test, y_train, y_val, y_test, hyperparameter_tuned_models
    ):
        # convert to pandas to be safe
        X_train_pd = X_train.to_pandas() if hasattr(X_train, 'to_pandas') else X_train
        X_val_pd = X_val.to_pandas() if hasattr(X_val, 'to_pandas') else X_val
        y_train_pd = y_train.to_pandas() if hasattr(y_train, 'to_pandas') else y_train
        y_val_pd = y_val.to_pandas() if hasattr(y_val, 'to_pandas') else y_val

        neg = (y_train_pd == 0).sum()
        pos = (y_train_pd == 1).sum()
        scale_pos_weight = neg / pos

        thresholds = {}
        fitted_model = {}
        SEED = 42

        # XGBRFClassifier fixed params
        RF_fixed_params = {
            "scale_pos_weight": scale_pos_weight,
            "random_state": SEED,
            "device": "cuda",
            "tree_method": "hist"
        }

        XG_fixed_params = {
            "scale_pos_weight": scale_pos_weight,
            "random_state": SEED,
            "eval_metric": "aucpr",
            "device": "cuda",
            "tree_method": "hist",
            "early_stopping_rounds": 15,
        }

        LGM_fixed_params = {
            "scale_pos_weight": scale_pos_weight,
            "random_state": SEED,
            "n_jobs":-1
        }

        mlflow.set_experiment("threshold_tuning_experiment")

        for name, best_params in hyperparameter_tuned_models.items():
            with mlflow.start_run(run_name=name):

                # -------------------------
                # Model selection
                # -------------------------
                if name == "Random Forest":
                    #filtering only XGBRFClassifier supported params from optuna
                    supported_rf_params = {
                        k: v for k, v in best_params.items()
                        if k in ["n_estimators", "max_depth", "subsample",
                                 "colsample_bynode", "min_child_weight"]
                    }
                    model = XGBRFClassifier(**supported_rf_params, **RF_fixed_params)

                elif name == "XGBoost":
                    model = XGBClassifier(**best_params, **XG_fixed_params)

                else:
                    model = LGBMClassifier(**best_params, **LGM_fixed_params)

                # -------------------------
                # Training
                # -------------------------
                if name == "Random Forest":
                    model.fit(X_train_pd, y_train_pd)  

                elif name == "XGBoost":
                    model.fit(
                        X_train_pd, y_train_pd,
                        eval_set=[(X_val_pd, y_val_pd)],
                        verbose=False
                    )

                else:  # LightGBM
                    model.fit(
                        X_train_pd, y_train_pd,
                        eval_set=[(X_val_pd, y_val_pd)],
                        callbacks=[early_stopping(stopping_rounds=15, verbose=False)]
                    )

                y_prob = model.predict_proba(X_val_pd)[:, 1]

                fitted_model[name] = model
                print(f"trained model {name}")

                # -------------------------
                # Metrics
                # -------------------------
                #evaluation on cross_validation dataset
                pr_auc = average_precision_score(y_val_pd, y_prob)
                roc_auc = roc_auc_score(y_val_pd, y_prob)

                precisions, recalls, thresh = precision_recall_curve(y_val_pd, y_prob)
                fbeta = (1 + 4) * (precisions * recalls) / (4 * precisions + recalls + 1e-8)
                best_idx = fbeta.argmax()
                best_threshold = thresh[best_idx]
                thresholds[name] = best_threshold

                y_pred = (y_prob >= best_threshold).astype(int)
                f1 = f1_score(y_val_pd, y_pred)
                precision = precision_score(y_val_pd, y_pred)
                recall = recall_score(y_val_pd, y_pred)

                # -------------------------
                # MLflow logging
                # -------------------------
                mlflow.log_params(best_params)
                mlflow.log_param("model_name", name)
                mlflow.log_param("best_threshold", best_threshold)
                mlflow.log_metric("pr_auc", pr_auc)
                mlflow.log_metric("roc_auc", roc_auc)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)

                if name == "LightGBM":
                    mlflow.lightgbm.log_model(model, artifact_path="model")
                else:
                    # both XGBoost and Random Forest use xgboost logger
                    mlflow.xgboost.log_model(model, artifact_path="model")

                print(f"best threshold for {name}: {best_threshold:.4f}")

        return fitted_model, thresholds