from src.utils import save_object
import mlflow
import mlflow.sklearn
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


class initiate_threshold_tuning:
    def model_with_optimal_threshold(
        X_train, X_val, X_test, y_train, y_val, y_test, hyperparameter_tuned_models
    ):
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()

        thresholds = {}
        fitted_model = {}

        SEED = 42

        rf_fixed_params = {
            "random_state": SEED,
            "n_jobs": -1,
            "class_weight": "balanced",
        }

        XG_fixed_params = {
            "scale_pos_weight": neg / pos,
            "random_state": SEED,
            "eval_metric": "aucpr",
            "device": "cpu",
            "tree_method": "hist",
            "early_stopping_rounds": 15,
        }

        LGM_fixed_params = {
            "scale_pos_weight": neg / pos,
            "random_state": SEED,
            "n_jobs": -1,
        }

        mlflow.set_experiment("threshold_tuning_experiment")

        for name, best_params in hyperparameter_tuned_models.items():

            with mlflow.start_run(run_name=name):

                # -------------------------
                # Model selection
                # -------------------------
                if name == "Random Forest":
                    model = RandomForestClassifier(**best_params, **rf_fixed_params)
                elif name == "XGBoost":
                    model = XGBClassifier(**best_params, **XG_fixed_params)
                else:
                    model = LGBMClassifier(**best_params, **LGM_fixed_params)

                # -------------------------
                # Training model
                # -------------------------
                model.fit(X_train, y_train)
                y_prob = model.predict_proba(X_val)[:, 1]

                fitted_model[name] = model
                print(f"trained model {name}")

                # -------------------------
                # Metrics
                # -------------------------
                pr_auc = average_precision_score(y_val, y_prob)
                roc_auc = roc_auc_score(y_val, y_prob)

                precisions, recalls, thresh = precision_recall_curve(y_val, y_prob)

                fbeta = (1 + 4) * (precisions * recalls) / (4 * precisions + recalls + 1e-8)
                best_idx = fbeta.argmax()
                best_threshold = thresh[best_idx]

                thresholds[name] = best_threshold

                # apply threshold
                y_pred = (y_prob >= best_threshold).astype(int)

                f1 = f1_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred)
                recall = recall_score(y_val, y_pred)

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

                # loggin model
                if name == 'XGBoost':
                    mlflow.xgboost.log_model(model,artifact_path="model")
                else:    
                    mlflow.sklearn.log_model(model, artifact_path="model")

                print(f"best threshold for {name}: {best_threshold:.4f}")

        return fitted_model, thresholds