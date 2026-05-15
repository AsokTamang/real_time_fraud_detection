from src.utils import save_object

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

        for name, best_params in hyperparameter_tuned_models.items():
            if name == "Random Forest":
                model = RandomForestClassifier(**best_params, **rf_fixed_params)
            elif name == "XGBoost":
                model = XGBClassifier(**best_params, **XG_fixed_params)
            else:
                model = LGBMClassifier(**best_params, **LGM_fixed_params)

            model.fit(X_train, y_train)  # fitting the model on the training data
            y_prob = model.predict_proba(X_val)[
                :, 1
            ]  # only extracting the probability of the positive class (fraud)

            fitted_model[name] = model
            print(f"saved the trained model {name}")

            precisions, recalls, thresh = precision_recall_curve(y_val, y_prob)
            fbeta = (1 + 4) * (precisions * recalls) / (4 * precisions + recalls + 1e-8)
            best_idx = fbeta.argmax()
            thresholds[name] = thresh[best_idx]
            print(f"Best threshold: {thresh[best_idx]:.4f}")

        return fitted_model,thresholds
