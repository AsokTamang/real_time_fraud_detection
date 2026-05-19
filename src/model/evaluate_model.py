# multimodel comparison with best threshold
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def model_evaluation(fitted_model,thresholds,X_train, X_val, X_test, y_train, y_val, y_test):
        result = {}
        for name, model in fitted_model.items():
            y_prob = model.predict_proba(X_test)[
                :, 1
            ]  # only extracting the probability of the positive class (fraud)
            y_pred = (y_prob >= thresholds[name]).astype(
                int
            )  # then based on the optimal threshold, we are finding the prediction done by each trained model

            result[name] = {
                "F1": round(f1_score(y_test, y_pred), 4),
                "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
                "PR-AUC": round(average_precision_score(y_test, y_prob), 4),
                "Recall": round(recall_score(y_test, y_pred), 4),
                "Precision": round(precision_score(y_test, y_pred), 4),
            }
        # Both metrics matter — use weighted average
        best_model_name = max(
            result,
            key=lambda name: 0.4 * result[name]["F1"] + 0.6 * result[name]["PR-AUC"],
        )

        best_model = fitted_model[best_model_name]  # trained model
        best_threshold = thresholds[best_model_name]

        print(f"Best Model  : {best_model_name}")
        print(f"F1          : {result[best_model_name]['F1']}")
        print(f"PR-AUC      : {result[best_model_name]['PR-AUC']}")
        print(f"Threshold   : {best_threshold:.4f}")
        best_model_info = {
            "model": best_model,
            "threshold": best_threshold,
            "model_name": best_model_name,
            "metrics": result[best_model_name],
        }
        return best_model_info