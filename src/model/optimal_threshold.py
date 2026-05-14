from dataclasses import dataclass
import os

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

@dataclass
class ModelTrainerConfig:
    trained_model_path:str = os.path.join('artifacts','model.pkl')




class initiate_threshold_tuning:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig().trained_model_path  #storing the path where we save the trained model

    def model_with_optimal_threshold(self,X_train,X_val,X_test,y_train,y_val,y_test,hyperparameter_tuned_models):
        thresholds = {}
        fitted_model = {}
        for name, best_params in hyperparameter_tuned_models.items():
            if name == 'Random Forest':
                model = RandomForestClassifier(**best_params)
            elif name == 'XGBoost': 
                model = XGBClassifier(**best_params)
            else:
                model = LGBMClassifier(**best_params)       

            model.fit(X_train, y_train)  #fitting the model on the training data
            y_prob = model.predict_proba(X_val)[:, 1]  #only extracting the probability of the positive class (fraud)
                
            fitted_model[name] = model   
            print(f'saved the trained model {name}')
            
            precisions, recalls, thresh = precision_recall_curve(y_val, y_prob)
            fbeta = (1 + 4) * (precisions * recalls) / (4 * precisions + recalls + 1e-8)
            best_idx = fbeta.argmax()
            thresholds[name] = thresh[best_idx]
            print(f"Best threshold: {thresh[best_idx]:.4f}")


        #multimodel comparison with best threshold
        result = {}
        for name,model in fitted_model.items():
            y_prob = model.predict_proba(X_val)[:, 1]  #only extracting the probability of the positive class (fraud)
            y_pred = (y_prob>=thresholds[name]).astype(int)    #then based on the optimal threshold, we are finding the prediction done by each trained model
            
            result[name] = {
                'F1': round(f1_score(y_val, y_pred), 4),
                'ROC-AUC': round(roc_auc_score(y_val, y_prob), 4),
                'PR-AUC': round(average_precision_score(y_val, y_prob), 4),
                'Recall': round(recall_score(y_val, y_pred), 4),
                'Precision': round(precision_score(y_val, y_pred), 4)
            }

