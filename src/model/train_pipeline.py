import os
import sys
from dataclasses import dataclass
from lightgbm import LGBMClassifier
from src.utils import save_object
from src.exception import CustomError
from src.logger import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from src.model.hyperparameter_tune import optimize_model_cv


@dataclass
class ModelTrainerConfig:
    trained_model_path:str = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
          self.model_trainer_config = ModelTrainerConfig()

    def train_model(self,X_train,X_val,X_test,y_train,y_val,y_test):
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        ratio = neg/pos
        models = {
        'LightGBM': LGBMClassifier(verbose=-1, class_weight='balanced',
                                        n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(scale_pos_weight=ratio,    #here we are using scale_pos_weight to handle the class imbalance in the dataset by balanced weight  to both classes based on their ratio, which helps the model to effectively classifying the fraudulent transactions.
                                    n_estimators=100, random_state=42, eval_metric='aucpr'),
        'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=100,
                                                    random_state=42, n_jobs=-1)
        }
        hyperparameter_tuned_models= optimize_model_cv(X_train,X_val,X_test,y_train,y_val,y_test, models,n_trials=25)
        
        