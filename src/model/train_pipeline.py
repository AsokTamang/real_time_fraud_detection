from lightgbm import LGBMClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                             average_precision_score, precision_recall_curve,
                             recall_score, precision_score, f1_score)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from tune_pipeline import optimize_model_cv


def train_model(X_train,X_test,y_train,y_test):
    models = {
    'LightGBM': LGBMClassifier(verbose=-1, class_weight='balanced',
                                    n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),    #here we are using scale_pos_weight to handle the class imbalance in the dataset by balanced weight  to both classes based on their ratio, which helps the model to effectively classifying the fraudulent transactions.
                                n_estimators=100, random_state=42, eval_metric='aucpr'),
    'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=100,
                                                random_state=42, n_jobs=-1)
    }
    best_model,best_model_score,test_prediction = optimize_model_cv(X_train, y_train, X_test, models,n_trials=5)
        