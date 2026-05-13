from lightgbm import LGBMClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                             average_precision_score, precision_recall_curve,
                             recall_score, precision_score, f1_score)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def train_model(X_train,X_test,y_train,y_test):
    models = {
    'LightGBM': LGBMClassifier(verbose=-1, class_weight='balanced',
                                    n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),    #here we are using scale_pos_weight to handle the class imbalance in the dataset by balanced weight  to both classes based on their ratio, which helps the model to effectively classifying the fraudulent transactions.
                                n_estimators=100, random_state=42, eval_metric='aucpr'),
    'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=100,
                                                random_state=42, n_jobs=-1),
    'Logistic Regression': LogisticRegression(class_weight='balanced',
                                                max_iter=1000, random_state=42)
    }
    result = {}
    for name,model in models.items():
        model.fit(X_train, y_train)  #fitting the model on the training data
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  #only extracting the probability of the positive class (fraud)
        result[name] = {
            'F1': round(f1_score(y_test, y_pred), 4),
            'ROC-AUC': round(roc_auc_score(y_test, y_prob), 4),
            'PR-AUC': round(average_precision_score(y_test, y_prob), 4),
            'Recall': round(recall_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred), 4),
            'classification_report':classification_report(y_test,y_pred, target_names=['Legit','Fraud'])
        }
        
    df = pd.DataFrame(result).T