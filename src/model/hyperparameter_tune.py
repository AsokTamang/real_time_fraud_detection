import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np
from lightgbm import early_stopping
optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimize_model_cv(X_train,X_val,X_test,y_train,y_val,y_test, models,n_trials=25):
    cv = TimeSeriesSplit(n_splits=5) 
    hyperparameter_tuned_models = {}
    for model_name in models:
        def objective(trial):
            SEED = 42
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X_train)):
                
                
                X_tr, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                neg = (y_tr == 0).sum()
                pos = (y_tr == 1).sum()
                
                # --- MODEL SETUP ---
                if model_name == 'Random Forest':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 150),
                        'max_depth': trial.suggest_int('max_depth', 3, 7),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 6),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                        'class_weight': 'balanced',
                        'random_state': SEED,
                        'n_jobs': -1,
                    }
                    model = RandomForestClassifier(**params)
                    model.fit(X_tr, y_tr)

                elif model_name == 'XGBoost':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                        'scale_pos_weight': neg / pos,
                        'random_state': SEED,
                        'eval_metric': 'aucpr',
                        'device': 'cuda',
                        'tree_method': 'hist',
                        'early_stopping_rounds': 15 # ✅ Stops building trees if validation PR-AUC stops improving
                    }
                    model = XGBClassifier(**params)
                    model.fit(X_tr, y_tr, eval_set=[(X_fold_val, y_fold_val)], verbose=False)

                elif model_name == 'LightGBM':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 7),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'min_child_samples': trial.suggest_int('min_child_samples', 2, 20),
                        'num_leaves': trial.suggest_int('num_leaves', 20, 50),
                        'scale_pos_weight': neg / pos,
                        'random_state': SEED,
                        'n_jobs':-1
                    }
                    model = LGBMClassifier(**params)
                    
                    model.fit(X_tr, y_tr, eval_set=[(X_fold_val, y_fold_val)], callbacks=[early_stopping(stopping_rounds=15, verbose=False)])

                # --- EVALUATION ---
                y_pred_proba = model.predict_proba(X_fold_val)[:, 1] 
                score = average_precision_score(y_fold_val, y_pred_proba)
                fold_scores.append(score)
                
                # --- OPTUNA PRUNING ---
                trial.report(np.mean(fold_scores), fold)
                
                # If the score is terrible compared to previous trials, we kill it now.
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(fold_scores)

        # Adding MedianPruner to study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1) # ✅ Activates Pruning
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=1 
        )
        hyperparameter_tuned_models[model_name] = study.best_params  #storing the model with their corresponding best hyperparameter
    return X_train,X_val,X_test,y_train,y_val,y_test,hyperparameter_tuned_models

 
        