from lightgbm import LGBMClassifier
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBClassifier
optuna.logging.set_verbosity(optuna.logging.WARNING)  # ✅ cleaner output



#this class is for stopping the hyperparameter tuning when the cross_validation score stops converging
class EarlyStoppingCallback:
    def __init__(self, patience):
        self.patience = patience
        self.best_value = None
        self.stagnant_trials = 0

    def __call__(self, study, trial):
        if self.best_value is None or study.best_value > self.best_value:
            self.best_value = study.best_value
            self.stagnant_trials = 0
        else:
            self.stagnant_trials += 1
        if self.stagnant_trials >= self.patience:
            print(f"Stopping study: No improvement for {self.patience} trials.")
            study.stop()


def optimize_model_cv(X_train,X_val,X_test,y_train,y_val,y_test, models,n_trials=25):
    SEED =42
    cv = TimeSeriesSplit(n_splits=5, shuffle=True, random_state=SEED)
    results = {}

    for model_name in models:

        def objective(trial):
            if model_name == 'Random Forest':
                params = {
                    'n_estimators'     : trial.suggest_int('n_estimators', 50, 150),
                    'max_depth'        : trial.suggest_int('max_depth', 3, 7),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 6),
                    'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 5),
                    'max_features'     : trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                    'class_weight'     : 'balanced',
                    'random_state'     : SEED,
                    'n_jobs'           : -1,    
                }
                model = RandomForestClassifier(**params)

            elif model_name == 'XGBoost':
                params = {
                    'n_estimators'    : trial.suggest_int('n_estimators', 100, 300),
                    'max_depth'       : trial.suggest_int('max_depth', 3, 10),
                    'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample'       : trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'scale_pos_weight': len(y_train[y_train==0]) / len(y_train[y_train==1]),
                    'random_state'    : SEED,
                    'eval_metric'     : 'aucpr',
                    'device'          : 'cuda',
                    'tree_method'     : 'hist'
                }
                model = XGBClassifier(**params)
            
            elif model_name == 'LightGBM':
                max_depth = trial.suggest_categorical('max_depth', [3, 5, 7])
                params = {
                'n_estimators'     : trial.suggest_categorical('n_estimators', [100, 200, 300]),
                'max_depth'        : max_depth,
                'learning_rate'    : trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1, 0.2]),
                'subsample'        : trial.suggest_categorical('subsample', [0.6, 0.8, 1.0]),
                'colsample_bytree' : trial.suggest_categorical('colsample_bytree', [0.6, 0.8, 1.0]),
                'num_leaves'       : trial.suggest_int('num_leaves', 2, min(2**max_depth, 128)),
                'min_child_samples': trial.suggest_int('min_child_samples', 2, 20),
                'min_child_weight' : 0.001, # Helps prevent the 'left_count > 0' error
                'scale_pos_weight' : len(y_train[y_train==0]) / len(y_train[y_train==1]),
                'random_state'     : SEED,
                'device'           : 'cpu', # Changed to CPU for stability
                'verbose'          : -1,
                }
                model = LGBMClassifier(**params)

            n_jobs = -1 if model_name in ['Random Forest','LightGBM'] else 1
            
            scores = cross_val_score(
                model, X_train, y_train,
                cv=cv,
                scoring='average_precision',
                n_jobs=n_jobs
            )
            return scores.mean()

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),

        )
        early_stop = EarlyStoppingCallback(patience=10)

        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=1,                      
            callbacks=[early_stop]
        )
        results[model_name] = study.best_params  #here we are storing the best parameter of the model with their best parameters
    return X_train,X_val,X_test,y_train,y_val,y_test,results







        