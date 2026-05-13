from lightgbm import LGBMClassifier
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
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


def optimize_model_cv(X_train, y_train, X_test, models,n_trials=5):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    best_overall_model= None
    best_overall_score  = -1
    best_overall_preds  = 0
    best_overall_name   = None

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
                    'random_state'     : 42,
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
                    'scale_pos_weight': len(y[y==0]) / len(y[y==1]),
                    'random_state'    : 42,
                    'eval_metric'     : 'aucpr',
                    'device'          : 'cuda',
                    'tree_method'     : 'hist'
                }
                model = XGBClassifier(**params)

            elif model_name == 'LightGBM':
                params = {
                    'n_estimators'     : trial.suggest_int('n_estimators', 100, 300),
                    'max_depth'        : trial.suggest_int('max_depth', 3, 10),
                    'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample'        : trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                    'scale_pos_weight' : len(y[y==0]) / len(y[y==1]),
                    'random_state'     : 42,
                    'device'           : 'gpu',
                }
                model = LGBMClassifier(**params)

            n_jobs = -1 if model_name == 'Random Forest' else 1
            scores = cross_val_score(
                model, X_train, y_train,
                cv=cv,
                scoring='average_precision',
                n_jobs=n_jobs
            )
            return scores.mean()

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)     # ✅ reproducible
        )
        early_stop = EarlyStoppingCallback(patience=10)

        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=1,                       # ✅ sequential for GPU safety
            callbacks=[early_stop]
        )

        print(f'{model_name} — Best CV PR-AUC: {study.best_value:.4f}')
        print(f'Best Params: {study.best_params}')
        best_params = study.best_params
        model_map = {
            'Random Forest': RandomForestClassifier(**best_params, class_weight='balanced', random_state=42, n_jobs=-1),
            'XGBoost'      : XGBClassifier(**best_params, random_state=42, eval_metric='aucpr', device='cuda', tree_method='hist'),
            'LightGBM'     : LGBMClassifier(**best_params, random_state=42, device='gpu'),
        }
        final_model = model_map[model_name]
        final_model.fit(X_train, y_train)
        preds = final_model.predict(X_test)

        if study.best_value > best_overall_score:
            best_overall_score  = study.best_value  #storing the latest best cross_validation score
            best_overall_model  = final_model
            best_overall_preds  = preds
            best_overall_name   = model_name

    print(f'\n✅ Best Overall Model : {best_overall_name}')
    print(f'✅ Best CV PR-AUC     : {best_overall_score:.4f}')
    return best_overall_model, best_overall_score, best_overall_preds