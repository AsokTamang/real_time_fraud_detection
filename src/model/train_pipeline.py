import os
from dataclasses import dataclass
import sys
from lightgbm import LGBMClassifier
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.data.validate_data import validate_data
from src.features.feature_engineering import build_features
from src.model.evaluate_model import model_evaluation
from src.transformer.save_processor import Datascalar
from src.utils import save_object
from src.exception import CustomError
from src.logger import logging
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.model.hyperparameter_tune import optimize_model_cv
from src.model.optimal_threshold import initiate_threshold_tuning

@dataclass
class ModelTrainerConfig:
    trained_model_path:str = os.path.join('artifacts','model.pkl')




class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig().trained_model_path  #storing the path where we save the trained model
    def train_model(self,X_train,X_val,X_test,y_train,y_val,y_test):
        try:
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
            #hyperparamter tuning
            X_train,X_val,X_test,y_train,y_val,y_test,hyperparameter_tuned_models= optimize_model_cv(X_train,X_val,X_test,y_train,y_val,y_test, models,n_trials=25)
            logging.info('hyperparameter tuned')
            #optimal threshold tuning
            fitted_model,thresholds = initiate_threshold_tuning.model_with_optimal_threshold(X_train,X_val,X_test,y_train,y_val,y_test,hyperparameter_tuned_models)
            #model evaluation
            best_model_info = model_evaluation(fitted_model,thresholds,X_train, X_val, X_test, y_train, y_val, y_test):
            save_object(self.model_trainer_config,best_model_info)
            logging.info('Best model with corresponding optimal threshold saved')
            return best_model_info
            


        except Exception as e:
            raise CustomError(e,sys)    

        
if __name__ =='__main__':
    datascalar = Datascalar()
    mt = ModelTrainer()
    df = load_data('data/pay_sim.csv')  #loading the data
    print('1. Data loaded')
    preprocessed_df = preprocess_data(df)  #preprocessing the data
    print('2. Preprocessed data')
    print(validate_data(preprocessed_df))  #validation the data
    print('3. Validated data')
    engineered_df = build_features(preprocessed_df)  #applied the feature engineering on preprocessed data
    print('4. Applied feature engineering')
    X_train,X_val,X_test,y_train,y_val,y_test=datascalar.split_data(engineered_df)
    print('5. Training,CV and Test dataset splitted.')
    mt.train_model(X_train,X_val,X_test,y_train,y_val,y_test)  #training the model
    logging.info('Best model info saved as pickle file.')


