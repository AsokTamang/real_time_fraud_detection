import os
from src.exception import CustomError
from src.logger import logging
import sys
import numpy as np
from src.utils import load_object


class Feature_engineering:
    def __init__(self):
        self.preprocessors = load_object(
            os.path.join("artifacts", "preprocessors.pkl")
        )  # loading the preprocessor object which is used for data transformation

    def feature_engineering(self, df):
        try:
            df = df.sort_values("step").reset_index(
                drop=True
            )  # sorting the dataframe based on time step
            df['day'] = np.ceil(df['step'] / 24).astype(int)  #converting the time step into days by dividing the step by 24 and taking the ceiling to get the day number
            df['day_name'] = df['day'].map({1:'Monday',2:'Tuesday',3:'Wednesday',4:'Thursday',5:'Friday',6:'Saturday',7:'Sunday'})  #mapping the day number to the day name
            df['hour_of_day']=df['step']%24
            df['dest_type'] = np.where(df['namedest'].str.startswith('M'), 'Merchant', 'Customer')
            df['is_transfer'] = (df['type'] == 'TRANSFER').astype(int)  #creating a new feature is_transfer which indicates whether the transaction type is a transfer or not
            df['is_cash_out'] = (df['type'] == 'CASH_OUT').astype(int)  #creating a new feature is_cash_out which indicates whether the transaction type is a cash out or not   
            df['is_merchant_dest'] = (df['dest_type'] == 'Merchant').astype(int)  #creating a new feature is_merchant_dest which indicates whether the destination account is a merchant or not based on the dest_type column   
            df['log_amount'] = np.log1p(df['amount'])  #creating a new feature log_amount which is the logarithm of the transaction amount to reduce the skewness of the distribution
            df['critical_transaction'] = (df['amount'] > 200000).astype(int)  #creating a new feature critical_amount which indicates whether the transaction amount is greater than 200000 or not, and this is totally based on the context of the dataset
            df['is_round'] = (df['amount'] % 1000 == 0).astype(int)  #creating a new feature is_round which indicates whether the transaction amount is a round number (multiple of 1000) or not, as fraudsters often use round numbers to avoid detection
#and this 200000 threshold is chosen based on the context of the dataset.
            df['day_of_week'] = (df['day'] - 1) % 7  # 0=Monday, 1=Tuesday, ... 6=Sunday  
            df['is_night_transaction'] = ((df['hour_of_day'] >= 0) & (df['hour_of_day'] <= 6)).astype(int)  #creating a new feature is_night_transaction which indicates whether the transaction occurred during the night hours (0-6) or not, as fraudulent transactions may be more likely to occur during these hours    
            drop_cols = ['type', 'namedest', 'isflaggedfraud',
        'oldbalanceorg', 'newbalanceorig', 'oldbalancedest', 'newbalancedest', 'hour_of_day', 'day','dest_type','day_name']
            df= df.drop(columns=[col for col in df.columns if col in drop_cols])
            df['amount_vs_account_mean'] = df['amount'] / (df['nameorig'].map(self.preprocessors['train_account_mean']).fillna(self.preprocessors['global_mean'])+ 1) 
            df['account_txn_counts'] = df['nameorig'].map(self.preprocessors['account_txn_counts']).fillna(0) + 1 
            df = df.drop(columns = ['amount','nameorig'])
            continuous_features = ['log_amount','amount_vs_account_mean','account_txn_counts', 'step']
            df[continuous_features] = self.preprocessors['scalar'].transform(df[continuous_features])
            logging.info('feature engineering applied on incoming transaction dataset.')
            return df
        except Exception as e:
            raise CustomError(e, sys)
