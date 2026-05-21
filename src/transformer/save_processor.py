from sklearn.preprocessing import RobustScaler
from src.initial_data_phase.validate_data import validate_data
from src.initial_data_phase.load_data import load_data
from src.initial_data_phase.preprocess import preprocess_data
from src.features.feature_engineering import build_features
from src.logger import logging
from src.exception import CustomError
from src.utils import save_object
import pandas as pd
import os
import sys


class Datascalar:
    def __init__(self):
        self.preprocessor_file_path = os.path.join('artifacts','preprocessors.pkl')

    def split_data(self,df:pd.DataFrame):
        try:
            
            df1=df.sort_values('step').reset_index(drop=True)  #sorting the dataframe based on time step
            X=df1.drop(columns = ['isfraud'])
            y=df1['isfraud']    
            
            
            #for steps to split dataset into training, cross_Validation and test dataset
            train_step= 575    # day 1 to 15, for the training dataset
            val_step=647      # day 16 to 23, fot the cross_validation dataset
            #after day 23, we assign the test dataset
            
            train_mask=df1['step']<=train_step  #training dataset
            val_mask=(df1['step']>train_step)&((df1['step']<=val_step))  #cross_validation dataset
            test_mask=df1['step']>val_step   #test dataset
            
            #splitting of dataset
            X_train,X_val, X_test=X[train_mask].copy(),X[val_mask].copy(), X[test_mask].copy()
            y_train,y_val, y_test=y[train_mask].copy(),y[val_mask].copy(), y[test_mask].copy() 


            #feature engineering based on training dataset
            #inorder to prevent the dataleakage, we are using the transaction mean based on training dataset only
            train_account_mean = X_train.groupby('nameorig')['amount'].mean() #finding the mean of all the transaction amounts based on the account holder name
            X_train['amount_vs_account_mean'] = X_train['amount'] / (train_account_mean + 1) #checking how much the current transaction amount is different from the usual transaction amount of the user, can be effective feature for fraud detection
            global_mean = X_train['amount'].mean()

            #here global mean is the fallback value, if the nameorig in validation or test dataset isnot found in training dataset
            X_test['amount_vs_account_mean'] = X_test['amount'] / (X_test['nameorig'].map(train_account_mean).fillna(global_mean)+ 1) 
            X_val['amount_vs_account_mean'] = X_val['amount'] / (X_val['nameorig'].map(train_account_mean).fillna(global_mean) + 1)
            
            train_transaction_counts = X_train.groupby('nameorig')['amount'].count()  #here we are calculating the transaction count for each account holder in the training dataset
            X_train['txn_count_per_account'] = X_train['nameorig'].map(train_transaction_counts).fillna(0) + 1  #here we are creating a new feature called transaction count per account which indicates how many transactions have been made by the account holder, based on the training dataset
            X_val['txn_count_per_account'] = X_val['nameorig'].map(train_transaction_counts).fillna(0) + 1  #same here for validation dataset
            X_test['txn_count_per_account'] = X_test['nameorig'].map(train_transaction_counts).fillna(0) + 1  #same here for test dataset

            X_train = X_train.drop(columns = ['amount','nameorig'])
            X_val = X_val.drop(columns = ['amount','nameorig'])
            X_test = X_test.drop(columns = ['amount','nameorig'])


            #scaling pipeline
            scalar = RobustScaler()
            continuous_features = ['log_amount','amount_vs_account_mean','txn_count_per_account', 'step']

            X_train[continuous_features] = scalar.fit_transform(X_train[continuous_features])
            X_val[continuous_features] = scalar.transform(X_val[continuous_features])
            X_test[continuous_features] = scalar.transform(X_test[continuous_features])
            # here we are measuring the frequency of transaction done by each account user till current time
            
                        
            #saving the preprocessor object
            save_object(self.preprocessor_file_path,{
                'scalar':scalar,
                'train_account_mean':train_account_mean,
                'global_mean':global_mean,
                'account_txn_counts':train_transaction_counts


            })  #saving the trained scalar preprocessor
            logging.info('Scalar saved as pickle file')
             
            
            return X_train,X_val,X_test,y_train,y_val,y_test
        except Exception as e:
            raise CustomError(e,sys)

if __name__ == '__main__':
    datascalar = Datascalar()
    df = load_data('data/pay_sim.csv')  #loading the data
    print('1. Data loaded')
    preprocessed_df = preprocess_data(df)  #preprocessing the data
    print('2. Preprocessed data')
    print(validate_data(preprocessed_df))  #validation the data
    print('3. Validated data')
    engineered_df = build_features(preprocessed_df)  #applied the feature engineering on preprocessed data
    print('4. Applied feature engineering')
    print(datascalar.split_data(engineered_df))
    print('5. preprocessor saved as pickle file')




