from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from src.logger import logging
from src.exception import CustomError
from src.utils import save_object
import pandas as pd
import os
from imblearn.under_sampling import RandomUnderSampler


class Datascalar:
    def __init__(self):
        self.preprocessor_file_path = os.path.join('artifacts','preprocessor.pkl')

    def split_data(self,df:pd.DataFrame):
        try:
            
            df1=df.sort_values('step').reset_index(drop=True)  #sorting the dataframe based on time step
            features=['amount', 'log_amount',
                'is_round_amount', 'day_of_week',
                'is_transfer', 'is_cashout', 'is_night_transaction', 'is_merchant_dest',
                ]
            X=df1[features]
            y=df1['is_fraud']      
            
            
            #for steps to split dataset into training, cross_Validation and test dataset
            train_step= 575    # day 1to 15, for the training dataset
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
            train_account_mean = X_train.groupby('nameorig')['amount'].transform('mean') #finding the mean of all the transaction amounts based on the account holder name
            X_train['amount_vs_account_mean'] = X_train['amount'] / (train_account_mean + 1) #checking how much the current transaction amount is different from the usual transaction amount of the user, can be effective feature for fraud detection
            global_mean = X_train['amount'].mean()

            #here global mean is the fallback value, if the nameorig in validation or test dataset isnot found in training dataset
            X_test['amount_vs_account_mean'] = X_test['amount'] / X_test['nameorig'].map(train_account_mean).fillna(global_mean)+ 1 
            X_val['amount_vs_account_mean'] = X_val['amount'] / X_val['nameorig'].map(train_account_mean).fillna(global_mean) + 1


            X_train = X_train.drop(columns = ['amount','nameorig'])
            X_val = X_val.drop(columns = ['amount','nameorig'])
            X_test = X_test.drop(columns = ['amount','nameorig'])


            #scaling pipeline
            scalar = RobustScaler()
            continuous_features = ['log_amount','amount_vs_account_mean','txn_count_per_account', 'step']

            X_train[continuous_features] = scalar.fit_transform(X_train[continuous_features])
            X_val[continuous_features] = scalar.transform(X_val[continuous_features])
            X_test[continuous_features] = scalar.transform(X_test[continuous_features])
                        
            #saving the preprocessor object
            save_object(self.preprocessor_file_path,scalar)  #saving the trained scalar preprocessor
            logging.info('Scalar saved as pickle file')
             
            
            return X_train,X_val,X_test,y_train,y_val,y_test
        except Exception as e:
            raise CustomError(e)




