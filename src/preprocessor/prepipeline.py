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


            #scaling pipeline
            scalar = RobustScaler()
            X_train.loc[:, 'log_amount'] = scalar.fit_transform(X_train[['log_amount']])
            X_val.loc[:, 'log_amount'] = scalar.transform(X_val[['log_amount']])
            X_test.loc[:, 'log_amount'] = scalar.transform(X_test[['log_amount']])  
            
            #saving the preprocessor object
            save_object(self.preprocessor_file_path,scalar)  #saving the trained scalar preprocessor
            logging.info('Scalar saved as pickle file')
             
            #using downsampling method on training dataset to reduce the number of majority class or legimate transaction in this case
            rus = RandomUnderSampler(sampling_strategy='majority', random_state=42)
            X_train, y_train = rus.fit_resample(X_train, y_train) 
           
            
            return X_train,X_val,X_test,y_train,y_val,y_test
        except Exception as e:
            raise CustomError(e)




