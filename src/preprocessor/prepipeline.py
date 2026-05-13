from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from src.logger import logging
from src.exception import CustomError
from src.utils import save_object
import pandas as pd
import os


class Datascalar:
    def __init__(self):
        self.preprocessor_file_path = os.path.join('artifacts','preprocessor.pkl')

    def split_data(self,df:pd.DataFrame):
        try:
            X, y = df.drop('isfraud', axis=1), df['isfraud']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  #here we are splitting the dataset into training and testing sets with stratification to maintain the class distribution in both sets
            scalar = RobustScaler()
            X_train['log_amount'] = scalar.fit_transform(X_train[['log_amount']])  #scaling the log_amount feature using RobustScaler to reduce the impact of outliers on the model training
            X_test['log_amount'] = scalar.transform(X_test[['log_amount']])  
            save_object(self.preprocessor_file_path,scalar)  #saving the trained scalar preprocessor
            logging.info('Scalar saved as pickle file')
            return X_train,X_test,y_train,y_test
        except Exception as e:
            raise CustomError(e)




