import sys
import great_expectations as ge
import pandas as pd
from src.logger import logging
from src.exception import CustomError
from src.initial_data_phase.load_data import load_data
from src.initial_data_phase.preprocess import preprocess_data

def validate_data(df:pd.DataFrame):
    try:

        gdf = ge.from_pandas(df) #converting the dataframe into great_expectations dataframe
        logging.info('Validating schema and required columns')
        #features check
        gdf.expect_column_to_exist('step')
        gdf.expect_column_to_exist('type')
        gdf.expect_column_to_exist('amount')  
        gdf.expect_column_to_exist('namedest')  #this is the destinational data representing whether the destination of transaction is merchant or customer
        gdf.expect_column_to_exist('nameorig')
        
        #values check
        gdf.expect_column_values_to_be_in_set('type', ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'])
        gdf.expect_column_values_to_be_between('amount', min_value=0)  #the transaction amount must not be lesser than 0

        #null check
        gdf.expect_column_values_to_not_be_null('amount')
        gdf.expect_column_values_to_not_be_null('type')
        gdf.expect_column_values_to_not_be_null('nameorig')
        gdf.expect_column_values_to_not_be_null('namedest')


        results = gdf.validate()  #performing the validation task
        failed_expectations = [r['expectation_config']['expectation_type'] for r in results['results'] if not r['success']]  #here we are storing all the main causes for the columns that had failed validation

        
        if results['success']:
            print('Data validation complete')
        else:
            print('Unvalidated data')
        return results['success'] , failed_expectations
    except Exception as e:
        raise CustomError(e,sys)
        

if __name__ == "__main__":
    df = load_data('data/pay_sim.csv')
    preprocessed_df = preprocess_data(df)
    print(validate_data(preprocessed_df))        
    


    


