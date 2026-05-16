import sys
import pandera as pa
from pandera import Column, DataFrameSchema, Check
import pandas as pd
from src.logger import logging
from src.exception import CustomError
from src.initial_data_phase.load_data import load_data
from src.initial_data_phase.preprocess import preprocess_data

# Define the schema
schema = DataFrameSchema({
    "step":      Column(int),
    "type":      Column(str, Check(lambda x: x.isin(['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']), element_wise=False)),
    "amount":    Column(float, [
                     Check(lambda x: x >= 0, element_wise=True),  # amount must not be less than 0
                     Check(lambda x: x.notna().all(), element_wise=False)  # null check
                 ]),
    "nameorig":  Column(str, Check(lambda x: x.notna().all(), element_wise=False)),  # null check
    "namedest":  Column(str, Check(lambda x: x.notna().all(), element_wise=False)),  # null check — destination (merchant or customer)
})

def validate_data(df: pd.DataFrame):
    try:
        logging.info('Validating schema and required columns')

        failed_expectations = []

        # Check required columns exist
        required_columns = ['step', 'type', 'amount', 'nameorig', 'namedest']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            failed_expectations.append(f"Missing columns: {missing_columns}")
            return False, failed_expectations

        # Validate using pandera schema
        try:
            schema.validate(df, lazy=True)  # lazy = True collects all errors instead of stopping at first
            print('Data validation complete')
            return True, failed_expectations

        except pa.errors.SchemaErrors as e:
            # Collect all failed checks
            failed_expectations = e.failure_cases['check'].unique().tolist()
            print('Unvalidated data')
            return False, failed_expectations

    except Exception as e:
        raise CustomError(e, sys)


if __name__ == "__main__":
    df = load_data('data/pay_sim.csv')
    preprocessed_df = preprocess_data(df)
    print(validate_data(preprocessed_df))