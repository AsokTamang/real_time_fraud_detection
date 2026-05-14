import pandas as pd
import numpy as np

from src.data.validate_data import validate_data
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # AMOUNT FEATURE
    df["log_amount"] = np.log1p(df["amount"])
    df["critical_transaction"] = (df["amount"] > 200000).astype(
        int
    )  # creating a new feature critical_amount which indicates whether the transaction amount is greater than 200000 or not, as we can see from the boxplot that there are some transactions with very high amounts which are likely to be fraudulent
    df["is_round"] = (df["amount"] % 1000 == 0).astype(int)
    df = df.sort_values("step").reset_index(
        drop=True
    )  # sorting the dataframe based on time step
    # here we are measuring the frequency of transaction done by each account user till current time
    df["txn_count_per_account"] = df.groupby("nameorig").cumcount() + 1

    
    # TRANSACTION TYPE
    df['dest_type'] = np.where(df['namedest'].str.startswith('M'), 'Merchant', 'Customer')
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(
        int
    )  # creating a new feature is_transfer which indicates whether the transaction type is a transfer or not
    df["is_cash_out"] = (df["type"] == "CASH_OUT").astype(
        int
    )  # creating a new feature is_cash_out which indicates whether the transaction type is a cash out or not
    df["is_merchant_dest"] = (df["dest_type"] == "Merchant").astype(
        int
    )  # creating a new feature is_merchant_dest which indicates whether the destination account is a merchant or not based on the dest_type column

    
    #TIME FEATURES
    df["day"] = np.ceil(df["step"] / 24).astype(
        int
    )  # converting the time step into days by dividing the step by 24 and taking the ceiling to get the day number
    df["day_of_week"] = (df["day"] - 1) % 7  # 0=Monday, 1=Tuesday, ... 6=Sunday
    df["hour_of_day"] = df["step"] % 24
    df["is_night_transaction"] = (
        (df["hour_of_day"] >= 0) & (df["hour_of_day"] <= 6)
    ).astype(
        int
    )  # creating a new feature is_night_transaction which indicates whether the transaction occurred during the night hours (0-6) or not, as fraudulent transactions may be more likely to occur during these hours
    # drop
    drop_cols = [
        "type",
        "namedest",
        "isflaggedfraud",
        "oldbalanceorg",
        "newbalanceorig",
        "oldbalancedest",
        "newbalancedest",
        "hour_of_day",
        "day",
        "dest_type",
    ]
    df = df.drop(columns=drop_cols)
    return df


if __name__ == '__main__':
    df = load_data('data/pay_sim.csv')
    preprocessed_df = preprocess_data(df)
    print(validate_data(preprocessed_df))
    print(build_features(preprocessed_df))  #applied the feature engineering on preprocessed data