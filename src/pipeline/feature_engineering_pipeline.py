import os
from typing import Union
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
            df["log_amount"] = np.log1p(df["amount"])
            df["critical_transaction"] = (df["amount"] > 200000).astype(
                int
            )  # creating a new feature critical_amount which indicates whether the transaction amount is greater than 200000 or not, as we can see from the boxplot that there are some transactions with very high amounts which are likely to be fraudulent
            # and the amount greater than 200000 is the critical amount
            df["is_round_amount"] = (df["amount"] % 1000 == 0).astype(
                int
            )  # as most of the valid transaction is never a fixed number, so we are making a feature called is_round_amount

            # TRANSACTION TYPE
            df["dest_type"] = np.where(
                df["namedest"].str.startswith("M"), "Merchant", "Customer"
            )
            df["is_transfer"] = (df["type"] == "TRANSFER").astype(
                int
            )  # creating a new feature is_transfer which indicates whether the transaction type is a transfer or not
            df["is_cashout"] = (df["type"] == "CASH_OUT").astype(
                int
            )  # creating a new feature is_cash_out which indicates whether the transaction type is a cash out or not
            df["is_merchant_dest"] = (df["dest_type"] == "Merchant").astype(
                int
            )  # creating a new feature is_merchant_dest which indicates whether the destination account is a merchant or not based on the dest_type column

            # TIME FEATURES
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
            # using the trained scalar preprocessor to transform the continuous features
            df["amount_vs_account_mean"] = (
                df["amount"]
                / df["nameorig"]
                .map(self.preprocessors["train_account_mean"])
                .fillna(self.preprocessors["global_mean"])
                + 1
            )
            #here we are creating a new feature called transaction count per account which indicates how many transactions have been made by the account holder, based on the training dataset
            df["txn_count_per_account"] = (
                df["nameorig"].map(self.preprocessors["account_txn_counts"]).fillna(0)
                + 1
            )
            continuous_features = [
                "log_amount",
                "amount_vs_account_mean",
                "txn_count_per_account",
                "step",
            ]
            df[continuous_features] = self.preprocessors["scalar"].transform(df[continuous_features])

            # columns to be dropped
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
                "amount",
                "nameorig",
            ]
            df = df.drop(columns=drop_cols)
            return df
        except Exception as e:
            raise CustomError(e, sys)
