import pandas as pd
from src.utils import reduce_memory
from src.data.load_data import load_data
import numpy as np

def preprocess_data(df:pd.DataFrame):
    #for uniformaty of the feature namess
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(' ', '_')
    #reducing the memory of the features of dataframe
    df = reduce_memory(df) 
    #classification into numeric and categorical feature
    target = 'isfraud'
    numerical = [feature for feature in df.select_dtypes(include=np.number).columns if feature!=target]
    categorical = [feature for feature in df.select_dtypes(exclude=np.number).columns if feature!=target]
    #filling the null values for numerical features
    df[numerical] = df[numerical].fillna(0)
    return df

if __name__ == "__main__":
    df = load_data('data/pay_sim.csv')
    preprocessed_df = preprocess_data(df)
    print(preprocessed_df)