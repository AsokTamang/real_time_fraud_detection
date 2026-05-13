import sys
import os
import numpy as np
from pandas.api.types import is_numeric_dtype
from src.logger import logging
from src.exception import CustomError
import dill


class EarlyStoppingCallback:
    def __init__(self, patience):
        self.patience = patience
        self.best_value = None
        self.stagnant_trials = 0

    def __call__(self, study, trial):
        if self.best_value is None or study.best_value > self.best_value:
            self.best_value = study.best_value
            self.stagnant_trials = 0
        else:
            self.stagnant_trials += 1

        if self.stagnant_trials >= self.patience:  #if the number of trials reached the number of patience then we just stop doing the trial test 
            print(f"Stopping study: No improvement for {self.patience} trials.")
            study.stop()





def reduce_memory(df,verbose=True):
    start_memory = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype   #extracting the type of column
        if is_numeric_dtype(df[col]) :  #only if the current column type is number, we change the memory accordingly to the type whether its integer or float
            c_min,c_max = df[col].min(),df[col].max()
            if np.issubdtype(col_type,np.integer):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            elif np.issubdtype(col_type,np.floating):
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
        else:  #if its a string then we change the type to category based on the number of unique values
           if (df[col].nunique() / len(df)) < 0.5:
                df[col] = df[col].astype('category')
    
    end_memory = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage: {start_memory:.2f} MB → {end_memory:.2f} MB '
              f'({100 * (start_memory - end_memory) / start_memory:.1f}% reduction)')
    
    return df

def save_object(file_path, obj):    
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)  # Create the directory if it doesn't exist
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)  # Use dill to serialize the object
            logging.info(f"Object saved successfully at {file_path}")  #logging the file path where the object is saved
    except Exception as e:
        raise CustomError(e, sys)
