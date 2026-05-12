import numpy as np
from pandas.api.types import is_numeric_dtype
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