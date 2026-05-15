import pandas as pd
import os
from src.logger import logging
from src.exception import CustomError

def load_data(filepath:str) ->pd.DataFrame:
    if not os.path.exists(filepath):
        raise CustomError(f'file not found: {filepath}')
    #reading the data from the passed filepath
    df = pd.read_csv(filepath)
    logging.info('Read data from the filepath')
    return df

if __name__ == '__main__':
    print(load_data('data/pay_sim.csv'))
