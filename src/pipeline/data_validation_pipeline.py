from typing import Union
from src.exception import CustomError
from src.logger import logging
import sys
import pandas as pd
class CustomData:
    def __init__(self,
                 step:Union[int, None] = None,
                 type: Union[str, None] = None,
                 amount: Union[int, None] = None,
                 nameorig: Union[str, None] = None,
                 namedest: Union[str, None] = None,
                 oldbalanceorg: Union[int, None] = None,
    ):
        self.step = step
        self.type = type
        self.amount = amount
        self.nameorig = nameorig
        self.namedest = namedest
        self.oldbalanceorg = oldbalanceorg

    def get_data_as_dataframe(self):
        try:
            data_dict = {
                'step':[self.step],
                'type':[self.type],
                'amount':[self.amount],
                'nameorig':[self.nameorig],
                'namedest':[self.namedest],
                'oldbalanceorg':[self.oldbalanceorg],}
            df = pd.DataFrame(data_dict)
            logging.info('Dataframe created from incoming features')
            return df
        except Exception as e:
            raise CustomError(e, sys)