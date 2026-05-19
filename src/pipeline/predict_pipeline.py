from src.logger import logging
from src.utils import load_object
import numpy as np
from src.exception import CustomError
import os
import sys

class PredictPipeline:
    def __init__(self):
        self.model_variables = load_object(os.path.join('artifacts','best_model_info.pkl'))  #loading the feature selector object which is used for feature selection
    def predict(self,features):    
        try:
            model = self.model_variables['model']  #extracting the model
            threshold = self.model_variables['threshold']   #extraction of the optimal threshold of the trained model
            y_prob = model.predict_proba(features)[
                :, 1
            ]  # only extracting the probability of the positive class (fraud)
            y_pred = (y_prob >= threshold).astype(
                int
            )     
            logging.info(f"The passed transaction is: {y_pred}")  #logging the predicted price of the house
            if y_pred == 1:
                return {'result': 'Fraud Transaction'}
            else:
                return {'result': 'Valid Transaction'}
        except Exception as e:
            import traceback
            traceback.print_exc()  # ← prints full stack trace to console
            raise CustomError(e, sys)