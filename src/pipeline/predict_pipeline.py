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
            print("DEBUG: logging =", logging)
              
           
            predicted_price =self.model.predict(features)  #predicting the target variable using the trained model on the input features
            logging.info(f"The predicted price of the house is {predicted_price[0]}")  #logging the predicted price of the house
            return np.expm1(predicted_price)   #as we have used log in the target variable during traininig, so converting the predicted price back into the oringinal price
        except Exception as e:
            import traceback
            traceback.print_exc()  # ← prints full stack trace to console
            raise CustomError(e, sys)