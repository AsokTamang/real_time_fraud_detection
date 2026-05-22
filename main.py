from src.exception import CustomError
from src.logger import logging
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.data_validation_pipeline import CustomData
from src.pipeline.feature_engineering_pipeline import Feature_engineering
from pydantic import BaseModel
from typing import Union
import uvicorn
import sys

app = FastAPI()
app.mount('/static', StaticFiles(directory='statics'), name='statics')

feature_engineering_pipeline = Feature_engineering()
predict_pipeline = PredictPipeline()  #calling the predict pipeline class



class PredictRequest(BaseModel):  #class for defining the input data for prediction
    step:Union[int,None] = None
    type: Union[str, None] = None
    amount: Union[float, None] = None
    nameorig: Union[str, None] = None
    namedest: Union[str, None] = None
    oldbalanceorg: Union[float, None] = None



@app.get('/')
def prediction_form():
    return FileResponse('statics/fraud_detection_form.html')


@app.post('/predict')
def predict(data: PredictRequest):
    try:
        features = CustomData(
        step= data.step,    
        type = data.type,
        amount = data.amount,
        nameorig = data.nameorig,
        namedest = data.namedest,
        oldbalanceorg = data.oldbalanceorg)

        df_features = features.get_data_as_dataframe() #converting into dataframe
        df_features = feature_engineering_pipeline.feature_engineering(df_features)  #applying the feature engineering pipeline
        result = predict_pipeline.predict(df_features)
        
        return result

    except Exception as e:
        raise CustomError(e,sys)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)