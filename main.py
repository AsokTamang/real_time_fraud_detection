import sys
from contextlib import asynccontextmanager
from confluent_kafka import Producer
from src.exception import CustomError
from src.logger import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.data_validation_pipeline import CustomData
from src.pipeline.feature_engineering_pipeline import Feature_engineering
from pydantic import BaseModel, field_validator
from typing import Optional, Union
import uvicorn
from kafka_client import producer_config, FRAUD_RESULT_TOPIC



#global variables for the expensive instances to be loaded at the time of app startup
feature_engineering_pipeline: Optional[Feature_engineering] = None
predict_pipeline: Optional[PredictPipeline] = None
kafka_producer: Optional[Producer] = None

#defining the lifespan of our fraud detection endpoint or app
@asynccontextmanager
async def lifespan_info(app:FastAPI):
    #using the predefined global instances
    global feature_engineering_pipeline,predict_pipeline,kafka_producer
    
    feature_engineering_pipeline = Feature_engineering()
    predict_pipeline = PredictPipeline()
    logging.info('ML pipelines loaded successfully')
    kafka_producer = Producer(producer_config)
    logging.info('kafka producer configured')


    yield
    
    logging.info('flushing kafka producer')
    kafka_producer.flush()
    logging.info('kafka producer flushed, shutdown complete')
#defining the app with the defined lifespan above
app = FastAPI(
    title = 'Fraud Detection API',
    description = 'Real-time fraud detection using ML + Kafka streaming',
    version='1.0.0',
    lifespan=lifespan_info 
)
app.mount("/static", StaticFiles(directory="statics"), name="statics")



class PredictRequest(BaseModel):  # class for defining the input data for prediction
    step: Union[int, None] = None
    type: Union[str, None] = None
    amount: Union[float, None] = None
    nameorig: Union[str, None] = None
    namedest: Union[str, None] = None
    oldbalanceorg: Union[float, None] = None

    #validation check for amount
    @field_validator('amount')
    def amount_must_be_positive(cls, v:float):
        if v is not None and v<0 :
            raise ValueError('Amount mustnot be negative')
        return v
    
    #validation check for type
    @field_validator('type')
    def type_must_be_valid(cls, v:str):
        valid_list = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
        if v is not None and v.upper() not in valid_list :
            raise ValueError('Type must be valid')
        return v
    

    





@app.get("/")
def prediction_form():
    return FileResponse("statics/fraud_detection_form.html")


@app.post("/predict")
def predict(data: PredictRequest):
    try:
        features = CustomData(
            step=data.step,
            type=data.type,
            amount=data.amount,
            nameorig=data.nameorig,
            namedest=data.namedest,
            oldbalanceorg=data.oldbalanceorg,
        )

        df_features = features.get_data_as_dataframe()  # converting into dataframe
        df_features = feature_engineering_pipeline.feature_engineering(
            df_features
        )  # applying the feature engineering pipeline in the incoming transaction datas
        result = predict_pipeline.predict(df_features)

        # Producer code to send the prediction result to the Kafka topic
        #here we are using the nameorig of the transaction user as the key , so that the messages of the same account holder of transaction will be stored in the same partition
        kafka_producer.produce(
            FRAUD_RESULT_TOPIC, key=str(data.nameorig), value=result['result']
        )  # producing the prediction result to the Kafka topic, and as we stored the output of the prediction in the key called 'result', we are using result['result']
        logging.info(
            f"Produced message to topic {FRAUD_RESULT_TOPIC}: key = {'prediction':12} value = {result['result']:12}"
        )
       
        
        return result

    except Exception as e:
        raise CustomError(e, sys)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
