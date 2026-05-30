# this file is for setting up the kafka client configuration for both producer and consumer
import os
from src.logger import logging
from confluent_kafka import Producer
import json
import streamlit as st
from dotenv import load_dotenv
load_dotenv(dotenv_path='secrets/.env')  # loading the environment variables from .env file

def read_config():
    # reads the client configuration from client.properties
    # and returns it as a key-value map
    config = {}
    with open(r"secrets/client.properties") as fh:
        for line in fh:
            line = line.strip()
            if len(line) != 0 and line[0] != "#":
                parameter, value = line.strip().split("=", 1)
                config[parameter] = value.strip()
    return config


#delivery report function to check whether the event is delivered or not
def delivery_report(err,msg):
    if err is not None:
        logging.info('kafka delivery failed', err)
    else:
        logging.info(
            f'kafka message delivered, topic->{msg.topic()}'
            f'partition->{msg.partition()} offset->{msg.offset()}'
        )


#here we are creating a function to send the failed messages to the dead letter queue (DLQ) topic in kafka with the reason for failure so that we can debug the problematic transactions later without stopping the rest of the pipeline.
def send_to_dlq(dlq_producer: Producer, raw_value: bytes, reason: str) -> None:
    dlq_payload = {
        "original_message": raw_value.decode("utf-8") if raw_value else None,
        "failure_reason": reason,
    }
    #we are also converting the dlq payload in the json format
    dlq_producer.produce(
        topic=DLQ_TOPIC,
        value=json.dumps(dlq_payload),
        on_delivery=delivery_report,
    )
    #using poll to trigger the delivery report
    dlq_producer.poll(0)
    logging.error(f" Message sent to DLQ | Reason: {reason}")



#GRACEFUL SHUTDOWN OF THE CONSUMER  
#signal handler to handle the shutdown signal and stop the consumer gracefully when the stop signal is received  such as ctrl + C or kill command
#and this stop signal is stored inside the streamlit session state
def handle_shutdown(signum, frame):
     
     logging.info("Shutdown signal received, stopping consumer...")
     st.session_state.running = False
#registering the signal handlers

            


# we must separate the config for producer and consumer because they have different configurations for the Kafka client
#configuration of producer in kafka client
producer_config = (
    read_config()
)  # as the producer config only needs the bootstrap server and auth configuration, no need for additional configuration for producer

# configuration of consumer in kafka client
consumer_config = read_config()
consumer_config["group.id"] = os.getenv("KAFKA_CONSUMER_GROUP_ID")
consumer_config["auto.offset.reset"] = "earliest"
TRANSACTION_TOPIC = os.getenv(
    "KAFKA_TRANSACTION_TOPIC"
)  # our topic where the event produced from producer reaches
FRAUD_RESULT_TOPIC = os.getenv(
    "KAFKA_FRAUD_RESULT_TOPIC"
)  # our topic where the prediction done by the model reaches
DLQ_TOPIC = os.getenv("KAFKA_DLQ_TOPIC")

