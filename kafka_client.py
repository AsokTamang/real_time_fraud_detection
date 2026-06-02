# this file is for setting up the kafka client configuration for both producer and consumer
import os
from src.logger import logging
from confluent_kafka import Producer
import json
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join('secrets','env'))  # loading the environment variables from .env file

CONFIG_PATH = os.getenv("KAFKA_CONFIG_PATH")
TRANSACTION_TOPIC = os.getenv(
    "KAFKA_TRANSACTION_TOPIC"
)  # our topic where the event produced from producer reaches
FRAUD_RESULT_TOPIC = os.getenv(
    "KAFKA_FRAUD_RESULT_TOPIC"
)  # our topic where the prediction done by the model reaches
DLQ_TOPIC = os.getenv("KAFKA_DLQ_TOPIC")



config = {
    "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
    "security.protocol": os.getenv("KAFKA_SECURITY_PROTOCOL"),
    "sasl.mechanism": os.getenv("KAFKA_SASL_MECHANISM"),
    "sasl.username": os.getenv("KAFKA_SASL_USERNAME"),
    "sasl.password": os.getenv("KAFKA_SASL_PASSWORD"),
    "client.id": os.getenv("CLIENT_ID"),
    "session.timeout.ms": int(os.getenv("SESSION_TIMEOUT_MS")),
}

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


            


# we must separate the config for producer and consumer because they have different configurations for the Kafka client
#configuration of producer in kafka client
producer_config = config.copy()
  # as the producer config only needs the bootstrap server and auth configuration, no need for additional configuration for producer

# configuration of consumer in kafka client
consumer_config =config.copy()
consumer_config["group.id"] = os.getenv("KAFKA_CONSUMER_GROUP_ID")
consumer_config["auto.offset.reset"] = "earliest"


