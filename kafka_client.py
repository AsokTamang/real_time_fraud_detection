# this file is for setting up the kafka client configuration for both producer and consumer
import os
from src.logger import logging


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

