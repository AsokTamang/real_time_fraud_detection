import os
import json
import signal 
from confluent_kafka import Consumer
from src.logger import logging
from kafka_client import consumer_config, FRAUD_RESULT_TOPIC



running = True
#signal handler to handle the shutdown signal and stop the consumer gracefully when the stop signal is received such as ctrl + C or kill command
def handle_shutdown(signum, frame):
     global running
     logging.info("Shutdown signal received, stopping consumer...")
     running = False

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

     

def consumer(topic, consumer_config):
        # creating a new consumer instance
        consumer = Consumer(consumer_config)

        # subscribing to the specified topic for this current consumer, then only our consumer can consume the event from this specific topic
        consumer.subscribe([topic])
        try:
            while True:
                # consumer polls the topic and prints any incoming messages
                msg = consumer.poll(1.0)  #our consumer waits for 1 second to recieve new events or messages, if new events or messages are not available at the moment
                if msg is not None and msg.error() is None:
                    key = msg.key().decode("utf-8")
                    value = msg.value().decode("utf-8")
                    logging.info(
                        f"Consumed message from topic {topic}: key = {key:12} value = {value:12}"
                    )
        except KeyboardInterrupt:
            pass
        finally:
            # closes the consumer connection
            consumer.commit(msg)
            consumer.close()

if __name__ == "__main__":
    consumer(FRAUD_RESULT_TOPIC, consumer_config)