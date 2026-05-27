import os
import json
import signal 
from confluent_kafka import Consumer,Producer, KafkaException, KafkaError
from src.exception import CustomError
from src.logger import logging
from kafka_client import consumer_config, FRAUD_RESULT_TOPIC
from kafka_client import producer_config, FRAUD_RESULT_TOPIC, delivery_report, DLQ_TOPIC



running = True
#GRACEFUL SHUTDOWN OF THE CONSUMER  
#signal handler to handle the shutdown signal and stop the consumer gracefully when the stop signal is received such as ctrl + C or kill command
def handle_shutdown(signum, frame):
     global running
     logging.info("Shutdown signal received, stopping consumer...")
     running = False

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

     

#here we are creating a function to send the failed messages to the dead letter queue (DLQ) topic in kafka with the reason for failure so that we can debug the problematic transactions later without stopping the rest of the pipeline.
def send_to_dlq(dlq_producer: Producer, raw_value: bytes, reason: str) -> None:
    dlq_payload = {
        "original_message": raw_value.decode("utf-8") if raw_value else None,
        "failure_reason": reason,
    }
    dlq_producer.produce(
        topic=DLQ_TOPIC,
        value=json.dumps(dlq_payload),
        on_delivery=delivery_report,
    )
    #using poll to trigger the delivery report
    dlq_producer.poll(0)
    logging.error(f" Message sent to DLQ | Reason: {reason}")


#here we need to build a real time dashboard of the prediction done by our model
def process_fraud_result(message_value: dict) -> None:
    pass

def run_consumer():
    consumer = Consumer(consumer_config)
    dlq_producer = Producer(producer_config)
    try:
        consumer.subscribe([FRAUD_RESULT_TOPIC])  #subscribing to the topic where the prediction result is produced by the producer
        logging.info(f"Consumer subscribed to topic {FRAUD_RESULT_TOPIC}")
        while running:
            msg = consumer.poll(1.0)  #polling for new messages with a timeout of 1 second
            if msg is None:
                continue  # no message received, continue polling
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:  #checking if the error is due to reaching the end of the partition, 
                    logging.info(f"End of partition reached {msg.topic()} [{msg.partition()}]")
                else:
                    logging.error(f"Kafka error: {msg.error()}")
                    send_to_dlq(dlq_producer, msg.value(), f"Kafka error: {msg.error()}")
                continue

            # Processing the message
            try:
                raw_value = msg.value()  #decoding the message value from bytes to string
                message_value = json.loads(raw_value.decode("utf-8")) if raw_value else None
                logging.info(f"Received message: {message_value} from topic {msg.topic()} partition {msg.partition()} offset {msg.offset()}")
                process_fraud_result(message_value)
                consumer.commit(asynchronous==False)  #committing the message offset after processing the message successfully  
                # Here you can add code to process the message as needed
            except Exception as e:
                logging.error(f"Error processing message: {e}")
                send_to_dlq(dlq_producer, msg.value(), f"Processing error: {e}")
    except KafkaException as ke:
        logging.error(f"Kafka error: {ke}")
    finally:
        consumer.close()
        dlq_producer.flush()  # Ensure all messages are sent before shutting down
        logging.info("Consumer and DLQ producer closed gracefully.")
if __name__ == "__main__":
    run_consumer()            