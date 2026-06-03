from datetime import datetime
import threading
import json
import time
from confluent_kafka import Consumer, Producer, KafkaException, KafkaError
from src.logger import logging
from kafka_client import (
    producer_config,
    FRAUD_RESULT_TOPIC,
    consumer_config,
    send_to_dlq,
)
import streamlit as st
import json
from state import  pause_event, state_lock, shared_state
from dashboard import display_ui
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="Real-Time Fraud Detection Dashboard", page_icon="🛡️", layout="wide"
)
st_autorefresh(interval=1000)
# At the top of your streamlit app file, after imports





def run_consumer():
    consumer = Consumer(consumer_config)  #configured consumer to consume messages from kafka topic
    dlq_producer = Producer(producer_config)  #configured producer to send the failed messages to the DLQ topic
    try:
        consumer.subscribe(
            [FRAUD_RESULT_TOPIC]
        )  # subscribing to the topic where the prediction result is produced by the producer
        logging.info(f"Consumer subscribed to topic {FRAUD_RESULT_TOPIC}")

        while True:
            # if the running state of the consumer is false then we wait for 1 second and again check the running state of the consumer
            if (
                not pause_event.is_set()
            ):  # checking if the consumer thread is in paused state or not by checking the pause_event state, if it is not set then it means the consumer thread is in paused state and we will wait for 0.2 second before checking again
                time.sleep(1)
                continue
            # only if the consumer state is running, we proceed with the transactions
            #checking new messages in kafka topic
            msg = consumer.poll(
                1.0
            )  # polling for new messages with a timeout of 1 second
            if msg is None:
                continue  # no message received, continue polling
            if msg.error():
                if (
                    msg.error().code() == KafkaError._PARTITION_EOF
                ):  # checking if the error is due to reaching the end of the partition,
                    logging.info(
                        f"End of partition reached {msg.topic()} [{msg.partition()}]"
                    )
                else:  # if the error is due to other reasons
                    logging.error(f"Kafka error: {msg.error()}")
                    send_to_dlq(
                        dlq_producer, msg.value(), f"Kafka error: {msg.error()}"
                    )  # sending the error message to the DLQ with reason for failure
                continue


            # Processing the message
            try:
                # loading the payload
                raw_value = msg.value()
                message_value = (
                    json.loads(raw_value.decode("utf-8")) if raw_value else None
                )  # decoding the message value from bytes to string and converting into python object
                logging.info(
                    f"Received message: {message_value} from topic {msg.topic()} partition {msg.partition()} offset {msg.offset()}"
                )
                transaction_data = message_value[
                    "transaction"
                ]  # All the incoming transaction details from the frontend are stored in the key called transaction inside the payload
                transaction_type = transaction_data[
                    "type"
                ]  # type of incoming transaction

                enriched = {
                    **message_value,
                    "received_at": datetime.now().strftime("%H:%M:%S"),
                    "partition": msg.partition(),
                    "offset": msg.offset(),
                }
                # if the passed transaction in the payload is fraudulent transaction, then we increase the number of variables accordingly
                with state_lock:  # acquiring the lock to update the shared state variables in a thread safe way
                    if message_value["is_fraud"]:
                        shared_state['fraud_count'] += 1  # incrementing the total number of transactions predicted as fraud
                        shared_state['type_counts'][transaction_type][
                            "fraud"
                        ] += 1  # incrementing the total number of transactions predicted as fraud for the specific type of transaction
                        shared_state['last_alert'] = enriched  # storing the enriched message in the session state to show it as the last alert in the dashboard if the transaction is predicted fraud by our model
                    else:
                        shared_state['legit_count'] += 1  # incrementing the total number of transactions predicted as legit
                        shared_state['type_counts'][transaction_type][
                            "legit"
                        ] += 1  # incrementing the total number of transactions predicted as legit for the specific type of transaction
                    shared_state['tpm_history'].append(
                        {"time": datetime.now().strftime("%H:%M"), "count": 1}
                    )  # storing the transactions per poll history in the session state to show it in the dashboard as a line chart
                    shared_state['messages'].appendleft(
                        enriched
                    )  # storing the enriched message in the session state at left to show it in the dashboard as we pop the messages that are too old than the current 200 transaction details
                    shared_state['total'] += (
                        1  # incrementing the total number of transactions processed
                    )
                consumer.commit(
                    asynchronous=False
                )  # committing the message offset after processing the message successfully

                # after all the initialization and setup of the consumer thread, we will run the consumer function to start consuming the messages from the kafka topic
                # then we update the dashboard in real time with the prediction results of our model.

            except Exception as e:
                                     # <-- add this
                logging.error(f"Unexpected error in consumer thread: {e}", exc_info=True)
                send_to_dlq(dlq_producer, msg.value(), f"Processing error: {e}")
    except KafkaException as ke:
        logging.error(f"Kafka error: {ke}")
    finally:
        consumer.close()
        dlq_producer.flush()  # Ensure all messages are sent before shutting down
        logging.info("Consumer and DLQ producer closed gracefully.")


def start_consumer():
    if "consumer_thread" not in st.session_state or not st.session_state["consumer_thread"].is_alive():
        
        new_thread = threading.Thread(target=run_consumer, daemon=True)
        new_thread.start()
        st.session_state["consumer_thread"] = new_thread
        logging.info("Consumer thread started.")
    else:
        logging.info("Consumer thread already running, skipping start.")
start_consumer()  # Starting the consumer thread to consume messages from the kafka topic

display_ui()