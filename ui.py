import threading
import json
import time
from datetime import datetime
from confluent_kafka import Consumer, Producer, KafkaException, KafkaError

from src.logger import logging
from kafka_client import producer_config, FRAUD_RESULT_TOPIC, consumer_config, send_to_dlq
from state import pause_event, stop_event, state_lock, shared_state, VALID_TRANSACTION_TYPES

import streamlit as st
from dashboard import display_ui
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="Real-Time Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
)
st_autorefresh(interval=2000)  


_consumer_thread: threading.Thread | None = None
_thread_lock = threading.Lock()


def run_consumer() -> None:
    consumer = Consumer(consumer_config)
    dlq_producer = Producer(producer_config)

    def on_assign(c, partitions):  #this functions is called when the consumer subscirbed to the particulat topic like below is assigned a partition
        logging.info(f"Partitions assigned: {repr(partitions)}")

    def on_revoke(c, partitions):  #this function is called when the consumer is removed from a partititon and the messages that are not yet committed from offsets are committed before the consumer is removed from the partition, inorder to prevent the processing of duplicate messages
        logging.info(f"Partitions revoked — committing offsets: {repr(partitions)}")
        c.commit(asynchronous=False)

    try:
        consumer.subscribe([FRAUD_RESULT_TOPIC], on_assign=on_assign, on_revoke=on_revoke)
        logging.info(f"Consumer subscribed to {FRAUD_RESULT_TOPIC}")

        while not stop_event.is_set():     #checking the stop_event to allow graceful shutdown, if the stop_event is not set the we keep the consumer running and polling for messages
            if not pause_event.is_set():   #checking whether the pause event is set or not , to check if the consumer thread is paused or not 
                time.sleep(0.2)
                continue

            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logging.info(f"EOF: {msg.topic()}[{msg.partition()}]")
                else:
                    logging.error(f"Kafka error: {msg.error()}")
                    send_to_dlq(dlq_producer, msg.value(), f"Kafka error: {msg.error()}")
                continue

            try:
                raw = msg.value()
                if not raw:
                    logging.warning("Received empty message — skipping")
                    consumer.commit(asynchronous=False)
                    continue

                payload: dict = json.loads(raw.decode("utf-8"))
                transaction: dict = payload.get("transaction", {})
                is_fraud: bool = payload.get("is_fraud", False)

             
                t_type: str = (transaction.get("type") or "UNKNOWN").upper()
                if t_type not in VALID_TRANSACTION_TYPES:
                    logging.warning(f"Unknown transaction type '{t_type}' — defaulting to UNKNOWN")
                    t_type = "UNKNOWN"

                enriched = {
                    **payload,
                    "received_at": datetime.now().strftime("%H:%M:%S"),
                    "partition": msg.partition(),
                    "offset": msg.offset(),
                }

               
                with state_lock:
                    shared_state["total"] += 1
                    if is_fraud:
                        shared_state["fraud_count"] += 1
                        shared_state["type_counts"][t_type]["fraud"] += 1
                        shared_state["last_alert"] = enriched
                    else:
                        shared_state["legit_count"] += 1
                        shared_state["type_counts"][t_type]["legit"] += 1

                    shared_state["tpm_history"].append(
                        {"time": datetime.now().strftime("%H:%M"), "count": 1}
                    )
                    shared_state["messages"].appendleft(enriched)

                consumer.commit(asynchronous=False)
                logging.info(
                    f"Processed offset {msg.offset()} | fraud={is_fraud} | type={t_type}"
                )

            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error: {e}")
                send_to_dlq(dlq_producer, msg.value(), f"JSON error: {e}")
                consumer.commit(asynchronous=False)   # skipping bad message
            except Exception as e:
                logging.error(f"Unexpected consumer error: {e}", exc_info=True)
                send_to_dlq(dlq_producer, msg.value(), f"Processing error: {e}")

    except KafkaException as ke:
        logging.error(f"Fatal Kafka error: {ke}", exc_info=True)
    finally:
        consumer.close()
        dlq_producer.flush()
        logging.info("Consumer shut down cleanly.")


def start_consumer() -> None:
    "Starting the consumer thread exactly once, even across multiple Streamlit reruns"
    global _consumer_thread
    with _thread_lock:  
        if _consumer_thread is None or not _consumer_thread.is_alive():
            stop_event.clear()   #clearing the stop event before starting the consumer thread to ensure it's not set from a previous run, so the dashboard won't be hanged when the consumer thread is restarted after being stopped
            _consumer_thread = threading.Thread(
                target=run_consumer, daemon=True, name="kafka-consumer"
            )
            _consumer_thread.start()
            logging.info("Consumer thread started.")
        else:
            logging.debug("Consumer thread already running — skipping.")


start_consumer()
display_ui()