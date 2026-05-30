from collections import defaultdict, deque
from datetime import datetime
import threading
import os
import json
import signal
import time 
from confluent_kafka import Consumer,Producer, KafkaException, KafkaError
from src.exception import CustomError
from src.logger import logging
from kafka_client import producer_config, FRAUD_RESULT_TOPIC, delivery_report, DLQ_TOPIC, consumer_config, send_to_dlq, handle_shutdown
import streamlit as st
import json
from state import initialize_state
from dashboard import display_ui


st.set_page_config(page_title="Real-Time Fraud Detection Dashboard",page_icon="🛡️",layout="wide")



#GRACEFUL SHUTDOWN OF THE CONSUMER 
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


def run_consumer():
    consumer = Consumer(consumer_config)
    dlq_producer = Producer(producer_config)
    try:
        consumer.subscribe([FRAUD_RESULT_TOPIC])  #subscribing to the topic where the prediction result is produced by the producer
        logging.info(f"Consumer subscribed to topic {FRAUD_RESULT_TOPIC}")
        
        while st.session_state.running:
            msg = consumer.poll(1.0)  #polling for new messages with a timeout of 1 second
            if msg is None:
                continue  # no message received, continue polling
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:  #checking if the error is due to reaching the end of the partition, 
                    logging.info(f"End of partition reached {msg.topic()} [{msg.partition()}]")
                else:  #if the error is due to other reason
                    logging.error(f"Kafka error: {msg.error()}")
                    send_to_dlq(dlq_producer, msg.value(), f"Kafka error: {msg.error()}")  #sending the error message to the DLQ with reason for failure
                continue

            # Processing the message
            try:
                #loading the payload
                raw_value = msg.value() 
                message_value = json.loads(raw_value.decode("utf-8")) if raw_value else None  #decoding the message value from bytes to string and converting into python object
                logging.info(f"Received message: {message_value} from topic {msg.topic()} partition {msg.partition()} offset {msg.offset()}")
                transaction_data = message_value['transaction']   #All the incoming transaction details from the frontend are stored in the key called transaction inside the payload
                transaction_type = transaction_data['type']

                enriched = {
                    **message_value,
                    "received_at": datetime.now().strftime("%H:%M:%S"),
                    "partition"  : msg.partition(),
                    "offset"     : msg.offset(),
                }
                #if the passed transaction in the payload is fraudulent transaction, then we increase the number of variables accordingly
                with st.session_state.lock:  #acquiring the lock to update the shared state variables in a thread safe way
                    if message_value["is_fraud"]:
                        st.session_state.fraud_count += 1  #incrementing the total number of transactions predicted as fraud
                        st.session_state.type_counts[transaction_type]["fraud"] += 1  #incrementing the total number of transactions predicted as fraud for the specific type of transaction
                        st.session_state.last_alert = enriched     #storing the enriched message in the session state to show it as the last alert in the dashboard if the transaction is predicted fraud by our model
                    else:
                        st.session_state.legit_count += 1  #incrementing the total number of transactions predicted as legit
                        st.session_state.type_counts[transaction_type]["legit"] += 1  #incrementing the total number of transactions predicted as legit for the specific type of transaction
                    st.session_state.tpm_history.append({"time": datetime.now().strftime("%H:%M"), "count": 1})  #storing the transactions per poll history in the session state to show it in the dashboard as a line chart
                    st.session_state.messages.appendleft(enriched)  #storing the enriched message in the session state at left to show it in the dashboard as we pop the messages that are too old than the current 200 transaction details
                    st.session_state.total += 1  #incrementing the total number of transactions processed
                consumer.commit(asynchronous=False)  #committing the message offset after processing the message successfully  
                
                #after all the initialization and setup of the consumer thread, we will run the consumer function to start consuming the messages from the kafka topic  
                #then we update the dashboard in real time with the prediction results of our model.
                
            except Exception as e:
                logging.error(f"Error processing message: {e}")
                send_to_dlq(dlq_producer, msg.value(), f"Processing error: {e}")
    except KafkaException as ke:
        logging.error(f"Kafka error: {ke}")
    finally:
        consumer.close()
        dlq_producer.flush()  # Ensure all messages are sent before shutting down
        logging.info("Consumer and DLQ producer closed gracefully.")


#session state initialization
initialize_state()  #calling the initialization function to initiate the streamlit session states

#here we are checking if the consumer thread is already running or not, if not then we will create a new thread to run the consumer function in the background so that it does not block the main thread of streamlit and allows us to update the dashboard in real time without any interruption.
if st.session_state.consumer_thread is None:
    st.session_state.consumer_thread = threading.Thread(target=run_consumer,daemon=True)   #creating a consumer thread to run the consumer function in the background so that it does not block the main thread of streamlit and allows us to update the dashboard in real time  
    st.session_state.consumer_thread.start()  #starting the consumer thread 
display_ui()  #calling the function to display the dashboard UI and only on main thread
time.sleep(1)
st.rerun()
