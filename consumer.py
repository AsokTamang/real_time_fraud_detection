from collections import defaultdict, deque
import datetime
import threading
import os
import json
import signal 
from confluent_kafka import Consumer,Producer, KafkaException, KafkaError
from src.exception import CustomError
from src.logger import logging
from kafka_client import producer_config, FRAUD_RESULT_TOPIC, delivery_report, DLQ_TOPIC, consumer_config
import streamlit as st
import json


st.set_page_config(page_title="Real-Time Fraud Detection Dashboard",page_icon="🛡️",layout="wide")

#session state initialization
if "messages" not in st.session_state:st.session_state.messages = deque(maxlen=200)  # List to store last 200 received messages
if "total"          not in st.session_state: st.session_state.total          = 0  #total number of transactions processed
if "fraud_count"    not in st.session_state: st.session_state.fraud_count    = 0  #total number of transactions predicted as fraud
if "legit_count"    not in st.session_state: st.session_state.legit_count    = 0   #total number of transactions predicted as legit
if "type_counts"    not in st.session_state: st.session_state.type_counts    = defaultdict(lambda: {"fraud": 0, "legit": 0})  #total number of either fraud or legit transactions for each type of transaction
if "tpm_history"    not in st.session_state: st.session_state.tpm_history    = deque(maxlen=20)   # total number of transactions per minute for the last 20 minutes to show the trend of the transaction volume
if "running"        not in st.session_state: st.session_state.running        = True   #the flag to control the running of the consumer thread, it will be set to false when we want to stop the consumer thread gracefully
if "consumer_thread"not in st.session_state: st.session_state.consumer_thread= None   #this state stores the consumer thread   
if "last_alert"     not in st.session_state: st.session_state.last_alert     = None   #the state to show the last alert message


#GRACEFUL SHUTDOWN OF THE CONSUMER  
#signal handler to handle the shutdown signal and stop the consumer gracefully when the stop signal is received  such as ctrl + C or kill command
#and this stop signal is stored inside the streamlit session state
def handle_shutdown(signum, frame):
     
     logging.info("Shutdown signal received, stopping consumer...")
     st.session_state.running = False
#registering the signal handlers
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

     

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


#here we need to build a real time dashboard of the prediction done by our model
def process_fraud_result(message_value: dict) -> None:
    st.title("Real-Time Fraud Detection")
    st.write("Fetching weather data from Kafka topic: weather-data")
    for message in message_value:
        transaction_data = message['transaction']
        result = message['result']
        final_result = message['is_fraud']

        st.write(f"transaction: {transaction_data['nameorig']}")
        st.write(f"result: {result}")
        st.write(f"final_result: {final_result}")

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
                    send_to_dlq(dlq_producer, msg.value(), f"Kafka error: {msg.error()}")
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
                
                if message_value["is_fraud"]:
                    st.session_state.fraud_count += 1  #incrementing the total number of transactions predicted as fraud
                    st.session_state.type_counts[transaction_type]["fraud"] += 1  #incrementing the total number of transactions predicted as fraud for the specific type of transaction
                    st.session_state.last_alert = enriched     #storing the enriched message in the session state to show it as the last alert in the dashboard if the transaction is predicted fraud by our model
                else:
                    st.session_state.legit_count += 1  #incrementing the total number of transactions predicted as legit
                    st.session_state.type_counts[transaction_type]["legit"] += 1  #incrementing the total number of transactions predicted as legit for the specific type of transaction
                
                
                st.session_state.messages.appendleft(enriched)  #storing the enriched message in the session state at left to show it in the dashboard as we pop the messages that are too old than the current 200 transaction details
                st.session_state.total += 1  #incrementing the total number of transactions processed
                
                process_fraud_result(message_value)
                consumer.commit(asynchronous=False)  #committing the message offset after processing the message successfully  
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


#here we are checking if the consumer thread is already running or not, if not then we will create a new thread to run the consumer function in the background so that it does not block the main thread of streamlit and allows us to update the dashboard in real time without any interruption.
if st.session_state.consumer_thread is None:
    st.session_state.consumer_thread = threading.Thread(target=run_consumer,daemon=True)   #creating a consumer thread to run the consumer function in the background so that it does not block the main thread of streamlit and allows us to update the dashboard in real time  
    st.session_state.consumer_thread.start()  #starting the consumer thread 


if __name__ == "__main__":
    run_consumer()            