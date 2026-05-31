import streamlit as st
import threading
from collections import deque, defaultdict

#the pause event is for controlling the pausing and resuming of the consumer thread running in a background,
pause_event = threading.Event()  # Event to control pausing and resuming the consumer thread
pause_event.set()  # running the consumer thread by default when the app starts as we use this pause_event in our consumer thread

def initialize_state():
    if "lock" not in st.session_state: st.session_state.lock = threading.RLock()  #we are using reentrant lock so that the same thread can acquire the lock multiple times 
    if "messages"       not in st.session_state: st.session_state.messages       = deque(maxlen=200)  #we are only storing the latest 200 messages in the session state to avoid memory issues,
    if "total"          not in st.session_state: st.session_state.total          = 0  #total number of transactions
    if "fraud_count"    not in st.session_state: st.session_state.fraud_count    = 0  #total number of fraud transactions
    if "legit_count"    not in st.session_state: st.session_state.legit_count    = 0  #total number of legit transactions
    if "type_counts"    not in st.session_state: st.session_state.type_counts    = defaultdict(lambda: {"fraud": 0, "legit": 0})  #type of transaction counts for fraud and legit transactions
    if "tpm_history"    not in st.session_state: st.session_state.tpm_history    = deque(maxlen=20)   # transactions-per-poll
    if "running"        not in st.session_state: st.session_state.running        = True  #running state of consumer thread
    if "consumer_thread"not in st.session_state: st.session_state.consumer_thread= None  #to store the consumer thread object
    if "last_alert"     not in st.session_state: st.session_state.last_alert     = None  #the last alert message shown in the dashoboard when the fraud transaction is detected by the model