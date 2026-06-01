import streamlit as st
import threading
from collections import deque, defaultdict

#the pause event is for controlling the pausing and resuming of the consumer thread running in a background,
pause_event = threading.Event()  # Event to control pausing and resuming the consumer thread
pause_event.set()  # running the consumer thread by default when the app starts as we use this pause_event in our consumer thread

#this lock is for synchronizing the access to the state called shared_state across main as well as background threads
state_lock = threading.RLock() 
# Shared runtime state (NOT Streamlit session_state) as the session state cannot be shared across multiple threads
shared_state = {
    "messages": deque(maxlen=200),  #
    "total": 0,
    "fraud_count": 0,
    "legit_count": 0,
    "type_counts": defaultdict(lambda: {"fraud": 0, "legit": 0}),
    "tpm_history": deque(maxlen=20),
    "last_alert": None,
    "running": True,
    "consumer_thread": None
}