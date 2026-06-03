import threading
from collections import deque, defaultdict

#the pause event is for controlling the pausing and resuming of the consumer thread running in a background,
pause_event = threading.Event()  # Event to control pausing and resuming the consumer thread
pause_event.set()  # running the consumer thread by default when the app starts as we use this pause_event in our consumer thread

#this lock is for synchronizing the access to the state called shared_state across main as well as background threads
state_lock = threading.RLock() 
# Shared runtime state (NOT Streamlit session_state) as the session state cannot be shared across multiple threads
shared_state = {
    "messages": deque(maxlen=200),  #total number of transaction details with the time of transaction received and the value of kafka partition
    "total": 0,  #total number of transactions
    "fraud_count": 0,  #total number of fraud transactions
    "legit_count": 0,  #total number of legit transactions
    "type_counts": defaultdict(lambda: {"fraud": 0, "legit": 0}),   #total number of fraud and legit transactions based on the type of transactions
    "tpm_history": deque(maxlen=20),   #history of transactions per minute
    "last_alert": None,     #last fraud transaction details
    "running": True,        #running state of consumer thread 
}