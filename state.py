import streamlit as st
from collections import deque, defaultdict
def initialize_state():
    if "messages"       not in st.session_state: st.session_state.messages       = deque(maxlen=200)
    if "total"          not in st.session_state: st.session_state.total          = 0
    if "fraud_count"    not in st.session_state: st.session_state.fraud_count    = 0
    if "legit_count"    not in st.session_state: st.session_state.legit_count    = 0
    if "type_counts"    not in st.session_state: st.session_state.type_counts    = defaultdict(lambda: {"fraud": 0, "legit": 0})
    if "tpm_history"    not in st.session_state: st.session_state.tpm_history    = deque(maxlen=20)   # transactions-per-poll
    if "running"        not in st.session_state: st.session_state.running        = True
    if "consumer_thread"not in st.session_state: st.session_state.consumer_thread= None
    if "last_alert"     not in st.session_state: st.session_state.last_alert     = None