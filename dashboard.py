from collections import defaultdict
import streamlit as st
import pandas as pd


def display_ui():
    st.title("Real-Time Fraud Detection Dashboard")
    col1, col2, col3, _ = st.columns([1, 1, 1, 5])
    with col1:
        if st.button("⏸ Pause" if st.session_state.running else "▶ Resume"):
            st.session_state.running = not st.session_state.running
    with col2:
        if st.button("🗑 Clear feed"):
            st.session_state.messages.clear()
    with col3:
        if st.button("↺ Reset stats"):
            st.session_state.total       = 0
            st.session_state.fraud_count = 0
            st.session_state.legit_count = 0
            st.session_state.type_counts = defaultdict(lambda: {"fraud": 0, "legit": 0})
            st.session_state.tpm_history.clear()
            st.session_state.last_alert  = None                
    
    st.divider()  #for better UI
    
    if st.session_state.last_alert:
        alert = st.session_state.last_alert
        incoming_transaction   = alert.get("transaction", {})  #as the incoming transaction details was stored inside the key called transaction in the payload produced by the producer
        st.error(
            f"🚨 **Fraud detected** — Account `{incoming_transaction.get('nameorig', 'N/A')}` · "
            f"Type: `{incoming_transaction.get('type', 'N/A')}` · "
            f"Amount: **${incoming_transaction.get('amount', 0):,.2f}** · "
            f"At: {alert.get('received_at', '')}",   #this is the time when the message is received by the consumer from the kafka topic
            icon="🚨",
        )        
    
    #metric cards for showing the total number of transactions, fraud transactions, legit transactions and fraud rate percentage
    total       = st.session_state.total
    fraud_count = st.session_state.fraud_count
    legit_count = st.session_state.legit_count
    fraud_rate  = (fraud_count / total * 100) if total > 0 else 0.0   #calculation of fraud transaction rate percentage
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📊 Total transactions", f"{total:,}")
    m2.metric("🚨 Fraudulent",         f"{fraud_count:,}",  delta=None)
    m3.metric("✅ Legitimate",          f"{legit_count:,}",  delta=None)
    m4.metric("📈 Fraud rate",          f"{fraud_rate:.1f}")
    
    st.divider()  
    
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Transactions per minute")
        buckets:dict = {}
        if st.session_state.tpm_history:
            for event in st.session_state.tpm_history:
                buckets[event["time"]] = buckets.get(event["time"], 0) + event["count"]  #if the time label of the event doesnot exist in bucket, then we create new count for this current time label of event inside the bucket
            df_tpm = pd.DataFrame(list(buckets.items()), columns=["time", "number of transactions"])
            st.line_chart(df_tpm.set_index("time"))
        else:
            st.info("waiting for the incoming transactions...")
            




    




