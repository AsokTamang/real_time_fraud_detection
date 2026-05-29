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
    #DISPLAY OF FRAUD ALERT BANNER IN THE DASHBOARD WHEN THE FRAUD TRANSACTION IS DETECTED BY OUR MODEL 
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
    
    #DISPLAY OF metric cards for showing the total number of transactions, fraud transactions, legit transactions and fraud rate percentage
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
    #DISPLAY OF TRANSACTION PER MINUTE CHART
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Transactions per minute")
        buckets:dict = {}
        if st.session_state.tpm_history:
            for event in st.session_state.tpm_history:
                buckets[event["time"]] = buckets.get(event["time"], 0) + event["count"]  #if the time label of the event doesnot exist in bucket, then we create new count for this current time label of event inside the bucket
            df_tpm = pd.DataFrame(list(buckets.items()), columns=["time", "number of transactions"])
            st.line_chart(df_tpm.set_index("time"))  #setting the time label as x-axis
        else:
            st.info("waiting for the incoming transactions...")
    
    with chart_col2:
        st.subheader('Fraud vs Legitimate transaction by transaction type')
        if st.session_state.type_counts:
            datas = [{'type':t, 'fraud':v['fraud'], 'legit':v['legit']} for t,v in st.session_state.type_counts.items()]  #preparing the data for the bar chart to show the count of fraud and legit transactions for each type of transaction
            df_type = pd.DataFrame(datas).set_index('type') #setting the type on x-axis
            st.bar_chart(df_type)  
        else:
            st.info("waiting for the incoming transactions...")    

    st.divider()
    #DISPLAY OF RECENT TRANSACTIONS FEED IN THE DASHBOARD
    st.subheader("Recent transactions feed")
    if st.session_state.messages:
        messages = list(st.session_state.messages)[:50] #getting the latest 50 messages stored in the session state to show it in the dashboard as a feed
        rows = []
        for msg in messages:
            transaction = msg.get("transaction", {})  #as the incoming transaction details was stored inside the key called transaction in the payload produced by the producer
            is_fraud      = msg.get("is_fraud", False)    #as the prediction result was stored inside the key called result in the payload produced by the producer
            rows.append({
                "time": msg.get("received_at", ""),
                "account": transaction.get("nameorig", ""),
                "type": transaction.get("type", ""),
                "Amount ($)": f"${transaction.get('amount', 0):,.2f}",
                "Result": "🚨 Fraud" if is_fraud else "✅ Legit",
                "Old balance": f"${transaction.get('oldbalanceorg', 0):,.2f}"
            })
        df_messages = pd.DataFrame(rows)
        st.dataframe(
            df_messages,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Amount ($)" : st.column_config.NumberColumn(format="$%.2f"),
                "Old balance": st.column_config.NumberColumn(format="$%.2f"),
                "Result"     : st.column_config.TextColumn(width="small"),
            },
        )
    else:
        st.info("Waiting for messages…")





    




