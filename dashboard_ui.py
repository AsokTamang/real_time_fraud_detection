from collections import defaultdict
import streamlit as st
import pandas as pd
from state import pause_event, state_lock, shared_state


def display_ui():
    st.title("Real-Time Fraud Detection Dashboard")
    col1, col2, col3, _ = st.columns([1, 1, 1, 5])
    with col1:
        if st.button("⏸ Pause" if shared_state['running'] else "▶ Resume"):
            with state_lock:  # acquiring the lock to check and update the running state of consumer thread in a thread safe way
                shared_state['running'] = not shared_state['running']
                if shared_state['running']:
                    pause_event.set()  # Resuming the consumer thread if session state is running
                else:
                    pause_event.clear()  # Pausing the consumer thread if session state is not running
    with col2:
        if st.button("🗑 Clear feed"):
            with state_lock:
                shared_state['messages'].clear()  #clearing all the stored transaction details in the state
    with col3:
        if st.button("↺ Reset stats"):
            with state_lock:
                shared_state['total'] = 0
                shared_state['fraud_count'] = 0
                shared_state['legit_count'] = 0
                shared_state['type_counts'] = defaultdict(
                    lambda: {"fraud": 0, "legit": 0}
                )
                shared_state['tpm_history'].clear()
                shared_state['last_alert'] = None

    st.divider()  # for better UI
    # DISPLAY OF FRAUD ALERT BANNER IN THE DASHBOARD WHEN THE FRAUD TRANSACTION IS DETECTED BY OUR MODEL
    with state_lock:
        last_alert = shared_state['last_alert']
    if last_alert:
        incoming_transaction = last_alert.get(
            "transaction", {}
        )  # as the incoming transaction details was stored inside the key called transaction in the payload produced by the producer
        st.error(
            f"🚨 **Fraud detected** — Account `{incoming_transaction.get('nameorig', 'N/A')}` · "
            f"Type: `{incoming_transaction.get('type', 'N/A')}` · "
            f"Amount: **${incoming_transaction.get('amount', 0):,.2f}** · "
            f"At: {last_alert.get('received_at', '')}",  # this is the time when the message is received by the consumer from the kafka topic
            icon="🚨",
        )
    with state_lock:  # acquiring the lock to read the shared state variables in a thread safe way
        # DISPLAY OF metric cards for showing the total number of transactions, fraud transactions, legit transactions and fraud rate percentage
        total = shared_state['total']
        fraud_count = shared_state['fraud_count']
        legit_count = shared_state['legit_count']
        fraud_rate = (
            (fraud_count / total * 100) if total > 0 else 0.0
        )  # calculation of fraud transaction rate percentage
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📊 Total transactions", f"{total:,}")
    m2.metric("🚨 Fraudulent", f"{fraud_count:,}", delta=None)
    m3.metric("✅ Legitimate", f"{legit_count:,}", delta=None)
    m4.metric("📈 Fraud rate", f"{fraud_rate:.1f}")

    st.divider()
    # DISPLAY OF TRANSACTION PER MINUTE CHART
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Transactions per minute")
        buckets: dict = {}
        with state_lock:  # acquiring the lock to read the shared state variables in a thread safe way
            tpm_snapshot = list(
                shared_state['tpm_history']
            )  # getting the transactions per minute history stored in the session state to show it in the dashboard as a line chart
        if tpm_snapshot:
            for (
                event
            ) in (
                tpm_snapshot
            ):  # as we have stored the transactions per minute history in the session state as a list of dict with time and count keys, we are iterating over this list to prepare the data for line chart to show the total count of transactions received in each minute
                #we are storing the count of transactions for each time labels in the bucket dictionary based on the time label as key and the count of transaction as value
                # here we are displaying the total count of transactions recieved in each minute
                buckets[event["time"]] = (
                    buckets.get(event["time"], 0) + event["count"]
                )  # if the time label of the event doesnot exist in bucket, then we create new count for this current time label of event inside the bucket
            df_tpm = pd.DataFrame(
                list(buckets.items()), columns=["time", "number of transactions"]
            )
            st.line_chart(df_tpm.set_index("time"))  # setting the time label as x-axis
        else:
            st.info("waiting for the incoming transactions...")

    with chart_col2:
        st.subheader("Fraud vs Legitimate transaction by transaction type")
        with state_lock:
            type_counts_snapshot = dict(
                shared_state['type_counts']
            )  # getting the transaction type counts for fraud and legit transactions stored in the session state to show it in the dashboard as a bar chart
        if type_counts_snapshot:
            datas = [
                {"type": t, "fraud": v["fraud"], "legit": v["legit"]}
                for t, v in type_counts_snapshot.items()
            ]  # preparing the data for the bar chart to show the count of fraud and legit transactions for each type of transaction
            df_type = pd.DataFrame(datas).set_index(
                "type"
            )  # setting the type on x-axis
            st.bar_chart(df_type)
        else:
            st.info("waiting for the incoming transactions...")

    st.divider()
    # DISPLAY OF RECENT TRANSACTIONS FEED IN THE DASHBOARD
    st.subheader("Recent transactions feed")
    with state_lock:
        messages_snapshot = list(
            shared_state['messages']
        )  # getting the recent transactions feed stored in the session state to show it in the dashboard as a table
    if messages_snapshot:
        messages = list(messages_snapshot)[
            :50
        ]  # getting the latest 50 messages stored in the session state to show it in the dashboard as a feed as we have stored the messages from left side
        rows = []
        for msg in messages:
            transaction = msg.get(
                "transaction", {}
            )  # as the incoming transaction details was stored inside the key called transaction in the payload produced by the producer
            is_fraud = msg.get(
                "is_fraud", False
            )  # as whether the incoming transaction is fraud or not, is represented by the key called is_fraud which is stored inside the payload produced by the producer
            rows.append(
                {
                    "time": msg.get("received_at", ""),
                    "account": transaction.get("nameorig", ""),
                    "type": transaction.get("type", ""),
                    "Amount ($)": f"${transaction.get('amount', 0):,.2f}",
                    "Result": "🚨 Fraud" if is_fraud else "✅ Legit",
                    "Old balance": f"${transaction.get('oldbalanceorg', 0):,.2f}",
                }
            )
        df_messages = pd.DataFrame(rows)
        st.dataframe(
            df_messages,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Amount ($)": st.column_config.NumberColumn(format="$%.2f"),
                "Old balance": st.column_config.NumberColumn(format="$%.2f"),
                "Result": st.column_config.TextColumn(width="small"),
            },
        )
    else:
        st.info("Waiting for messages…")
