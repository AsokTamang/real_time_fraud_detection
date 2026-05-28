import os
import json
import signal
import time
import threading
from collections import deque, defaultdict
from datetime import datetime

import streamlit as st
from confluent_kafka import Consumer, Producer, KafkaException, KafkaError

from kafka_client import (
    producer_config,
    consumer_config,
    FRAUD_RESULT_TOPIC,
    DLQ_TOPIC,
    delivery_report,
)
from src.logger import logging


# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Monitor",
    page_icon="🛡️",
    layout="wide",
)

# ─────────────────────────────────────────────
#  Session-state initialisation
#  (persists across Streamlit reruns)
# ─────────────────────────────────────────────
if "messages"       not in st.session_state: st.session_state.messages       = deque(maxlen=200)
if "total"          not in st.session_state: st.session_state.total          = 0
if "fraud_count"    not in st.session_state: st.session_state.fraud_count    = 0
if "legit_count"    not in st.session_state: st.session_state.legit_count    = 0
if "type_counts"    not in st.session_state: st.session_state.type_counts    = defaultdict(lambda: {"fraud": 0, "legit": 0})
if "tpm_history"    not in st.session_state: st.session_state.tpm_history    = deque(maxlen=20)   # transactions-per-poll
if "running"        not in st.session_state: st.session_state.running        = True
if "consumer_thread"not in st.session_state: st.session_state.consumer_thread= None
if "last_alert"     not in st.session_state: st.session_state.last_alert     = None




# ─────────────────────────────────────────────
#  Background Kafka consumer thread
#  Reads messages and appends to session-state
# ─────────────────────────────────────────────
def kafka_consumer_thread():
    consumer    = Consumer(consumer_config)
    dlq_producer = Producer(producer_config)

    try:
        consumer.subscribe([FRAUD_RESULT_TOPIC])
        logging.info(f"Dashboard consumer subscribed to {FRAUD_RESULT_TOPIC}")

        while st.session_state.running:
            msg = consumer.poll(1.0)

            if msg is None:
                continue

            # ── Kafka-level errors ──────────────────────────────────────────
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logging.info(
                        f"End of partition: {msg.topic()} [{msg.partition()}]"
                    )
                else:
                    logging.error(f"Kafka error: {msg.error()}")
                    send_to_dlq(dlq_producer, msg.value(), f"Kafka error: {msg.error()}")
                continue

            # ── Parse message ───────────────────────────────────────────────
            try:
                raw_value     = msg.value()
                message_value = (
                    json.loads(raw_value.decode("utf-8")) if raw_value else None
                )
                if message_value is None:
                    continue

                # ── Update shared state ─────────────────────────────────────
                tx      = message_value.get("transaction", {})
                result  = message_value.get("result", "Unknown")
                is_fraud= message_value.get("is_fraud", False)
                tx_type = tx.get("type", "UNKNOWN")

                enriched = {
                    **message_value,
                    "received_at": datetime.now().strftime("%H:%M:%S"),
                    "partition"  : msg.partition(),
                    "offset"     : msg.offset(),
                }

                st.session_state.messages.appendleft(enriched)
                st.session_state.total       += 1
                st.session_state.tpm_history.append(
                    {"time": datetime.now().strftime("%H:%M:%S"), "count": 1}
                )

                if is_fraud:
                    st.session_state.fraud_count += 1
                    st.session_state.type_counts[tx_type]["fraud"] += 1
                    st.session_state.last_alert = enriched          # trigger banner
                else:
                    st.session_state.legit_count += 1
                    st.session_state.type_counts[tx_type]["legit"] += 1

                consumer.commit(asynchronous=False)
                logging.info(
                    f"Processed message | result={result} | "
                    f"partition={msg.partition()} offset={msg.offset()}"
                )

            except Exception as e:
                logging.error(f"Error processing message: {e}")
                send_to_dlq(dlq_producer, msg.value(), f"Processing error: {e}")

    except KafkaException as ke:
        logging.error(f"Kafka exception in dashboard consumer: {ke}")

    finally:
        consumer.close()
        dlq_producer.flush()
        logging.info("Dashboard consumer closed.")


# ─────────────────────────────────────────────
#  Start background thread once per session
# ─────────────────────────────────────────────
if (
    st.session_state.consumer_thread is None
    or not st.session_state.consumer_thread.is_alive()
):
    t = threading.Thread(target=kafka_consumer_thread, daemon=True)
    t.start()
    st.session_state.consumer_thread = t


# ─────────────────────────────────────────────
#  ─── UI ────────────────────────────────────
# ─────────────────────────────────────────────

st.title("🛡️ Real-Time Fraud Detection Monitor")
st.caption(f"Kafka topic: `{FRAUD_RESULT_TOPIC}` · auto-refreshes every 2 s")

# ── Controls ────────────────────────────────────────────────────────────────
col_ctrl1, col_ctrl2, col_ctrl3, _ = st.columns([1, 1, 1, 5])
with col_ctrl1:
    if st.button("⏸ Pause" if st.session_state.running else "▶ Resume"):
        st.session_state.running = not st.session_state.running
with col_ctrl2:
    if st.button("🗑 Clear feed"):
        st.session_state.messages.clear()
with col_ctrl3:
    if st.button("↺ Reset stats"):
        st.session_state.total       = 0
        st.session_state.fraud_count = 0
        st.session_state.legit_count = 0
        st.session_state.type_counts = defaultdict(lambda: {"fraud": 0, "legit": 0})
        st.session_state.tpm_history.clear()
        st.session_state.last_alert  = None

st.divider()

# ── Fraud alert banner ───────────────────────────────────────────────────────
if st.session_state.last_alert:
    alert = st.session_state.last_alert
    tx    = alert.get("transaction", {})
    st.error(
        f"🚨 **Fraud detected** — Account `{tx.get('nameorig', 'N/A')}` · "
        f"Type: `{tx.get('type', 'N/A')}` · "
        f"Amount: **${tx.get('amount', 0):,.2f}** · "
        f"At: {alert.get('received_at', '')}",
        icon="🚨",
    )

# ── Metric cards ─────────────────────────────────────────────────────────────
total       = st.session_state.total
fraud_count = st.session_state.fraud_count
legit_count = st.session_state.legit_count
fraud_rate  = (fraud_count / total * 100) if total > 0 else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("📊 Total transactions", f"{total:,}")
m2.metric("🚨 Fraudulent",         f"{fraud_count:,}",  delta=None)
m3.metric("✅ Legitimate",          f"{legit_count:,}",  delta=None)
m4.metric("📈 Fraud rate",          f"{fraud_rate:.1f}%")

st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Transaction volume over time")
    if st.session_state.tpm_history:
        import pandas as pd

        # bucket counts by second label
        buckets: dict = {}
        for entry in st.session_state.tpm_history:
            buckets[entry["time"]] = buckets.get(entry["time"], 0) + entry["count"]

        df_tpm = pd.DataFrame(
            list(buckets.items()), columns=["time", "transactions"]
        ).tail(20)
        st.line_chart(df_tpm.set_index("time"))
    else:
        st.info("Waiting for messages…")

with chart_col2:
    st.subheader("Fraud vs legit by transaction type")
    type_counts = st.session_state.type_counts
    if type_counts:
        import pandas as pd

        rows = [
            {"type": t, "fraud": v["fraud"], "legit": v["legit"]}
            for t, v in type_counts.items()
        ]
        df_types = pd.DataFrame(rows).set_index("type")
        st.bar_chart(df_types)
    else:
        st.info("Waiting for messages…")

st.divider()

# ── Live transaction feed ─────────────────────────────────────────────────────
st.subheader("Live transaction feed")
st.caption(f"{len(st.session_state.messages)} messages in buffer (max 200)")

messages = list(st.session_state.messages)[:50]    # show latest 50

if messages:
    import pandas as pd

    rows = []
    for m in messages:
        tx      = m.get("transaction", {})
        is_fraud= m.get("is_fraud", False)
        rows.append({
            "Time"       : m.get("received_at", ""),
            "Account"    : tx.get("nameorig", "N/A"),
            "Type"       : tx.get("type", "N/A"),
            "Amount ($)" : tx.get("amount", 0),
            "Old balance": tx.get("oldbalanceorg", 0),
            "Destination": tx.get("namedest", "N/A"),
            "Result"     : "🚨 Fraud" if is_fraud else "✅ Legit",
            "Partition"  : m.get("partition", ""),
            "Offset"     : m.get("offset", ""),
        })

    df_feed = pd.DataFrame(rows)
    st.dataframe(
        df_feed,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Amount ($)" : st.column_config.NumberColumn(format="$%.2f"),
            "Old balance": st.column_config.NumberColumn(format="$%.2f"),
            "Result"     : st.column_config.TextColumn(width="small"),
        },
    )
else:
    st.info("No messages yet. Waiting for Kafka messages on the topic…")

# ── Auto-refresh every 2 seconds ─────────────────────────────────────────────
time.sleep(2)
st.rerun()