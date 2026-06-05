from collections import defaultdict
import streamlit as st
import pandas as pd
from state import pause_event, state_lock, shared_state


def _snapshot_state() -> dict:
    with state_lock:
        return {
            "running":     shared_state["running"],
            "last_alert":  shared_state["last_alert"],
            "total":       shared_state["total"],
            "fraud_count": shared_state["fraud_count"],
            "legit_count": shared_state["legit_count"],
            "tpm_history": list(shared_state["tpm_history"]),
            "type_counts": {k: dict(v) for k, v in shared_state["type_counts"].items()},
            "messages":    list(shared_state["messages"]),
        }


def display_ui() -> None:
    st.title("Real-Time Fraud Detection Dashboard")

    # ── Controls ──────────────────────────────────────────────────────────────
    col1, col2, col3, _ = st.columns([1, 1, 1, 5])

    with col1:
        #keeping shared_state['running'] in sync with pause_event
        with state_lock:
            is_running = shared_state["running"]
        label = "⏸ Pause" if is_running else "▶ Resume"
        if st.button(label):
            with state_lock:
                shared_state["running"] = not shared_state["running"]
                if shared_state["running"]: #if the state is running then we set the consumer thread to run
                    pause_event.set()
                else:
                    pause_event.clear()  #otherwise we pause the consumer thread

    with col2:
        if st.button("🗑 Clear feed"):
            with state_lock:
                shared_state["messages"].clear()

    with col3:
        if st.button("↺ Reset stats"):
            with state_lock:
                shared_state.update({
                    "total": 0,
                    "fraud_count": 0,
                    "legit_count": 0,
                    "type_counts": defaultdict(lambda: {"fraud": 0, "legit": 0}),
                    "last_alert": None,
                })
                shared_state["tpm_history"].clear()

    st.divider()

    #single lock acquisition for the whole render cycle
    snap = _snapshot_state()

    # ── Fraud alert banner ────────────────────────────────────────────────────
    if snap["last_alert"]:
        txn = snap["last_alert"].get("transaction", {})
        st.error(
            f"🚨 **Fraud detected** — Account `{txn.get('nameorig', 'N/A')}` · "
            f"Type: `{txn.get('type', 'N/A')}` · "
            f"Amount: **${txn.get('amount', 0):,.2f}** · "
            f"At: {snap['last_alert'].get('received_at', '')}",
            icon="🚨",
        )

    # ── Metric cards ──────────────────────────────────────────────────────────
    total, fraud_count, legit_count = snap["total"], snap["fraud_count"], snap["legit_count"]
    fraud_rate = (fraud_count / total * 100) if total > 0 else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📊 Total", f"{total:,}")
    m2.metric("🚨 Fraudulent", f"{fraud_count:,}")
    m3.metric("✅ Legitimate", f"{legit_count:,}")
    m4.metric("📈 Fraud rate", f"{fraud_rate:.1f}%")

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Transactions per minute")
        tpm = snap["tpm_history"]
        if tpm:
            buckets: dict = {}
            for event in tpm:
                buckets[event["time"]] = buckets.get(event["time"], 0) + event["count"]
            df_tpm = pd.DataFrame(list(buckets.items()), columns=["time", "transactions"])
            st.line_chart(df_tpm.set_index("time"))
        else:
            st.info("Waiting for incoming transactions…")

    with chart_col2:
        st.subheader("Fraud vs Legitimate by transaction type")
        type_counts = snap["type_counts"]
        if type_counts:
            df_type = pd.DataFrame([
                {"type": t, "fraud": v["fraud"], "legit": v["legit"]}
                for t, v in type_counts.items()
            ]).set_index("type")
            st.bar_chart(df_type)
        else:
            st.info("Waiting for incoming transactions…")

    st.divider()

    # ── Recent transactions feed ──────────────────────────────────────────────
    st.subheader("Recent transactions feed")
    messages = snap["messages"][:50]
    if messages:
        rows = []
        for msg in messages:
            txn = msg.get("transaction", {})
            rows.append({
                "Time":        msg.get("received_at", ""),
                "Account":     txn.get("nameorig", ""),
                "Type":        txn.get("type", ""),
                "Amount ($)":  txn.get("amount", 0),
                "Result":      "🚨 Fraud" if msg.get("is_fraud") else "✅ Legit",
                "Old Balance": txn.get("oldbalanceorg", 0),
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Amount ($)":  st.column_config.NumberColumn(format="$%.2f"),
                "Old Balance": st.column_config.NumberColumn(format="$%.2f"),
                "Result":      st.column_config.TextColumn(width="small"),
            },
        )
    else:
        st.info("Waiting for messages…")