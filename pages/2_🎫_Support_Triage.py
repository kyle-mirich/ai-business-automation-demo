"""
Support Ticket Triage Demo - LangGraph multi-agent workflow
"""

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import sys

# Ensure project root on sys.path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from agents.support_agents import SupportAgentOrchestrator

load_dotenv()

st.set_page_config(
    page_title="Support Ticket Triage",
    page_icon="🎫",
    layout="wide",
)

st.title("🎫 Support Ticket Triage")
st.markdown(
    """
Experience a LangGraph-driven support workflow that classifies, prioritizes, routes,
and drafts responses for real support tickets using Google Gemini.
"""
)

st.divider()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key or api_key == "your_api_key_here":
    st.error("⚠️ Google Gemini API key not configured!")
    st.info(
        """
        To run this demo:
        1. Create a key in [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Add it to your `.env` as `GOOGLE_API_KEY=your_key`
        3. Restart the Streamlit app
        """
    )
    st.stop()

# Load support tickets
TICKETS_PATH = ROOT_DIR / "data" / "tickets.json"
if not TICKETS_PATH.exists():
    st.error(f"Missing ticket data file: {TICKETS_PATH}")
    st.stop()

with TICKETS_PATH.open("r", encoding="utf-8") as f:
    tickets_data: List[Dict] = json.load(f)

# Session state setup
if "support_orchestrator" not in st.session_state:
    st.session_state.support_orchestrator = SupportAgentOrchestrator(api_key=api_key, model="gemini-2.5-flash-lite")

if "support_results" not in st.session_state:
    st.session_state.support_results: List[Dict] = []

if "support_usage" not in st.session_state:
    st.session_state.support_usage: Dict = {"total_tokens": 0, "estimated_cost": 0.0}

# Ticket preview cards
st.subheader("Pending Tickets")
ticket_columns = st.columns(len(tickets_data))
results_map = {res["ticket"]["id"]: res for res in st.session_state.support_results}

for col, ticket in zip(ticket_columns, tickets_data):
    with col.container():
        customer = ticket.get("customer", {})
        col.markdown(f"**{customer.get('name', 'Customer')}**")
        col.caption(f"{ticket.get('subject', 'Support request')}")
        col.markdown(
            f"<div style='min-height: 120px; font-size: 0.9em; color: #334155;'>{ticket.get('message')}</div>",
            unsafe_allow_html=True,
        )
        result = results_map.get(ticket["id"])
        if result:
            col.success(
                f"Category: {result.get('category', '—')}\n\nPriority: {result.get('priority', '—')}\n\n"
                f"Department: {result.get('department', '—')}",
                icon="✅",
            )
        else:
            col.info("Awaiting processing", icon="🕒")

st.divider()

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Process Tickets")
    st.markdown(
        "Click the button below to run tickets through the LangGraph workflow. "
        "Each agent's actions will appear live while the ticket is processed."
    )

with col_right:
    process_button = st.button("🚀 Process Pending Tickets", use_container_width=True)

st.divider()

# Run workflow when requested
if process_button:
    orchestrator: SupportAgentOrchestrator = st.session_state.support_orchestrator
    orchestrator.reset_usage()
    st.session_state.support_results = []

    progress_columns = st.columns(len(tickets_data))

    for idx, (ticket, placeholder) in enumerate(zip(tickets_data, progress_columns)):
        customer_name = ticket.get("customer", {}).get("name", "Customer")
        with placeholder.status(
            label=f"Ticket {ticket['id']} • {customer_name}",
            state="running",
            expanded=True,
        ) as status:
            status.write("Classifier Agent is analyzing the message...")
            result = orchestrator.process_ticket(ticket)

            for step in result.get("steps", []):
                status.write(f"**{step['agent']}** — {step['summary']}")
                status.write(step["details"])
                tokens = step.get("token_usage")
                if tokens:
                    status.write(
                        f"Tokens: {tokens['total_tokens']} "
                        f"(prompt {tokens['prompt_tokens']}, completion {tokens['completion_tokens']})"
                    )
                time.sleep(0.3)

            status.update(
                label=f"Ticket {ticket['id']} complete",
                state="complete",
            )

        st.session_state.support_results.append(result)

    st.session_state.support_usage = orchestrator.calculate_cost()
    st.success("All tickets processed successfully!")

# Display results if available
results = st.session_state.support_results
if results:
    usage = st.session_state.support_usage

    st.subheader("Summary")
    total_tickets = len(results)
    avg_priority_score = sum(res.get("priority_score", 0) for res in results) / total_tickets

    sum_col1, sum_col2, sum_col3 = st.columns(3)
    sum_col1.metric("Tickets processed", total_tickets)
    sum_col2.metric("Average priority score", f"{avg_priority_score:.1f}")
    sum_col3.metric("Estimated Gemini cost", f"${usage.get('estimated_cost', 0.0):.4f}")

    # Distribution tables
    category_counts = Counter(res.get("category", "UNKNOWN") for res in results)
    priority_counts = Counter(res.get("priority", "UNKNOWN") for res in results)

    cat_df = pd.DataFrame(
        [{"Category": cat, "Tickets": count} for cat, count in category_counts.most_common()]
    )
    pri_df = pd.DataFrame(
        [{"Priority": pri, "Tickets": count} for pri, count in priority_counts.most_common()]
    )

    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        st.markdown("**Category Breakdown**")
        st.dataframe(cat_df, use_container_width=True, hide_index=True)
    with dist_col2:
        st.markdown("**Priority Breakdown**")
        st.dataframe(pri_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Ticket Responses")

    for res in results:
        ticket = res["ticket"]
        customer = ticket.get("customer", {})
        with st.expander(f"Ticket {ticket['id']} • {customer.get('name', 'Customer')}"):
            st.markdown(f"**Subject:** {ticket.get('subject', '')}")
            st.markdown(f"**Message:** {ticket.get('message', '')}")
            st.markdown("---")
            st.markdown(f"**Category:** {res.get('category', '—')}")
            st.markdown(f"**Priority:** {res.get('priority', '—')} (score {res.get('priority_score', 0):.0f})")
            st.markdown(f"**Department:** {res.get('department', '—')}")
            st.markdown(f"**Response:** {res.get('response', '')}")

    st.divider()
    st.subheader("Behind the Scenes")
    for res in results:
        ticket = res["ticket"]
        customer = ticket.get("customer", {})
        with st.expander(f"Workflow for Ticket {ticket['id']} • {customer.get('name', 'Customer')}"):
            for step in res.get("steps", []):
                st.markdown(f"**{step['agent']}** — {step['summary']}")
                st.markdown(step["details"])
                tokens = step.get("token_usage")
                if tokens:
                    st.caption(
                        f"Tokens: {tokens['total_tokens']} "
                        f"(prompt {tokens['prompt_tokens']}, completion {tokens['completion_tokens']})"
                    )
                st.markdown("---")

