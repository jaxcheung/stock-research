import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import streamlit as st

from src.dashboard.ui import chart, current_theme
from src.data.fmp_client import FMPError
from src.portfolio.pnl import portfolio_summary
from src.portfolio.transactions import list_transactions, record_buy, record_sell
from src.reporting.charts import category_bars

st.set_page_config(page_title="Portfolio", page_icon="📈", layout="wide")
st.title("Portfolio & PnL")

st.markdown("### Record a Transaction")
with st.form("transaction_form", clear_on_submit=True):
    c1, c2, c3 = st.columns(3)
    txn_symbol = c1.text_input("Symbol", value="AAPL")
    txn_type = c2.selectbox("Type", ["BUY", "SELL"])
    txn_date = c3.date_input("Date", value=date.today())

    c4, c5, c6 = st.columns(3)
    quantity = c4.number_input("Quantity", min_value=0.0, step=1.0, format="%.4f")
    price = c5.number_input("Price per share", min_value=0.0, step=0.01, format="%.2f")
    fees = c6.number_input("Fees", min_value=0.0, step=0.01, format="%.2f", value=0.0)

    notes = st.text_input("Notes (optional)")
    submitted = st.form_submit_button("Record Transaction", type="primary")

    if submitted:
        try:
            record_fn = record_buy if txn_type == "BUY" else record_sell
            txn_id = record_fn(
                txn_symbol, quantity, price, txn_date.isoformat(), fees=fees, notes=notes or None
            )
            st.success(f"Recorded {txn_type} #{txn_id}: {quantity} {txn_symbol.strip().upper()} @ ${price}")
            st.session_state.pop("portfolio_summary", None)  # force PnL refresh
        except (ValueError, FMPError) as exc:
            st.error(str(exc))

st.markdown("### Positions & PnL")
if st.button("Refresh PnL") or "portfolio_summary" not in st.session_state:
    try:
        with st.spinner("Pricing open positions..."):
            st.session_state["portfolio_summary"] = portfolio_summary()
    except FMPError as exc:
        st.error(str(exc))

summary = st.session_state.get("portfolio_summary")
if summary:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cost Basis", f"${summary['totalCostBasis']:,.2f}")
    c2.metric("Market Value", f"${summary['totalMarketValue']:,.2f}")
    c3.metric("Unrealized PnL", f"${summary['totalUnrealizedPnL']:,.2f}")
    c4.metric("Realized PnL (all-time)", f"${summary['totalRealizedPnL']:,.2f}")

    if summary["positions"]:
        positions_df = pd.DataFrame(summary["positions"])
        st.dataframe(positions_df, width="stretch")

        chart(
            category_bars(
                positions_df.set_index("symbol")["marketValue"],
                current_theme(),
                "Market value by position",
                "Positions that could not be priced are omitted",
            )
        )
    else:
        st.info("No open positions.")

st.markdown("### Transaction History")
history_symbol = st.text_input("Filter by symbol (optional)", key="history_symbol_filter")
history = list_transactions(history_symbol or None)
if history:
    st.dataframe(pd.DataFrame(history), width="stretch")
else:
    st.info("No transactions recorded yet.")
