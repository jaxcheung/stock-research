import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st

from src.knowledge.notes import add_quote, delete_quote, get_random_quote, list_quotes

st.set_page_config(page_title="Knowledge Log", page_icon="📈", layout="wide")
st.title("Knowledge Log")
st.caption("Quotes and notes from books, fundamental analysis, Warren Buffett, and your own research.")

col_quote, col_refresh = st.columns([5, 1])
with col_quote:
    random_quote = get_random_quote()
    if random_quote:
        attribution = " — " + random_quote["author"] if random_quote["author"] else ""
        st.info(f"“{random_quote['text']}”{attribution}")
    else:
        st.info("No quotes logged yet — add your first one below.")
with col_refresh:
    st.button("New quote")  # button click alone triggers a rerun, refreshing the random pick above

st.markdown("### Add a Quote or Note")
with st.form("add_quote_form", clear_on_submit=True):
    text = st.text_area("Text", placeholder="Price is what you pay. Value is what you get.")
    c1, c2, c3 = st.columns(3)
    author = c1.text_input("Author (optional)", placeholder="Warren Buffett")
    source = c2.text_input("Source (optional)", placeholder="2008 Letter to Shareholders")
    category = c3.text_input("Category (optional)", placeholder="valuation / mindset / fundamentals / macro")
    c4, c5 = st.columns(2)
    symbol = c4.text_input("Related ticker (optional)", placeholder="AAPL")
    context = c5.text_input("Your context/notes (optional)")

    submitted = st.form_submit_button("Save", type="primary")
    if submitted:
        try:
            quote_id = add_quote(
                text,
                author=author or None,
                source=source or None,
                category=category or None,
                symbol=symbol or None,
                context=context or None,
            )
            st.success(f"Saved quote #{quote_id}")
        except ValueError as exc:
            st.error(str(exc))

st.markdown("### Browse")
f1, f2, f3, f4 = st.columns(4)
filter_category = f1.text_input("Filter: category")
filter_author = f2.text_input("Filter: author")
filter_symbol = f3.text_input("Filter: ticker")
filter_search = f4.text_input("Search text")

quotes = list_quotes(
    category=filter_category or None,
    author=filter_author or None,
    symbol=filter_symbol or None,
    search=filter_search or None,
)

if quotes:
    st.caption(f"{len(quotes)} quote(s)")
    for q in quotes:
        with st.container(border=True):
            attribution = " — " + q["author"] if q["author"] else ""
            st.markdown(f"“{q['text']}”{attribution}")
            meta_bits = [b for b in [q["source"], q["category"], q["symbol"]] if b]
            if meta_bits:
                st.caption(" · ".join(meta_bits))
            if q["context"]:
                st.caption(f"Note: {q['context']}")
            if st.button("Delete", key=f"delete_{q['id']}"):
                delete_quote(q["id"])
                st.rerun()
else:
    st.info("No quotes match these filters.")
