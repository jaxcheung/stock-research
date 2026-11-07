import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

PAGES = Path(__file__).resolve().parent / "pages"

st.set_page_config(page_title="Stock Research", page_icon="📈", layout="wide")


def home() -> None:
    from src.dashboard import ui
    from src.data.fmp_client import FMPClient, FMPError
    from src.data.fred_client import FREDClient

    ui.setup_page("Stock Research")
    ui.page_header(
        "Personal research terminal",
        "Stock Research & Asset Management",
        "Fundamental equity research, quantitative macro analysis and portfolio tracking — every number with "
        "its methodology, significance and data-quality caveats.",
    )
    if "symbol" not in st.session_state:
        st.session_state["symbol"] = "AAPL"
    st.text_input("Default ticker (shared across pages)", key="symbol")

    sections = {
        "Research": [
            (
                "Stock Analysis",
                "10-year return, risk and valuation indicators (incl. gearing), statements, price risk.",
            ),
            (
                "Industry Comparison",
                "Return vs risk vs peers over time, industry aggregates, cap-size filters.",
            ),
            ("DCF Valuation", "Two-stage FCF model with CAPM WACC and live risk-free rate."),
            ("Research Report", "Full equity research report: scorecard, DCF + comps target, risks."),
        ],
        "Quant Lab": [
            (
                "Macro Lab",
                "Multi-variable statistics: stationarity, correlation, lead/lag, regression, cycles, forecast.",
            ),
            (
                "Rate Sensitivity",
                "Industry cost of debt vs a macro rate — pass-through and rate-shock scenarios.",
            ),
            ("Relative Valuation", "P/E vs industry, sector, peers, market and own history."),
            (
                "Fundamentals Explorer",
                "10-year income-statement panel (Parquet + DuckDB): sector trends, leaderboard, SQL.",
            ),
            (
                "Backtest Lab",
                "Point-in-time factor and sector-ETF backtests vs SPY, with deflated Sharpe and walk-forward.",
            ),
        ],
        "Portfolio": [
            ("Portfolio", "Record transactions; FIFO realised and live unrealised PnL."),
            ("Knowledge Log", "Investing quotes and lessons."),
        ],
    }
    for name, items in sections.items():
        st.markdown(f"#### {name}")
        ui.bullet_card([f"**{title}** — {desc}" for title, desc in items])

    st.markdown("#### Data connections")
    status = []
    try:
        FMPClient()
        status.append("**FMP** — connected (Starter tier).")
    except FMPError as exc:
        status.append(f"**FMP** — not configured: {exc}")
    fred = FREDClient()
    status.append(
        "**FRED** — "
        + (
            "API key (JSON API)."
            if not fred.keyless
            else "keyless public CSV (set FRED_API_KEY for the official API)."
        )
    )
    status.append("**World Bank** — keyless public API.")
    ui.bullet_card(status)


nav = st.navigation(
    {
        "Home": [st.Page(home, title="Home", icon="🏠", default=True)],
        "Research": [
            st.Page(PAGES / "1_Stock_Analysis.py", title="Stock Analysis", icon="🔎"),
            st.Page(PAGES / "2_Industry_Comparison.py", title="Industry Comparison", icon="🏭"),
            st.Page(PAGES / "3_DCF_Valuation.py", title="DCF Valuation", icon="💵"),
            st.Page(PAGES / "6_Research_Report.py", title="Research Report", icon="📑"),
        ],
        "Quant Lab": [
            st.Page(PAGES / "7_Macro_Lab.py", title="Macro Lab", icon="🧪"),
            st.Page(PAGES / "8_Rate_Sensitivity.py", title="Rate Sensitivity", icon="🏦"),
            st.Page(PAGES / "9_Relative_Valuation.py", title="Relative Valuation", icon="⚖️"),
            st.Page(PAGES / "10_Fundamentals_Explorer.py", title="Fundamentals Explorer", icon="🗄️"),
            st.Page(PAGES / "11_Backtest_Lab.py", title="Backtest Lab", icon="🧮"),
        ],
        "Portfolio": [
            st.Page(PAGES / "4_Portfolio.py", title="Portfolio", icon="💼"),
            st.Page(PAGES / "5_Knowledge_Log.py", title="Knowledge Log", icon="📚"),
        ],
    }
)
nav.run()
