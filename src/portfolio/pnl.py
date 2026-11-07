# src/portfolio/pnl.py
#
# FIFO cost-basis accounting: the oldest open BUY lot is consumed first when
# a SELL is matched against it. Realized PnL is computed per SELL transaction
# at match time; open lots left after all SELLs determine each position's
# current cost basis for unrealized PnL.

from collections import deque
from pathlib import Path
from typing import Any

from src.config import PORTFOLIO_QTY_TOLERANCE
from src.data.fmp_client import FMPClient, FMPError, normalize_symbol
from src.portfolio.db import get_connection
from src.portfolio.transactions import list_transactions
from src.utils.logging import get_logger

logger = get_logger("portfolio.pnl")


class _Lot:
    __slots__ = ("quantity", "cost_per_share", "buy_date")

    def __init__(self, quantity: float, cost_per_share: float, buy_date: str):
        self.quantity = quantity
        self.cost_per_share = cost_per_share
        self.buy_date = buy_date


def _list_symbols(db_path: Path | None = None) -> list[str]:
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT DISTINCT symbol FROM transactions").fetchall()
    return [row["symbol"] for row in rows]


def compute_fifo(symbol: str, db_path: Path | None = None) -> dict[str, Any]:
    """FIFO-matched realized trades and remaining open lots for one symbol."""
    symbol = normalize_symbol(symbol)
    transactions = list_transactions(symbol, db_path=db_path)

    open_lots: deque[_Lot] = deque()
    realized_trades: list[dict[str, Any]] = []

    for txn in transactions:
        if txn["transaction_type"] == "BUY":
            cost_per_share = (txn["quantity"] * txn["price"] + txn["fees"]) / txn["quantity"]
            open_lots.append(_Lot(txn["quantity"], cost_per_share, txn["transaction_date"]))
        else:  # SELL
            remaining = txn["quantity"]
            proceeds_per_share = (txn["quantity"] * txn["price"] - txn["fees"]) / txn["quantity"]
            cost_basis = 0.0
            proceeds = 0.0

            while remaining > PORTFOLIO_QTY_TOLERANCE and open_lots:
                lot = open_lots[0]
                matched = min(lot.quantity, remaining)
                cost_basis += matched * lot.cost_per_share
                proceeds += matched * proceeds_per_share
                lot.quantity -= matched
                remaining -= matched
                if lot.quantity <= PORTFOLIO_QTY_TOLERANCE:
                    open_lots.popleft()

            if remaining > PORTFOLIO_QTY_TOLERANCE:
                logger.warning(
                    "SELL of %s %s on %s exceeds tracked open lots by %s "
                    "(transactions recorded before this symbol's history began?)",
                    txn["quantity"],
                    symbol,
                    txn["transaction_date"],
                    remaining,
                )

            realized_trades.append(
                {
                    "transactionId": txn["id"],
                    "date": txn["transaction_date"],
                    "quantity": txn["quantity"],
                    "proceeds": proceeds,
                    "costBasis": cost_basis,
                    "realizedPnL": proceeds - cost_basis,
                    # > 0 only for legacy rows recorded before timeline validation; such a
                    # trade's PnL covers the matched shares only.
                    "unmatchedQuantity": remaining if remaining > PORTFOLIO_QTY_TOLERANCE else 0.0,
                }
            )

    open_lots_out = [
        {"buyDate": lot.buy_date, "quantity": lot.quantity, "costPerShare": lot.cost_per_share}
        for lot in open_lots
    ]
    return {"symbol": symbol, "realizedTrades": realized_trades, "openLots": open_lots_out}


def get_open_positions(db_path: Path | None = None) -> list[dict[str, Any]]:
    """Current open position per symbol: quantity held and FIFO cost basis."""
    positions = []
    for symbol in _list_symbols(db_path):
        fifo = compute_fifo(symbol, db_path=db_path)
        quantity = sum(lot["quantity"] for lot in fifo["openLots"])
        if quantity <= PORTFOLIO_QTY_TOLERANCE:
            continue
        cost_basis = sum(lot["quantity"] * lot["costPerShare"] for lot in fifo["openLots"])
        positions.append(
            {
                "symbol": symbol,
                "quantity": quantity,
                "costBasis": cost_basis,
                "avgCostPerShare": cost_basis / quantity,
            }
        )
    return positions


def get_total_realized_pnl(db_path: Path | None = None) -> float:
    total = 0.0
    for symbol in _list_symbols(db_path):
        fifo = compute_fifo(symbol, db_path=db_path)
        total += sum(t["realizedPnL"] for t in fifo["realizedTrades"])
    return total


def portfolio_summary(client: FMPClient | None = None, db_path: Path | None = None) -> dict[str, Any]:
    """Open positions with live unrealized PnL, plus all-time realized PnL totals."""
    client = client or FMPClient()
    positions = get_open_positions(db_path=db_path)

    enriched = []
    total_cost_basis = 0.0
    total_market_value = 0.0
    for pos in positions:
        try:
            price = client.get_quote(pos["symbol"]).get("price")
        except FMPError as exc:
            logger.warning("Could not price %s for unrealized PnL: %s", pos["symbol"], exc)
            price = None

        market_value = price * pos["quantity"] if price is not None else None
        unrealized_pnl = market_value - pos["costBasis"] if market_value is not None else None

        total_cost_basis += pos["costBasis"]
        if market_value is not None:
            total_market_value += market_value

        enriched.append(
            {
                **pos,
                "currentPrice": price,
                "marketValue": market_value,
                "unrealizedPnL": unrealized_pnl,
            }
        )

    total_realized_pnl = get_total_realized_pnl(db_path=db_path)
    total_unrealized_pnl = sum(p["unrealizedPnL"] for p in enriched if p["unrealizedPnL"] is not None)

    return {
        "positions": enriched,
        "totalCostBasis": total_cost_basis,
        "totalMarketValue": total_market_value,
        "totalUnrealizedPnL": total_unrealized_pnl,
        "totalRealizedPnL": total_realized_pnl,
        "totalPnL": total_unrealized_pnl + total_realized_pnl,
    }
