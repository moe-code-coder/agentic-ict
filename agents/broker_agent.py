"""
agents/broker_agent.py
========================
New in this rewrite — v5 never had a real execution layer (its
"shadow trading" functions existed but were dead code, see README).

Two-step design, matching the project's stated phase-1 requirement
("supervised first, autonomous later" — see /areas/agentic-trading-system.md):

    1. stage_for_human_review(verdict) — always runs. Writes the proposed
       order to a review queue. Never touches MT5's order_send.
    2. confirm_and_send(ticket_id) — the ONLY method that calls
       mt5.order_send, and it must be invoked explicitly by a human
       action (e.g. clicking "approve" in whatever review UI gets built
       on top of this), never by the Orchestrator itself.

This separation is deliberate: Orchestrator.run_symbol() only ever calls
stage_for_human_review(). If that boundary needs to move later (graduating
to autonomous execution), it should be a conscious code change to
Orchestrator, not a config flag that's easy to leave on by accident.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from ..schemas import RiskVerdict


@dataclass
class PendingOrder:
    ticket_id: str
    symbol: str
    broker_symbol: str
    direction: str          # "LONG" | "SHORT"
    entry: float
    stop: float
    target: float
    position_size: float
    risk_pct_of_equity: float
    quality_score: int | None
    execution_score: int | None
    reasons: list[str]
    staged_at: datetime
    status: str = "PENDING_HUMAN_REVIEW"  # -> "SENT" | "REJECTED" | "EXPIRED"


class BrokerAgent:
    def __init__(self):
        self._queue: dict[str, PendingOrder] = {}

    def stage_for_human_review(self, verdict: RiskVerdict) -> PendingOrder:
        if not verdict.approved:
            raise ValueError("BrokerAgent.stage_for_human_review called with an unapproved verdict.")

        signal = verdict.signal
        direction = "LONG" if signal.setup.context.weekly_bias.value == "BULLISH" else "SHORT"

        order = PendingOrder(
            ticket_id=str(uuid.uuid4()),
            symbol=verdict.symbol,
            broker_symbol=verdict.symbol,  # DataAgent's snapshot.broker_symbol should be threaded through here once available end-to-end
            direction=direction,
            entry=signal.proposed_entry,
            stop=signal.proposed_stop,
            target=signal.proposed_target,
            position_size=verdict.position_size,
            risk_pct_of_equity=verdict.risk_pct_of_equity,
            quality_score=signal.setup.quality_score,
            execution_score=signal.execution_score,
            reasons=list(verdict.reasons),
            staged_at=datetime.now(timezone.utc),
        )
        self._queue[order.ticket_id] = order
        return order

    def pending_orders(self) -> list[PendingOrder]:
        return [o for o in self._queue.values() if o.status == "PENDING_HUMAN_REVIEW"]

    def reject(self, ticket_id: str) -> None:
        self._queue[ticket_id].status = "REJECTED"

    def confirm_and_send(self, ticket_id: str) -> dict:
        """The ONLY place that touches mt5.order_send. Must be called by
        an explicit human action, never by Orchestrator."""
        if mt5 is None:
            raise RuntimeError("MetaTrader5 package not available in this environment.")
        order = self._queue.get(ticket_id)
        if order is None:
            raise KeyError(f"Unknown ticket_id: {ticket_id}")
        if order.status != "PENDING_HUMAN_REVIEW":
            raise ValueError(f"Ticket {ticket_id} is not pending review (status={order.status}).")

        order_type = mt5.ORDER_TYPE_BUY if order.direction == "LONG" else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": order.broker_symbol,
            "volume": order.position_size,
            "type": order_type,
            "sl": order.stop,
            "tp": order.target,
            "deviation": 20,
            "comment": f"agentic-ict {order.ticket_id[:8]}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        order.status = "SENT" if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE else "REJECTED"
        return {"ticket_id": ticket_id, "mt5_result": result, "status": order.status}
