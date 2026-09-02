"""
schemas.py
==========
Shared data contracts between agents.

Design principle (from Professional_ICT_Trading_Prompt_v4):
    Context -> Setup -> Execution -> Risk, strictly one-directional.
    A lower layer may WATCHLIST/WAIT/REJECT, but can never overrule
    an invalidation set by a higher layer. These dataclasses encode
    that boundary: each agent consumes only the layer(s) above it and
    produces exactly one typed object for the layer below it.

Nothing here talks to MT5, pandas, or an LLM. Pure data.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Direction(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class Decision(str, Enum):
    WATCHLIST = "WATCHLIST"
    WAIT = "WAIT"
    INVALID = "INVALID"
    TRADE_SIGNAL_CANDIDATE_PENDING_RISK = "TRADE_SIGNAL_CANDIDATE_PENDING_RISK"


# ---------------------------------------------------------------------------
# Layer 0 — raw market data (produced by DataAgent)
# ---------------------------------------------------------------------------

@dataclass
class MarketSnapshot:
    """One symbol's multi-timeframe OHLCV + indicators, as of a knowledge-time.

    knowledge_time is the timestamp this snapshot is allowed to "know about" —
    every downstream agent must filter its own view of history to
    time <= knowledge_time. This is how we keep the whole system
    look-ahead-safe by construction instead of by convention.
    """
    symbol: str
    broker_symbol: str          # the exact wmmarket ticker, e.g. "XAUUSD" (TBD — confirm in MT5 Market Watch)
    knowledge_time: datetime
    frames: dict                # {"M5": DataFrame, "M15": ..., ..., "MN1": ...}


# ---------------------------------------------------------------------------
# Layer 1 — Context (produced by ContextAgent, consumed by SetupAgent)
# ---------------------------------------------------------------------------

@dataclass
class ContextView:
    symbol: str
    knowledge_time: datetime
    d1_regime: str                       # "Trending" | "Ranging" | "Indeterminate"
    weekly_bias: Direction
    directional_permission: bool         # False -> hard stop, nothing below may proceed
    htf_institutional_map_notes: str = ""
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Layer 2 — Setup (produced by SetupAgent, consumed by ExecutionAgent)
# ---------------------------------------------------------------------------

@dataclass
class SetupState:
    symbol: str
    knowledge_time: datetime
    context: ContextView
    h4_sweep_confirmed: bool
    mss_confirmed: bool
    bos_confirmed: bool
    entry_zone_low: Optional[float]
    entry_zone_high: Optional[float]
    zone_type: Optional[str]             # "FVG" | "OB" | None
    valid_retracement: bool
    quality_score: Optional[int] = None  # ranks candidates; NEVER authorizes a trade on its own
    decision: Decision = Decision.WATCHLIST
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Layer 3 — Execution (produced by ExecutionAgent, consumed by RiskAgent)
# ---------------------------------------------------------------------------

@dataclass
class ExecutionSignal:
    symbol: str
    knowledge_time: datetime
    setup: SetupState
    m30_context: str                     # "Aligned" | "Conflicting" | "Neutral"
    m15_trigger: bool
    m5_precision: bool
    session_ok: bool
    liquidity_ok: bool
    execution_score: Optional[int] = None
    proposed_entry: Optional[float] = None
    proposed_stop: Optional[float] = None
    proposed_target: Optional[float] = None
    structural_rr: Optional[float] = None
    decision: Decision = Decision.WAIT
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Layer 4 — Risk (produced by RiskAgent, consumed by BrokerAgent / human)
# ---------------------------------------------------------------------------

@dataclass
class RiskVerdict:
    symbol: str
    knowledge_time: datetime
    signal: ExecutionSignal
    position_size: Optional[float]
    risk_pct_of_equity: Optional[float]
    portfolio_correlation_ok: bool
    approved: bool
    requires_human_confirmation: bool = True   # hard-coded True until the human-supervision phase is graduated
    decision: Decision = Decision.WAIT
    reasons: list[str] = field(default_factory=list)
