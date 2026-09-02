"""
agents/risk_agent.py
======================
Layer 4: the only agent allowed to size a trade or approve/reject it
against portfolio-level exposure. Everything here (calculate_position_size,
PortfolioRiskConfig, portfolio_risk_snapshot, approve_portfolio_entry) is
ported from fetch_forex_data_v5_FINAL.py's phase-19+ risk engine — which
existed in the old file but was NEVER CALLED from __main__ (see README:
"Port map" — this was the largest chunk of dead code found in the audit).
The logic itself was sound; it just wasn't wired in. It is wired in here.

correlation_adjusted_risk() is NOT yet ported — it needs a live
cross-symbol correlation matrix, which requires running the orchestrator
across the whole symbol universe first and feeding the result back in.
Until that exists, RiskAgent behaves as if correlation_matrix=None (v5's
own fallback path — a explicit, not a silent approximation).
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from ..schemas import ExecutionSignal, RiskVerdict, Decision


@dataclass
class PortfolioRiskConfig:
    max_open_risk_pct: float = 3.0
    max_symbol_risk_pct: float = 1.0
    max_directional_risk_pct: float = 2.0
    max_open_positions: int = 5
    correlation_threshold: float = 0.70
    correlation_penalty: float = 0.50
    max_symbol_exposure_pct: float | None = None

    def validate(self):
        if self.max_open_risk_pct <= 0:
            raise ValueError("max_open_risk_pct must be > 0")
        if self.max_symbol_risk_pct <= 0:
            raise ValueError("max_symbol_risk_pct must be > 0")
        if self.max_directional_risk_pct <= 0:
            raise ValueError("max_directional_risk_pct must be > 0")
        if self.max_open_positions <= 0:
            raise ValueError("max_open_positions must be > 0")
        if not 0 <= self.correlation_threshold <= 1:
            raise ValueError("correlation_threshold must be between 0 and 1")
        if not 0 <= self.correlation_penalty <= 1:
            raise ValueError("correlation_penalty must be between 0 and 1")


def calculate_position_size(equity: float, risk_pct: float, entry_price: float,
                             stop_price: float, value_per_price_unit: float = 1.0,
                             min_size: float = 0.0, max_size: float | None = None) -> dict:
    """Sizes one position from an explicit monetary risk budget.
    value_per_price_unit is broker/instrument-specific (contract size x
    tick value) and must be supplied by the caller — this deliberately
    does not guess pip/lot economics."""
    equity, risk_pct = float(equity), float(risk_pct)
    entry_price, stop_price = float(entry_price), float(stop_price)
    value_per_price_unit = float(value_per_price_unit)

    if equity <= 0:
        raise ValueError("equity must be > 0")
    if risk_pct <= 0:
        raise ValueError("risk_pct must be > 0")
    if entry_price <= 0 or stop_price <= 0:
        raise ValueError("prices must be > 0")
    if value_per_price_unit <= 0:
        raise ValueError("value_per_price_unit must be > 0")

    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        raise ValueError("entry_price and stop_price must differ")

    risk_amount = equity * risk_pct / 100.0
    raw_size = risk_amount / (stop_distance * value_per_price_unit)
    size = max(float(min_size), raw_size)
    if max_size is not None:
        size = min(size, float(max_size))

    actual_risk = size * stop_distance * value_per_price_unit
    return {
        "equity": equity, "risk_pct_budget": risk_pct, "risk_amount_budget": risk_amount,
        "entry_price": entry_price, "stop_price": stop_price, "stop_distance": stop_distance,
        "value_per_price_unit": value_per_price_unit, "position_size": size,
        "actual_risk_amount": actual_risk, "actual_risk_pct": actual_risk / equity * 100.0,
    }


def portfolio_risk_snapshot(open_positions: pd.DataFrame | None, equity: float,
                             config: PortfolioRiskConfig | None = None) -> dict:
    config = config or PortfolioRiskConfig()
    config.validate()
    if equity <= 0:
        raise ValueError("equity must be > 0")

    if open_positions is None or open_positions.empty:
        return {
            "Open_Positions": 0, "Open_Risk_Pct": 0.0, "Long_Risk_Pct": 0.0,
            "Short_Risk_Pct": 0.0, "Max_Symbol_Risk_Pct": 0.0,
            "Risk_Budget_Remaining_Pct": config.max_open_risk_pct,
        }

    frame = open_positions.copy()
    frame["Risk_Pct"] = pd.to_numeric(frame.get("Risk_Pct", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    frame["Direction"] = frame.get("Direction", pd.Series("", index=frame.index)).astype(str).str.upper()
    frame["Symbol"] = frame.get("Symbol", pd.Series("", index=frame.index)).astype(str)

    open_risk = float(frame["Risk_Pct"].sum())
    long_risk = float(frame.loc[frame["Direction"] == "LONG", "Risk_Pct"].sum())
    short_risk = float(frame.loc[frame["Direction"] == "SHORT", "Risk_Pct"].sum())
    symbol_risk = frame.groupby("Symbol")["Risk_Pct"].sum()
    max_symbol = float(symbol_risk.max()) if len(symbol_risk) else 0.0

    return {
        "Open_Positions": int(len(frame)), "Open_Risk_Pct": open_risk,
        "Long_Risk_Pct": long_risk, "Short_Risk_Pct": short_risk,
        "Max_Symbol_Risk_Pct": max_symbol,
        "Risk_Budget_Remaining_Pct": max(0.0, config.max_open_risk_pct - open_risk),
        "Max_Open_Risk_Pct": config.max_open_risk_pct,
        "Max_Directional_Risk_Pct": config.max_directional_risk_pct,
        "Max_Open_Positions": config.max_open_positions,
    }


def approve_portfolio_entry(open_positions: pd.DataFrame | None, equity: float, symbol: str,
                             direction: str, risk_pct: float,
                             correlation_matrix: pd.DataFrame | None = None,
                             config: PortfolioRiskConfig | None = None) -> dict:
    """Hard admission gate. Does not mutate portfolio state. NOTE:
    correlation_matrix is currently always None (see module docstring) —
    this runs the same explicit no-correlation-data fallback v5 used,
    not a silent approximation."""
    config = config or PortfolioRiskConfig()
    config.validate()
    direction = str(direction).upper()
    if direction not in {"LONG", "SHORT"}:
        raise ValueError("direction must be LONG or SHORT")
    if not symbol:
        raise ValueError("symbol must not be empty")

    snapshot = portfolio_risk_snapshot(open_positions, equity, config)
    reasons = []
    if snapshot["Open_Positions"] >= config.max_open_positions:
        reasons.append("max_open_positions")

    symbol_risk = directional_risk = 0.0
    if open_positions is not None and not open_positions.empty:
        frame = open_positions.copy()
        frame["Risk_Pct"] = pd.to_numeric(frame.get("Risk_Pct", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
        frame["Direction"] = frame.get("Direction", pd.Series("", index=frame.index)).astype(str).str.upper()
        frame["Symbol"] = frame.get("Symbol", pd.Series("", index=frame.index)).astype(str)
        symbol_risk = float(frame.loc[frame["Symbol"] == symbol, "Risk_Pct"].sum())
        directional_risk = float(frame.loc[frame["Direction"] == direction, "Risk_Pct"].sum())

    if symbol_risk + float(risk_pct) > config.max_symbol_risk_pct + 1e-12:
        reasons.append("max_symbol_risk")
    if directional_risk + float(risk_pct) > config.max_directional_risk_pct + 1e-12:
        reasons.append("max_directional_risk")

    # correlation_matrix is always None today (see module docstring) —
    # this branch is v5's own explicit no-data fallback.
    effective_added_risk_pct = float(risk_pct)
    effective_open_risk = snapshot["Open_Risk_Pct"] + effective_added_risk_pct
    if effective_open_risk > config.max_open_risk_pct + 1e-12:
        reasons.append("max_effective_open_risk")

    return {
        "approved": len(reasons) == 0,
        "symbol": symbol, "direction": direction, "requested_risk_pct": float(risk_pct),
        "symbol_risk_before_pct": symbol_risk, "directional_risk_before_pct": directional_risk,
        "effective_added_risk_pct": effective_added_risk_pct,
        "effective_open_risk_after_pct": effective_open_risk,
        "reasons": reasons,
    }


class RiskAgent:
    """Holds the portfolio state (equity + open positions) across calls —
    this is per-account state, not per-symbol, unlike the other agents."""

    def __init__(self, get_equity, get_open_positions, value_per_price_unit_map: dict,
                 risk_pct_per_trade: float = 1.0, config: PortfolioRiskConfig | None = None):
        """
        get_equity: callable() -> float (reads live account equity, e.g. from MT5).
        get_open_positions: callable() -> pd.DataFrame with columns
            [Symbol, Direction(LONG/SHORT), Risk_Pct] — one row per open ticket.
        value_per_price_unit_map: {symbol: value_per_price_unit} — broker-specific,
            NOT guessed (see calculate_position_size docstring).
        """
        self.get_equity = get_equity
        self.get_open_positions = get_open_positions
        self.value_per_price_unit_map = value_per_price_unit_map
        self.risk_pct_per_trade = risk_pct_per_trade
        self.config = config or PortfolioRiskConfig()

    def evaluate(self, signal: ExecutionSignal) -> RiskVerdict:
        reasons: list[str] = []
        equity = self.get_equity()
        open_positions = self.get_open_positions()

        direction = "LONG" if signal.setup.context.weekly_bias.value == "BULLISH" else "SHORT"
        value_per_price_unit = self.value_per_price_unit_map.get(signal.symbol)

        if value_per_price_unit is None:
            reasons.append(f"No value_per_price_unit configured for {signal.symbol} — cannot size safely.")
            return RiskVerdict(
                symbol=signal.symbol, knowledge_time=signal.knowledge_time, signal=signal,
                position_size=None, risk_pct_of_equity=None, portfolio_correlation_ok=False,
                approved=False, decision=Decision.INVALID, reasons=reasons,
            )

        sizing = calculate_position_size(
            equity=equity, risk_pct=self.risk_pct_per_trade,
            entry_price=signal.proposed_entry, stop_price=signal.proposed_stop,
            value_per_price_unit=value_per_price_unit,
        )

        admission = approve_portfolio_entry(
            open_positions=open_positions, equity=equity, symbol=signal.symbol,
            direction=direction, risk_pct=sizing["actual_risk_pct"],
            correlation_matrix=None,  # see module docstring
            config=self.config,
        )
        reasons.extend(admission["reasons"])

        approved = admission["approved"]
        return RiskVerdict(
            symbol=signal.symbol,
            knowledge_time=signal.knowledge_time,
            signal=signal,
            position_size=sizing["position_size"],
            risk_pct_of_equity=sizing["actual_risk_pct"],
            portfolio_correlation_ok=True,  # no correlation data yet — see docstring
            approved=approved,
            requires_human_confirmation=True,  # hard-coded during the supervised phase
            decision=Decision.TRADE_SIGNAL_CANDIDATE_PENDING_RISK if approved else Decision.WAIT,
            reasons=reasons if reasons else ["Portfolio admission passed; position sized."],
        )
