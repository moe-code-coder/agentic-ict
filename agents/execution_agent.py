"""
agents/execution_agent.py
===========================
Layer 3 of the ICT prompt's decision model: LTF trigger hierarchy
(M30 context -> M15 primary trigger -> M5 precision, never overriding
HTF), session/liquidity quality, and the trade plan (entry/stop/target
+ RR). This is the last agent before RiskAgent — it can still WAIT or
INVALID a candidate, but the moment it emits TRADE_SIGNAL_CANDIDATE it
means "the full non-risk gate hierarchy passed" (prompt Sec. 24).

Ported unchanged from fetch_forex_data_v5_FINAL.py: assess_liquidity(),
infer_ltf_context(), estimate_trade_plan(), structural_rr(),
execution_score(). EXECUTION_MIN and MIN_RR are the same hard gates
v5 used (7/10 and 3.0R) — changing these is a strategy decision, not
a refactor, so they're called out explicitly rather than buried.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..schemas import MarketSnapshot, SetupState, ExecutionSignal, Decision
from ..primitives import latest_closed_row

MIN_RR = 3.0
EXECUTION_MIN = 7
STOP_ATR_BUFFER_MULT = 0.25
SPREAD_LOOKBACK = 100
SPREAD_ACCEPTABLE_MULT = 1.5


def structural_rr(direction: str, entry: float, stop: float, target: float) -> float:
    if any(pd.isna(x) for x in [entry, stop, target]):
        return np.nan
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk <= 0:
        return np.nan
    return reward / risk


def estimate_trade_plan(direction: str, zone_row, h4_row, sweep_level: float) -> dict:
    """Sec. 2 rule preserved: UNKNOWN/np.nan fields rather than inventing
    values when a required input is missing. Entry = zone midpoint (or
    last close). Stop = beyond the invalidating sweep swing + small ATR
    buffer. Target = nearest opposing confirmed pivot, else dealing-range
    edge."""
    result = {"Entry": np.nan, "Stop": np.nan, "Target": np.nan, "RR": np.nan}
    if h4_row is None or direction not in ("BUY", "SELL"):
        return result

    atr = h4_row.get("ATR", np.nan)

    entry = np.nan
    if zone_row is not None:
        top = zone_row.get("FVG_Top", np.nan)
        bottom = zone_row.get("FVG_Bottom", np.nan)
        if pd.isna(top) or pd.isna(bottom):
            top = zone_row.get("OB_Top", np.nan)
            bottom = zone_row.get("OB_Bottom", np.nan)
        if pd.notna(top) and pd.notna(bottom):
            entry = (top + bottom) / 2
    if pd.isna(entry):
        entry = h4_row.get("Close", np.nan)

    stop = np.nan
    if pd.notna(sweep_level) and pd.notna(atr):
        buffer = STOP_ATR_BUFFER_MULT * atr
        stop = sweep_level - buffer if direction == "BUY" else sweep_level + buffer

    target = np.nan
    if direction == "BUY":
        ph = h4_row.get("Confirmed_Pivot_High", np.nan)
        range_high = h4_row.get("Dealing_Range_High", np.nan)
        if pd.notna(ph) and pd.notna(entry) and ph > entry:
            target = ph
        elif pd.notna(range_high) and pd.notna(entry) and range_high > entry:
            target = range_high
    else:
        pl = h4_row.get("Confirmed_Pivot_Low", np.nan)
        range_low = h4_row.get("Dealing_Range_Low", np.nan)
        if pd.notna(pl) and pd.notna(entry) and pl < entry:
            target = pl
        elif pd.notna(range_low) and pd.notna(entry) and range_low < entry:
            target = range_low

    rr = structural_rr(direction, entry, stop, target)
    result.update({"Entry": entry, "Stop": stop, "Target": target, "RR": rr})
    return result


def assess_liquidity(df: pd.DataFrame) -> str:
    """Compares latest closed bar's spread to its own recent rolling
    median, instead of a hardcoded 'Good' (v5's original bug)."""
    if df is None or df.empty or "Spread" not in df.columns:
        return "Unknown"
    row = latest_closed_row(df)
    if row is None:
        return "Unknown"
    closed = df[~df["Is_Current_Bar"].astype(bool)] if "Is_Current_Bar" in df.columns else df
    if closed.empty:
        return "Unknown"
    recent = closed["Spread"].tail(SPREAD_LOOKBACK)
    median_spread = recent.median()
    current_spread = row.get("Spread", np.nan)
    if pd.isna(median_spread) or pd.isna(current_spread) or median_spread <= 0:
        return "Unknown"
    return "Good" if current_spread <= SPREAD_ACCEPTABLE_MULT * median_spread else "Poor"


def _direction_word(weekly_bias_code: str) -> str:
    return {"BUY": "Bullish", "SELL": "Bearish"}.get(weekly_bias_code, "")


def infer_ltf_context(m30: pd.DataFrame, m15: pd.DataFrame, m5: pd.DataFrame,
                       weekly_bias_code: str = "NEUTRAL"):
    """M30=context, M15=primary trigger, M5=precision-only — none of them
    may create a thesis the HTF doesn't support (prompt Sec. 16); a
    confirmed LTF event against the HTF thesis is 'Conflicting', not a
    trigger, however clean it looks on its own timeframe."""
    rows = {"M30": latest_closed_row(m30), "M15": latest_closed_row(m15), "M5": latest_closed_row(m5)}
    thesis_dir = _direction_word(weekly_bias_code)

    m30_state = "Neutral"
    if rows["M30"] is not None:
        r = rows["M30"]
        d = r.get("BOS_Direction", "") or r.get("MSS_Direction", "")
        if d in ("Bullish", "Bearish") and thesis_dir:
            m30_state = "Aligned" if d == thesis_dir else "Conflicting"

    m15_state = "Pending"
    if rows["M15"] is not None:
        r = rows["M15"]
        d = r.get("BOS_Direction", "") or r.get("MSS_Direction", "")
        confirmed = r.get("MSS_Event", "") == "Confirmed" or r.get("BOS_Event", "") == "Confirmed"
        if confirmed:
            m15_state = "Conflicting" if (thesis_dir and d and d != thesis_dir) else "Confirmed"

    m5_state = "Optional"
    if rows["M5"] is not None:
        r = rows["M5"]
        d = r.get("BOS_Direction", "") or r.get("MSS_Direction", "")
        confirmed = r.get("MSS_Event", "") == "Confirmed" or r.get("BOS_Event", "") == "Confirmed"
        if confirmed:
            m5_state = "Conflicting" if (thesis_dir and d and d != thesis_dir) else "Confirmed"

    return m30_state, m15_state, m5_state


def execution_score(m30: str, m15: str, m5: str, session: str, liquidity: str) -> int:
    s = 0
    if m30 == "Aligned":
        s += 2
    elif m30 == "Neutral":
        s += 1
    if m15 == "Confirmed":
        s += 4
    elif m15 == "Developing":
        s += 2
    if m5 == "Confirmed":
        s += 2
    elif m5 == "Supportive":
        s += 1
    if session == "Preferred":
        s += 1
    if liquidity == "Good":
        s += 1
    return int(min(10, s))


class ExecutionAgent:
    def evaluate(self, setup: SetupState, snapshot: MarketSnapshot) -> ExecutionSignal:
        reasons: list[str] = []
        weekly_bias_code = {"BULLISH": "BUY", "BEARISH": "SELL"}.get(setup.context.weekly_bias.value, "NEUTRAL")

        m30, m15, m5 = snapshot.frames.get("M30"), snapshot.frames.get("M15"), snapshot.frames.get("M5")
        m30_state, m15_state, m5_state = infer_ltf_context(m30, m15, m5, weekly_bias_code)

        h4 = snapshot.frames.get("H4")
        session = h4["Session_Priority"].iloc[-1] if h4 is not None and "Session_Priority" in h4 else "Other_Acceptable"
        session_quality = h4["Session_Quality"].iloc[-1] if h4 is not None and "Session_Quality" in h4 else "Acceptable"
        liquidity = assess_liquidity(h4) if h4 is not None else "Unknown"

        exec_score = execution_score(m30_state, m15_state, m5_state, session_quality, liquidity)

        if m30_state == "Conflicting":
            reasons.append("M30 context conflicts with weekly thesis — execution withheld (Sec. 16).")
            return ExecutionSignal(
                symbol=snapshot.symbol, knowledge_time=snapshot.knowledge_time, setup=setup,
                m30_context=m30_state, m15_trigger=False, m5_precision=False,
                session_ok=session_quality != "Low_Priority", liquidity_ok=liquidity == "Good",
                execution_score=exec_score, decision=Decision.WAIT, reasons=reasons,
            )
        if m15_state != "Confirmed":
            reasons.append(f"M15 primary trigger not yet confirmed (state={m15_state}).")
            return ExecutionSignal(
                symbol=snapshot.symbol, knowledge_time=snapshot.knowledge_time, setup=setup,
                m30_context=m30_state, m15_trigger=False, m5_precision=(m5_state == "Confirmed"),
                session_ok=session_quality != "Low_Priority", liquidity_ok=liquidity == "Good",
                execution_score=exec_score, decision=Decision.WAIT, reasons=reasons,
            )

        # Build the trade plan. Needs the H4 sweep level (invalidation
        # point) and the qualified zone — recover them the same way
        # SetupAgent found them (re-deriving here is cheap and keeps
        # ExecutionAgent decoupled from SetupAgent's internals).
        sweeps = h4[(h4["Sweep_Event"] == "Confirmed") & (h4["Sweep_Direction"] ==
                    _direction_word(weekly_bias_code))]
        sweep_level = sweeps.iloc[-1]["Sweep_Level"] if not sweeps.empty else np.nan
        h4_row = latest_closed_row(h4)

        zone_row = {"FVG_Top": setup.entry_zone_high, "FVG_Bottom": setup.entry_zone_low} \
            if setup.zone_type == "FVG" else {"OB_Top": setup.entry_zone_high, "OB_Bottom": setup.entry_zone_low}

        plan = estimate_trade_plan(weekly_bias_code, zone_row, h4_row, sweep_level)
        rr = plan["RR"]

        if pd.isna(rr) or rr < MIN_RR:
            reasons.append(f"RR gate failed: {rr!r} < required {MIN_RR} (Sec. 12/24).")
            return ExecutionSignal(
                symbol=snapshot.symbol, knowledge_time=snapshot.knowledge_time, setup=setup,
                m30_context=m30_state, m15_trigger=True, m5_precision=(m5_state == "Confirmed"),
                session_ok=session_quality != "Low_Priority", liquidity_ok=liquidity == "Good",
                execution_score=exec_score, structural_rr=None if pd.isna(rr) else float(rr),
                decision=Decision.INVALID, reasons=reasons,
            )

        if exec_score < EXECUTION_MIN:
            reasons.append(f"Execution score {exec_score} below minimum {EXECUTION_MIN}.")
            return ExecutionSignal(
                symbol=snapshot.symbol, knowledge_time=snapshot.knowledge_time, setup=setup,
                m30_context=m30_state, m15_trigger=True, m5_precision=(m5_state == "Confirmed"),
                session_ok=session_quality != "Low_Priority", liquidity_ok=liquidity == "Good",
                execution_score=exec_score, structural_rr=float(rr), decision=Decision.WAIT, reasons=reasons,
            )

        reasons.append(f"All execution gates passed (state S6): exec_score={exec_score}, RR={rr:.2f}.")
        return ExecutionSignal(
            symbol=snapshot.symbol,
            knowledge_time=snapshot.knowledge_time,
            setup=setup,
            m30_context=m30_state,
            m15_trigger=True,
            m5_precision=(m5_state == "Confirmed"),
            session_ok=session_quality != "Low_Priority",
            liquidity_ok=liquidity == "Good",
            execution_score=exec_score,
            proposed_entry=float(plan["Entry"]) if pd.notna(plan["Entry"]) else None,
            proposed_stop=float(plan["Stop"]) if pd.notna(plan["Stop"]) else None,
            proposed_target=float(plan["Target"]) if pd.notna(plan["Target"]) else None,
            structural_rr=float(rr),
            decision=Decision.TRADE_SIGNAL_CANDIDATE_PENDING_RISK,
            reasons=reasons,
        )
