"""
agents/setup_agent.py
=======================
Layer 2 of the ICT prompt's decision model: "Has the required
institutional sequence actually occurred?" Consumes the ContextAgent's
verdict (already gated — this agent never runs if directional_permission
is False) plus the pre-enriched H4/D1 frames, and produces a SetupState.

Ported unchanged from fetch_forex_data_v5_FINAL.py: find_latest_zone(),
zone_hit(), score_weekly_bias(), score_quality(), quality_band(),
determine_state() (the S0-S6 state machine).

H4's Sweep/MSS/BOS/FVG/OB columns are already computed by
primitives.enrich_frame() inside DataAgent — this agent reads them, it
does not recompute structure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..schemas import MarketSnapshot, ContextView, SetupState, Direction, Decision

QUALITY_WEIGHTS = {
    "weekly_bias": 15, "sweep": 20, "mss": 20, "bos": 15,
    "htf_zone": 10, "premium_discount": 10, "session": 5, "freshness_coherence": 5,
}


def score_weekly_bias(direction: str, state: str) -> int:
    return QUALITY_WEIGHTS["weekly_bias"] if state == "Confirmed" and direction in ("BUY", "SELL") else 0


def score_quality(weekly_bias: str, weekly_state: str, sweep: str, mss: str, bos: str,
                   zone_quality: str, pd_zone: str, session_quality: str, freshness: str) -> int:
    score = score_weekly_bias(weekly_bias, weekly_state)
    if sweep == "Confirmed":
        score += QUALITY_WEIGHTS["sweep"]
    if mss == "Confirmed":
        score += QUALITY_WEIGHTS["mss"]
    if bos == "Confirmed":
        score += QUALITY_WEIGHTS["bos"]
    if zone_quality == "Primary":
        score += QUALITY_WEIGHTS["htf_zone"]
    elif zone_quality == "Secondary":
        score += QUALITY_WEIGHTS["htf_zone"] // 2
    if (weekly_bias == "BUY" and pd_zone == "Discount") or (weekly_bias == "SELL" and pd_zone == "Premium"):
        score += QUALITY_WEIGHTS["premium_discount"]
    if session_quality == "Preferred":
        score += QUALITY_WEIGHTS["session"]
    elif session_quality == "Acceptable":
        score += QUALITY_WEIGHTS["session"] // 2
    if freshness == "Fresh":
        score += QUALITY_WEIGHTS["freshness_coherence"]
    elif freshness == "Coherent":
        score += QUALITY_WEIGHTS["freshness_coherence"] // 2
    return int(min(100, score))


def quality_band(score: int) -> str:
    if score >= 85:
        return "A+"
    if score >= 75:
        return "A"
    if score >= 65:
        return "B"
    if score >= 50:
        return "Monitor"
    return "Background"


def determine_state(weekly_bias: str, weekly_state: str, sweep: str, mss: str, bos: str,
                     zone_reached: bool, m30: str, m15: str, invalidated: bool = False) -> str:
    if invalidated:
        return "SX"
    if weekly_bias not in ("BUY", "SELL") or weekly_state != "Confirmed":
        return "S0"
    if sweep != "Confirmed":
        return "S1"
    if mss != "Confirmed":
        return "S2"
    if bos != "Confirmed":
        return "S3"
    if not zone_reached:
        return "S4"
    if m30 == "Conflicting" or m15 != "Confirmed":
        return "S5"
    return "S6"


def find_latest_zone(df: pd.DataFrame, direction: str, current_time: pd.Timestamp):
    """BUGFIX (v4.1, preserved): an Order Block is stored on its ORIGIN
    candle but only becomes knowable at OB_Event_Time — both FVG and OB
    are aligned to a single "confirmed at" timestamp before filtering,
    so nothing here can be found before it was actually confirmed."""
    df = df.copy()
    df["_Zone_Confirmed_At"] = df["FVG_Event_Time"].fillna(df["OB_Event_Time"])

    eligible = df[df["_Zone_Confirmed_At"].notna() & (df["_Zone_Confirmed_At"] <= current_time)]

    if direction == "BUY":
        candidates = eligible[(eligible["FVG_Type"] == "Bullish") | (eligible["OB_Type"] == "Bullish")]
    else:
        candidates = eligible[(eligible["FVG_Type"] == "Bearish") | (eligible["OB_Type"] == "Bearish")]

    if candidates.empty:
        return None

    fresh = candidates[candidates["FVG_Status"] != "Fully_Mitigated"]
    pool = fresh if not fresh.empty else candidates
    pool = pool.sort_values("_Zone_Confirmed_At")
    return pool.iloc[-1]


def zone_hit(row, price: float) -> bool:
    if row is None:
        return False
    top, bottom = row.get("FVG_Top", np.nan), row.get("FVG_Bottom", np.nan)
    if pd.notna(top) and pd.notna(bottom) and bottom <= price <= top:
        return True
    top, bottom = row.get("OB_Top", np.nan), row.get("OB_Bottom", np.nan)
    if pd.notna(top) and pd.notna(bottom) and bottom <= price <= top:
        return True
    return False


def _direction_word(weekly_bias) -> str:
    return {Direction.BULLISH: "Bullish", Direction.BEARISH: "Bearish"}.get(weekly_bias, "")


def _latest_row_asof(df: pd.DataFrame, event_time: pd.Timestamp):
    if df is None or df.empty or pd.isna(event_time):
        return None
    x = df[df["time"] <= event_time]
    return x.iloc[-1] if not x.empty else None


class SetupAgent:
    def evaluate(self, context: ContextView, snapshot: MarketSnapshot) -> SetupState:
        h4 = snapshot.frames.get("H4")
        d1 = snapshot.frames.get("D1")
        reasons: list[str] = []

        if h4 is None or h4.empty:
            return SetupState(
                symbol=snapshot.symbol, knowledge_time=snapshot.knowledge_time, context=context,
                h4_sweep_confirmed=False, mss_confirmed=False, bos_confirmed=False,
                entry_zone_low=None, entry_zone_high=None, zone_type=None,
                valid_retracement=False, decision=Decision.WATCHLIST,
                reasons=["No H4 data available."],
            )

        t = h4["time"].iloc[-1]
        thesis_dir = _direction_word(context.weekly_bias)
        weekly_bias_code = {"BULLISH": "BUY", "BEARISH": "SELL"}.get(context.weekly_bias.value, "NEUTRAL")

        d1_row = _latest_row_asof(d1, t) if d1 is not None else None
        pd_zone = d1_row.get("PD_Zone", "Uncertain") if d1_row is not None else "Uncertain"

        # Sweep -> MSS -> BOS must be the same directional thesis and in order.
        sweeps = h4[(h4["Sweep_Event"] == "Confirmed") & h4["Sweep_Time"].notna() & (h4["Sweep_Time"] <= t)]
        sweep = sweeps.iloc[-1] if not sweeps.empty else None
        sweep_ok = sweep is not None and sweep.get("Sweep_Direction", "") == thesis_dir
        if not sweep_ok:
            reasons.append("No confirmed H4 sweep matching weekly thesis direction (state S1).")
            return SetupState(
                symbol=snapshot.symbol, knowledge_time=snapshot.knowledge_time, context=context,
                h4_sweep_confirmed=False, mss_confirmed=False, bos_confirmed=False,
                entry_zone_low=None, entry_zone_high=None, zone_type=None,
                valid_retracement=False, decision=Decision.WATCHLIST, reasons=reasons,
            )

        mss_rows = h4[(h4["MSS_Event"] == "Confirmed") & h4["MSS_Time"].notna()
                      & (h4["MSS_Time"] <= t) & (h4["MSS_Time"] >= sweep["Sweep_Time"])
                      & (h4["MSS_Direction"] == thesis_dir)]
        mss = mss_rows.iloc[-1] if not mss_rows.empty else None
        if mss is None:
            reasons.append("Sweep confirmed but no MSS yet (state S2).")
            return SetupState(
                symbol=snapshot.symbol, knowledge_time=snapshot.knowledge_time, context=context,
                h4_sweep_confirmed=True, mss_confirmed=False, bos_confirmed=False,
                entry_zone_low=None, entry_zone_high=None, zone_type=None,
                valid_retracement=False, decision=Decision.WATCHLIST, reasons=reasons,
            )

        bos_rows = h4[(h4["BOS_Event"] == "Confirmed") & h4["BOS_Time"].notna()
                      & (h4["BOS_Time"] <= t) & (h4["BOS_Time"] >= mss["MSS_Time"])
                      & (h4["BOS_Direction"] == thesis_dir)]
        bos = bos_rows.iloc[-1] if not bos_rows.empty else None
        if bos is None:
            reasons.append("MSS confirmed but no BOS yet (state S3).")
            return SetupState(
                symbol=snapshot.symbol, knowledge_time=snapshot.knowledge_time, context=context,
                h4_sweep_confirmed=True, mss_confirmed=True, bos_confirmed=False,
                entry_zone_low=None, entry_zone_high=None, zone_type=None,
                valid_retracement=False, decision=Decision.WATCHLIST, reasons=reasons,
            )

        zone = find_latest_zone(h4, weekly_bias_code, t)
        current_price = float(h4["Close"].iloc[-1])
        reached = zone_hit(zone, current_price)
        if zone is None or not reached:
            reasons.append("Sweep/MSS/BOS confirmed but price hasn't reached an entry zone yet (state S4).")
            return SetupState(
                symbol=snapshot.symbol, knowledge_time=snapshot.knowledge_time, context=context,
                h4_sweep_confirmed=True, mss_confirmed=True, bos_confirmed=True,
                entry_zone_low=None, entry_zone_high=None, zone_type=None,
                valid_retracement=False, decision=Decision.WATCHLIST, reasons=reasons,
            )

        zone_type = "FVG" if pd.notna(zone.get("FVG_Type", np.nan)) else "OB"
        low = zone.get("FVG_Bottom") if zone_type == "FVG" else zone.get("OB_Bottom")
        high = zone.get("FVG_Top") if zone_type == "FVG" else zone.get("OB_Top")

        zone_quality = zone.get("OB_Quality", "")
        if zone_quality not in ("Primary", "Secondary"):
            zone_quality = "Invalid"

        freshness = "Unknown"
        if zone.get("FVG_Status", "") == "Unmitigated":
            freshness = "Fresh"
        elif zone.get("FVG_Status", "") == "Partially_Mitigated":
            freshness = "Coherent"

        session_quality = h4["Session_Quality"].iloc[-1] if "Session_Quality" in h4 else "Acceptable"

        score = score_quality(
            weekly_bias_code, "Confirmed", "Confirmed", "Confirmed", "Confirmed",
            zone_quality, pd_zone, session_quality, freshness,
        )

        # All four gates passed at H4 -> candidate goes to ExecutionAgent
        # for the M30/M15/M5 trigger. It is NOT yet a trade signal
        # (quality_score ranks candidates, per prompt Sec. 0, it never
        # authorizes on its own).
        return SetupState(
            symbol=snapshot.symbol,
            knowledge_time=snapshot.knowledge_time,
            context=context,
            h4_sweep_confirmed=True,
            mss_confirmed=True,
            bos_confirmed=True,
            entry_zone_low=float(low) if pd.notna(low) else None,
            entry_zone_high=float(high) if pd.notna(high) else None,
            zone_type=zone_type,
            valid_retracement=True,
            quality_score=score,
            decision=Decision.WAIT,  # handed to ExecutionAgent, not yet a signal
            reasons=[f"Setup complete (S4 reached), quality_band={quality_band(score)}."],
        )
