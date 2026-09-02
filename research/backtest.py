"""
research/backtest.py
======================
Offline event-driven H4 backtest. NOT part of the live agent loop —
imports from agents/ (read-only reuse of scoring/planning logic), never
the other way around.

CRITICAL FIX vs v5: v5's `_backtest_setup_state` read `zone.get("FVG_Status")`
directly from an H4 frame whose FVG_Status column was computed ONCE, globally,
using `primitives.detect_fvg()`'s reverse cummin/cummax over the ENTIRE
dataset. That column encodes "was this zone eventually mitigated by the end
of the whole fetched history" — not "was it mitigated as of the moment being
backtested". Reusing it while simulating decisions at an earlier bar i leaks
information from bars after i into the eligibility check, inflating apparent
edge. This module recomputes mitigation honestly, from only the bars up to
the simulated "now" (see `mitigation_status_asof` below).

Everything else (simulate_trade, backtest_symbol structure, performance
stats) is ported from v5 unchanged — those were already causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..agents.setup_agent import score_quality, quality_band, find_latest_zone, zone_hit
from ..agents.execution_agent import (
    estimate_trade_plan, execution_score, infer_ltf_context, assess_liquidity, MIN_RR, EXECUTION_MIN,
)
from ..primitives import latest_closed_row

BACKTEST_HORIZON_BARS = 100


def _latest_row_asof(df: pd.DataFrame, event_time: pd.Timestamp):
    if df is None or df.empty or pd.isna(event_time):
        return None
    x = df[df["time"] <= event_time]
    return x.iloc[-1] if not x.empty else None


def mitigation_status_asof(h4: pd.DataFrame, zone_row, t: pd.Timestamp) -> str:
    """THE FIX: mitigation status of an FVG zone using only bars strictly
    after the zone's confirmation and at/before t — never bars after t.
    Mirrors the classification rule in primitives.detect_fvg (fully vs.
    partially vs. not mitigated, relative to the zone midpoint), just
    computed on a truncated, time-bounded window instead of the whole
    dataset."""
    if zone_row is None:
        return "Unknown"

    zone_type = "FVG" if pd.notna(zone_row.get("FVG_Type", np.nan)) else "OB"
    top = zone_row.get("FVG_Top") if zone_type == "FVG" else zone_row.get("OB_Top")
    bottom = zone_row.get("FVG_Bottom") if zone_type == "FVG" else zone_row.get("OB_Bottom")
    confirmed_at = zone_row.get("_Zone_Confirmed_At")
    if pd.isna(top) or pd.isna(bottom) or pd.isna(confirmed_at):
        return "Unknown"

    window = h4[(h4["time"] > confirmed_at) & (h4["time"] <= t)]
    if window.empty:
        return "Unmitigated"

    direction = "Bullish" if zone_row.get("FVG_Type") == "Bullish" or zone_row.get("OB_Type") == "Bullish" else "Bearish"
    mid = (top + bottom) / 2.0

    if direction == "Bullish":
        min_low = window["Low"].min()
        if min_low <= bottom:
            return "Fully_Mitigated"
        if min_low <= mid:
            return "Partially_Mitigated"
        return "Unmitigated"
    else:
        max_high = window["High"].max()
        if max_high >= top:
            return "Fully_Mitigated"
        if max_high >= mid:
            return "Partially_Mitigated"
        return "Unmitigated"


def find_latest_zone_asof(h4: pd.DataFrame, direction: str, current_time: pd.Timestamp):
    """Same eligibility rule as agents.setup_agent.find_latest_zone (a zone
    must be confirmed at/before current_time), but freshness is decided
    with mitigation_status_asof() instead of the globally-computed
    FVG_Status column."""
    h4 = h4.copy()
    h4["_Zone_Confirmed_At"] = h4["FVG_Event_Time"].fillna(h4["OB_Event_Time"])
    eligible = h4[h4["_Zone_Confirmed_At"].notna() & (h4["_Zone_Confirmed_At"] <= current_time)]

    if direction == "BUY":
        candidates = eligible[(eligible["FVG_Type"] == "Bullish") | (eligible["OB_Type"] == "Bullish")]
    else:
        candidates = eligible[(eligible["FVG_Type"] == "Bearish") | (eligible["OB_Type"] == "Bearish")]
    if candidates.empty:
        return None

    candidates = candidates.copy()
    candidates["_Status_Asof"] = candidates.apply(
        lambda r: mitigation_status_asof(h4, r, current_time), axis=1
    )
    fresh = candidates[candidates["_Status_Asof"] != "Fully_Mitigated"]
    pool = fresh if not fresh.empty else candidates
    pool = pool.sort_values("_Zone_Confirmed_At")
    return pool.iloc[-1]


def evaluate_setup_asof(h4: pd.DataFrame, w1: pd.DataFrame, d1: pd.DataFrame,
                         m30: pd.DataFrame, m15: pd.DataFrame, m5: pd.DataFrame,
                         i: int, min_rr: float = MIN_RR, execution_min: int = EXECUTION_MIN):
    """Causal replacement for v5's _backtest_setup_state. Sweep/MSS/BOS,
    session, PD-zone, and Weekly Bias columns were already computed
    causally by primitives.enrich_frame(), so those are read directly
    (same as v5). Only the zone-freshness check is redone here to close
    the mitigation leak (see module docstring)."""
    r = h4.iloc[i]
    t = r["time"]

    w = _latest_row_asof(w1, t)
    weekly_bias = w.get("Weekly_Bias", "NEUTRAL") if w is not None else "NEUTRAL"
    weekly_state = w.get("Weekly_Bias_State", "Unconfirmed") if w is not None else "Unconfirmed"
    d = _latest_row_asof(d1, t)
    pd_zone = d.get("PD_Zone", "Uncertain") if d is not None else "Uncertain"
    d1_regime = d.get("Regime", "Indeterminate") if d is not None else "Indeterminate"

    if weekly_bias not in ("BUY", "SELL") or weekly_state != "Confirmed":
        return None
    thesis_dir = {"BUY": "Bullish", "SELL": "Bearish"}[weekly_bias]

    sweeps = h4[(h4["Sweep_Event"] == "Confirmed") & h4["Sweep_Time"].notna() & (h4["Sweep_Time"] <= t)]
    sweep = sweeps.iloc[-1] if not sweeps.empty else None
    if sweep is None or sweep.get("Sweep_Direction", "") != thesis_dir:
        return None

    mss_rows = h4[(h4["MSS_Event"] == "Confirmed") & h4["MSS_Time"].notna()
                  & (h4["MSS_Time"] <= t) & (h4["MSS_Time"] >= sweep["Sweep_Time"])
                  & (h4["MSS_Direction"] == thesis_dir)]
    mss = mss_rows.iloc[-1] if not mss_rows.empty else None
    if mss is None:
        return None

    bos_rows = h4[(h4["BOS_Event"] == "Confirmed") & h4["BOS_Time"].notna()
                  & (h4["BOS_Time"] <= t) & (h4["BOS_Time"] >= mss["MSS_Time"])
                  & (h4["BOS_Direction"] == thesis_dir)]
    bos = bos_rows.iloc[-1] if not bos_rows.empty else None
    if bos is None:
        return None

    zone = find_latest_zone_asof(h4, weekly_bias, t)
    if zone is None or not zone_hit(zone, float(r["Close"])):
        return None

    zone_quality = zone.get("OB_Quality", "")
    if zone_quality not in ("Primary", "Secondary"):
        zone_quality = "Invalid"

    status_asof = mitigation_status_asof(h4, zone, t)
    freshness = "Fresh" if status_asof == "Unmitigated" else ("Coherent" if status_asof == "Partially_Mitigated" else "Unknown")

    m30r, m15r, m5r = _latest_row_asof(m30, t), _latest_row_asof(m15, t), _latest_row_asof(m5, t)

    # infer_ltf_context expects full frames (it calls latest_closed_row
    # internally); pass asof-truncated frames so Is_Current_Bar semantics
    # stay intact instead of passing single rows.
    m30_asof = m30[m30["time"] <= t] if m30 is not None else None
    m15_asof = m15[m15["time"] <= t] if m15 is not None else None
    m5_asof = m5[m5["time"] <= t] if m5 is not None else None
    m30_state, m15_state, m5_state = infer_ltf_context(m30_asof, m15_asof, m5_asof, weekly_bias)

    liquidity = assess_liquidity(h4[h4["time"] <= t])
    session = r.get("Session_Quality", "Acceptable")
    score = score_quality(weekly_bias, weekly_state, "Confirmed", "Confirmed", "Confirmed",
                           zone_quality, pd_zone, session, freshness)
    exec_score = execution_score(m30_state, m15_state, m5_state, session, liquidity)

    plan = estimate_trade_plan(weekly_bias, zone, r, sweep.get("Sweep_Level", np.nan))
    rr = plan["RR"]

    eligible = (m30_state != "Conflicting" and m15_state == "Confirmed"
                and exec_score >= execution_min and pd.notna(rr) and rr >= min_rr)

    return {
        "time": t, "weekly_bias": weekly_bias, "d1_regime": d1_regime, "pd_zone": pd_zone,
        "sweep_time": sweep["Sweep_Time"], "mss_time": mss["MSS_Time"], "bos_time": bos["BOS_Time"],
        "zone_time": zone.get("_Zone_Confirmed_At"),
        "m30_state": m30_state, "m15_state": m15_state, "m5_state": m5_state,
        "quality_score": score, "execution_score": exec_score, "liquidity": liquidity,
        "plan": plan, "eligible": bool(eligible),
    }


def simulate_trade(h4: pd.DataFrame, start_i: int, direction: str, entry: float,
                    stop: float, target: float, horizon: int = BACKTEST_HORIZON_BARS) -> dict:
    """Unchanged from v5 — already causal (only ever looks at bars at/after
    start_i, i.e. strictly after the signal bar)."""
    n = len(h4)
    end_i = min(n - 1, start_i + horizon - 1)
    if start_i >= n or any(pd.isna(x) for x in (entry, stop, target)):
        return {"Outcome": "INVALID", "Exit_Time": pd.NaT, "Exit_Price": np.nan, "Bars_Held": 0,
                "MFE": np.nan, "MAE": np.nan, "R_Multiple": np.nan}

    highs = h4.iloc[start_i:end_i + 1]["High"].to_numpy(dtype=float)
    lows = h4.iloc[start_i:end_i + 1]["Low"].to_numpy(dtype=float)
    times = h4.iloc[start_i:end_i + 1]["time"].to_numpy()
    risk = abs(entry - stop)

    if direction == "BUY":
        mfe, mae = float(np.nanmax(highs) - entry), float(entry - np.nanmin(lows))
        for j, (hi, lo) in enumerate(zip(highs, lows)):
            if lo <= stop:
                return {"Outcome": "LOSS", "Exit_Time": times[j], "Exit_Price": stop, "Bars_Held": j + 1,
                        "MFE": mfe, "MAE": mae, "R_Multiple": -1.0 if risk > 0 else np.nan}
            if hi >= target:
                return {"Outcome": "WIN", "Exit_Time": times[j], "Exit_Price": target, "Bars_Held": j + 1,
                        "MFE": mfe, "MAE": mae, "R_Multiple": (target - entry) / risk if risk > 0 else np.nan}
        exit_price = float(h4.iloc[end_i]["Close"])
        r_mult = (exit_price - entry) / risk if risk > 0 else np.nan
    else:
        mfe, mae = float(entry - np.nanmin(lows)), float(np.nanmax(highs) - entry)
        for j, (hi, lo) in enumerate(zip(highs, lows)):
            if hi >= stop:
                return {"Outcome": "LOSS", "Exit_Time": times[j], "Exit_Price": stop, "Bars_Held": j + 1,
                        "MFE": mfe, "MAE": mae, "R_Multiple": -1.0 if risk > 0 else np.nan}
            if lo <= target:
                return {"Outcome": "WIN", "Exit_Time": times[j], "Exit_Price": target, "Bars_Held": j + 1,
                        "MFE": mfe, "MAE": mae, "R_Multiple": (entry - target) / risk if risk > 0 else np.nan}
        exit_price = float(h4.iloc[end_i]["Close"])
        r_mult = (entry - exit_price) / risk if risk > 0 else np.nan

    return {"Outcome": "TIMEOUT", "Exit_Time": times[-1], "Exit_Price": exit_price,
            "Bars_Held": len(times), "MFE": mfe, "MAE": mae, "R_Multiple": r_mult}


def backtest_symbol(symbol: str, frames: dict, horizon_bars: int = BACKTEST_HORIZON_BARS,
                     min_rr: float = MIN_RR, execution_min: int = EXECUTION_MIN) -> pd.DataFrame:
    """Ported from v5, using evaluate_setup_asof (leak-fixed) instead of
    _backtest_setup_state. One position at a time — no overlapping trades."""
    h4, d1, w1 = frames.get("H4"), frames.get("D1"), frames.get("W1")
    m30, m15, m5 = frames.get("M30"), frames.get("M15"), frames.get("M5")
    if any(x is None or x.empty for x in (h4, d1, w1, m30, m15, m5)):
        return pd.DataFrame()

    h4 = h4.sort_values("time").reset_index(drop=True)
    rows = []
    i = 0
    while i < len(h4) - 1:
        setup = evaluate_setup_asof(h4, w1, d1, m30, m15, m5, i, min_rr, execution_min)
        if setup is None or not setup["eligible"]:
            i += 1
            continue

        next_i = i + 1
        entry = float(h4.iloc[next_i]["Open"])
        plan = setup["plan"].copy()
        stop, target, direction = plan["Stop"], plan["Target"], setup["weekly_bias"]

        invalid_open = (direction == "BUY" and (entry <= stop or entry >= target)) or \
                       (direction == "SELL" and (entry >= stop or entry <= target))
        if invalid_open:
            result = {"Outcome": "INVALIDATED_AT_ENTRY", "Exit_Time": h4.iloc[next_i]["time"],
                      "Exit_Price": entry, "Bars_Held": 0, "MFE": np.nan, "MAE": np.nan, "R_Multiple": np.nan}
        else:
            result = simulate_trade(h4, next_i, direction, entry, stop, target, horizon_bars)

        rows.append({
            "Symbol": symbol, "Signal_Time": setup["time"], "Entry_Time": h4.iloc[next_i]["time"],
            "Direction": direction, "Sweep_Time": setup["sweep_time"], "MSS_Time": setup["mss_time"],
            "BOS_Time": setup["bos_time"], "Zone_Confirmed_Time": setup["zone_time"],
            "D1_Regime": setup["d1_regime"], "PD_Zone_D1": setup["pd_zone"],
            "M30_Context": setup["m30_state"], "M15_Execution": setup["m15_state"],
            "M5_Precision": setup["m5_state"], "Quality_Score": setup["quality_score"],
            "Execution_Quality": setup["execution_score"], "Liquidity": setup["liquidity"],
            "Planned_Entry": plan["Entry"], "Entry_Price": entry, "Stop": stop, "Target": target,
            "Planned_RR": plan["RR"], **result,
        })

        if result["Exit_Time"] is not pd.NaT and pd.notna(result["Exit_Time"]):
            exits = h4.index[h4["time"] == result["Exit_Time"]].tolist()
            if exits:
                i = max(i + 1, exits[0] + 1)
                continue
        i += 1

    return pd.DataFrame(rows)


def calculate_backtest_performance(trades: pd.DataFrame, starting_equity: float = 1.0) -> pd.DataFrame:
    """Unchanged from v5. Analysis-only — converts trade R results into
    equity/drawdown stats, never used for decisions."""
    empty = {
        "Trades": 0, "Wins": 0, "Losses": 0, "Win_Rate": np.nan, "Profit_Factor": np.nan,
        "Expectancy_R": np.nan, "Average_R": np.nan, "Median_R": np.nan, "Total_R": 0.0,
        "Max_Drawdown_R": 0.0, "Max_Drawdown_Pct": 0.0, "Max_Consecutive_Losses": 0,
        "Max_Consecutive_Wins": 0, "Average_MFE": np.nan, "Average_MAE": np.nan, "Average_Bars_Held": np.nan,
    }
    if trades is None or trades.empty:
        return pd.DataFrame([empty])
    t = trades.copy()
    r_col = "R_Multiple" if "R_Multiple" in t.columns else ("R" if "R" in t.columns else None)
    if r_col is None:
        return pd.DataFrame([empty])
    r = pd.to_numeric(t[r_col], errors="coerce").dropna()
    if r.empty:
        return pd.DataFrame([empty])

    equity = starting_equity + r.cumsum()
    peak = equity.cummax()
    dd = equity - peak
    dd_pct = np.where(peak != 0, dd / peak, np.nan)
    wins, losses = r[r > 0], r[r < 0]

    def max_streak(values, positive=True):
        best = cur = 0
        for value in values:
            hit = value > 0 if positive else value < 0
            cur = cur + 1 if hit else 0
            best = max(best, cur)
        return int(best)

    return pd.DataFrame([{
        "Trades": int(len(r)), "Wins": int((r > 0).sum()), "Losses": int((r < 0).sum()),
        "Win_Rate": float((r > 0).mean()),
        "Profit_Factor": float(wins.sum() / abs(losses.sum())) if not losses.empty else np.inf,
        "Expectancy_R": float(r.mean()), "Average_R": float(r.mean()), "Median_R": float(r.median()),
        "Total_R": float(r.sum()), "Max_Drawdown_R": float(dd.min()),
        "Max_Drawdown_Pct": float(dd_pct.min()) if len(dd_pct) else np.nan,
        "Max_Consecutive_Losses": max_streak(r.to_numpy(), positive=False),
        "Max_Consecutive_Wins": max_streak(r.to_numpy(), positive=True),
        "Average_MFE": float(pd.to_numeric(t["MFE"], errors="coerce").mean()) if "MFE" in t.columns else np.nan,
        "Average_MAE": float(pd.to_numeric(t["MAE"], errors="coerce").mean()) if "MAE" in t.columns else np.nan,
        "Average_Bars_Held": float(pd.to_numeric(t["Bars_Held"], errors="coerce").mean()) if "Bars_Held" in t.columns else np.nan,
    }])


def summarize_backtest(trades: pd.DataFrame) -> pd.DataFrame:
    """Unchanged from v5."""
    if trades is None or trades.empty:
        return pd.DataFrame([{"Trades": 0}])
    valid = trades[trades["Outcome"].isin(["WIN", "LOSS", "TIMEOUT", "INVALIDATED_AT_ENTRY"])].copy()
    wins = int((valid["Outcome"] == "WIN").sum())
    losses = int((valid["Outcome"] == "LOSS").sum())
    closed = wins + losses
    r = pd.to_numeric(valid["R_Multiple"], errors="coerce")
    return pd.DataFrame([{
        "Trades": int(len(valid)), "Closed_Win_Loss": closed, "Wins": wins, "Losses": losses,
        "Win_Rate": (wins / closed) if closed else np.nan,
        "Average_R": float(r.mean()) if r.notna().any() else np.nan,
        "Median_R": float(r.median()) if r.notna().any() else np.nan,
        "Total_R": float(r.sum()) if r.notna().any() else np.nan,
    }])
