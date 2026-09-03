"""
primitives.py
==============
Low-level structural building blocks shared by ContextAgent and
SetupAgent. These are pure pandas transforms with no agent logic and no
decision-making — kept in one place instead of duplicated, since both
weekly bias (Context) and liquidity/FVG/OB detection (Setup) are built
on the same confirmed-pivot mechanics.

Ported unchanged from fetch_forex_data_v5_FINAL.py:
    detect_pivots, build_confirmed_liquidity_levels,
    _new_time_column, _new_object_column

These three already had a documented look-ahead-bias fix in v4.1 (a
pivot at bar i is only "confirmed" — i.e. knowable — at i+PIVOT_PERIOD,
never at i itself) and a dtype bugfix (tz-aware time / string columns
must not be initialized as bare NaN). Both are preserved exactly.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

PIVOT_PERIOD = 10


def _new_time_column(df: pd.DataFrame) -> pd.Series:
    """Object/NaT column pre-typed to match df['time']'s tz-aware dtype,
    so later per-row Timestamp writes never silently downcast the column
    or raise on tz-naive/tz-aware comparison (pandas 3.x bugfix, v4.1)."""
    return pd.Series(pd.NaT, index=df.index, dtype=df["time"].dtype)


def _new_object_column(df: pd.DataFrame) -> pd.Series:
    """Object-dtype NaN column for eventual string labels (FVG_Type,
    OB_Type, ...) — avoids the float64-can't-hold-a-string crash (v4.1)."""
    return pd.Series(np.nan, index=df.index, dtype=object)


def detect_pivots(df: pd.DataFrame, prd: int = PIVOT_PERIOD) -> pd.DataFrame:
    """Strict local pivots, vectorized. A pivot at i is confirmed at i+prd;
    the confirmation timestamp is preserved so nothing downstream can
    treat it as known before that time."""
    df = df.copy().reset_index(drop=True)

    df["Pivot_High"] = np.nan
    df["Pivot_Low"] = np.nan
    df["Pivot_High_Time"] = _new_time_column(df)
    df["Pivot_Low_Time"] = _new_time_column(df)
    df["Pivot_High_Confirmation_Time"] = _new_time_column(df)
    df["Pivot_Low_Confirmation_Time"] = _new_time_column(df)

    if len(df) < 2 * prd + 1:
        return df

    high = df["High"]
    low = df["Low"]

    left_high = high.rolling(prd, min_periods=prd).max().shift(1)
    right_high = high.rolling(prd, min_periods=prd).max().shift(-prd)
    left_low = low.rolling(prd, min_periods=prd).min().shift(1)
    right_low = low.rolling(prd, min_periods=prd).min().shift(-prd)

    ph_mask = high.gt(left_high) & high.gt(right_high)
    pl_mask = low.lt(left_low) & low.lt(right_low)

    interior = pd.Series(False, index=df.index)
    interior.iloc[prd:len(df) - prd] = True
    ph_mask &= interior
    pl_mask &= interior

    df.loc[ph_mask, "Pivot_High"] = high.loc[ph_mask]
    df.loc[pl_mask, "Pivot_Low"] = low.loc[pl_mask]
    df.loc[ph_mask, "Pivot_High_Time"] = df.loc[ph_mask, "time"]
    df.loc[pl_mask, "Pivot_Low_Time"] = df.loc[pl_mask, "time"]

    ph_conf = df["time"].shift(-prd)
    pl_conf = df["time"].shift(-prd)
    df.loc[ph_mask, "Pivot_High_Confirmation_Time"] = ph_conf.loc[ph_mask]
    df.loc[pl_mask, "Pivot_Low_Confirmation_Time"] = pl_conf.loc[pl_mask]

    return df


def build_confirmed_liquidity_levels(df: pd.DataFrame, prd: int = PIVOT_PERIOD) -> pd.DataFrame:
    """Carries the most recent CONFIRMED pivot high/low forward (a pivot
    originating at i only becomes usable at i+prd)."""
    df = df.copy()
    df["Confirmed_Pivot_High"] = df["Pivot_High"].shift(prd).ffill()
    df["Confirmed_Pivot_Low"] = df["Pivot_Low"].shift(prd).ffill()
    return df


# ---------------------------------------------------------------------------
# Shared per-timeframe enrichment
# ---------------------------------------------------------------------------
# v5's process_timeframe() applies the SAME transform stack to every
# timeframe (D1, W1, H4, M30, M15, M5, ...), regardless of which decision
# layer later reads the result. Only the regime lookback varies by
# timeframe. Reproduced here as one function so DataAgent enriches once
# and every agent (Context/Setup/Execution) reads already-computed
# columns instead of each recomputing — which is what caused the wrong
# regime lookback (20 vs the correct default of 10) in an earlier draft
# of context_agent.py.

REGIME_RANGE_ATR_LOOKBACK = 10
REGIME_RANGE_ATR_LOOKBACK_BY_TF = {"M5": 20, "M15": 20, "M30": 12}

FVG_MIN_ATR_MULT = 0.30
FVG_DISPLACEMENT_MULT = 1.20
OB_DISPLACEMENT_MULT = 1.50
AVG_RANGE_LOOKBACK = 10
BROKER_UTC_OFFSET_HOURS = 3  # TODO: confirm against wmmarket demo server time, not yet verified


def classify_regime(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Unchanged from v5. ADX + Range/ATR expansion count + normalized
    EMA20 slope -> Parabolic / Orderly_Trend / Range_Bound / Indeterminate."""
    df = df.copy()
    df["Regime"] = "Indeterminate"

    ratio = df["Range_ATR_Ratio"]
    expansion_count = ratio.rolling(lookback, min_periods=lookback).apply(
        lambda x: np.sum(x > 1.5), raw=True
    )
    atr = df["ATR"]
    slope_atr = df["EMA20_Slope"].abs().div(atr.where(atr > 0))
    trending_slope = slope_atr.ge(0.5)

    valid = (
        df["ADX"].notna() & ratio.notna() & expansion_count.notna()
        & df["EMA20_Slope"].notna()
    )
    parabolic = valid & df["ADX"].gt(35) & expansion_count.ge(3) & trending_slope
    orderly = (
        valid & df["ADX"].between(20, 35, inclusive="both")
        & ratio.le(1.3) & trending_slope
    )
    range_bound = valid & df["ADX"].lt(20) & ratio.le(1.3)

    df.loc[parabolic, "Regime"] = "Parabolic"
    df.loc[orderly, "Regime"] = "Orderly_Trend"
    df.loc[range_bound, "Regime"] = "Range_Bound"
    return df


def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """Unchanged from v5.

    IMPORTANT — backtest-safety note (not a live-agent bug): FVG_Status
    (Unmitigated/Partially_Mitigated/Fully_Mitigated) is computed with a
    single O(n) pass using min/max of ALL bars after each FVG, all the
    way to the end of whatever DataFrame is passed in. In live use the
    last bar IS "now", so this is causal and correct. If this function's
    output is ever reused inside a walk-forward/backtest loop that
    evaluates decisions at multiple earlier points in time (as v5's
    _backtest_setup_state did, reading this same globally-computed
    FVG_Status at every historical bar i), that reuse leaks future
    information the strategy would not have known at time i — the
    backtest's reported quality scores would be systematically
    optimistic. Any research/backtest module must recompute this
    per-simulated-time on a truncated frame, never reuse a globally
    computed column.
    """
    df = df.copy().reset_index(drop=True)
    for c in ["FVG_Top", "FVG_Bottom", "FVG_Size", "FVG_ATR_Size", "FVG_Displacement_Ratio"]:
        if c not in df:
            df[c] = np.nan
    if "FVG_Type" not in df:
        df["FVG_Type"] = _new_object_column(df)
    if "FVG_Event_Time" not in df:
        df["FVG_Event_Time"] = _new_time_column(df)

    n = len(df)
    if n == 0:
        df["FVG_Status"] = ""
        return df

    atr = df["ATR"].to_numpy(dtype=float)
    avg_range_prior = (
        df["Candle_Range"].rolling(AVG_RANGE_LOOKBACK, min_periods=AVG_RANGE_LOOKBACK)
        .mean().shift(1).to_numpy(dtype=float)
    )
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    candle_range = df["Candle_Range"].to_numpy(dtype=float)

    fvg_type = np.full(n, np.nan, dtype=object)
    top = np.full(n, np.nan, dtype=float)
    bottom = np.full(n, np.nan, dtype=float)
    size = np.full(n, np.nan, dtype=float)
    atr_size = np.full(n, np.nan, dtype=float)
    displacement = np.full(n, np.nan, dtype=float)
    event_time = pd.Series(pd.NaT, index=df.index, dtype=df["time"].dtype)

    if n > 2:
        valid = (
            np.isfinite(atr[2:]) & (atr[2:] > 0)
            & np.isfinite(avg_range_prior[2:]) & (avg_range_prior[2:] > 0)
        )
        idx = np.arange(2, n)
        disp = np.full(n, np.nan)
        disp[idx] = candle_range[idx - 1] / avg_range_prior[idx]
        valid &= disp[idx] >= FVG_DISPLACEMENT_MULT

        gap_up = low[idx] - high[idx - 2]
        gap_down = low[idx - 2] - high[idx]
        bull = valid & (gap_up >= FVG_MIN_ATR_MULT * atr[idx])
        bear = (~bull) & valid & (gap_down >= FVG_MIN_ATR_MULT * atr[idx])

        bi = idx[bull]
        si = idx[bear]

        fvg_type[bi] = "Bullish"
        top[bi] = low[bi]
        bottom[bi] = high[bi - 2]
        size[bi] = gap_up[bull]
        atr_size[bi] = size[bi] / atr[bi]
        displacement[bi] = disp[bi]

        fvg_type[si] = "Bearish"
        top[si] = low[si - 2]
        bottom[si] = high[si]
        size[si] = gap_down[bear]
        atr_size[si] = size[si] / atr[si]
        displacement[si] = disp[si]

        event_time.iloc[bi] = df["time"].iloc[bi].to_numpy()
        event_time.iloc[si] = df["time"].iloc[si].to_numpy()

    df["FVG_Type"] = fvg_type
    df["FVG_Top"] = top
    df["FVG_Bottom"] = bottom
    df["FVG_Size"] = size
    df["FVG_ATR_Size"] = atr_size
    df["FVG_Displacement_Ratio"] = displacement
    df["FVG_Event_Time"] = event_time

    future_min_low = pd.Series(low[::-1]).cummin().shift(1).to_numpy()[::-1]
    future_max_high = pd.Series(high[::-1]).cummax().shift(1).to_numpy()[::-1]

    mid = (top + bottom) / 2.0
    status = np.full(n, "", dtype=object)
    bull_mask = fvg_type == "Bullish"
    bear_mask = fvg_type == "Bearish"

    status[bull_mask & np.isfinite(future_min_low) & (future_min_low <= bottom)] = "Fully_Mitigated"
    status[bull_mask & np.isfinite(future_min_low) & (future_min_low > bottom) &
           (future_min_low <= mid)] = "Partially_Mitigated"
    status[bull_mask & (status == "")] = "Unmitigated"

    status[bear_mask & np.isfinite(future_max_high) & (future_max_high >= top)] = "Fully_Mitigated"
    status[bear_mask & np.isfinite(future_max_high) & (future_max_high < top) &
           (future_max_high >= mid)] = "Partially_Mitigated"
    status[bear_mask & (status == "")] = "Unmitigated"

    df["FVG_Status"] = status
    return df


def detect_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Unchanged from v5. OB confirmation uses only the immediate next bar
    (i+1) — a real confirmation event, not future leakage."""
    df = df.copy()
    for c in ["OB_Top", "OB_Bottom", "OB_Displacement_Ratio"]:
        if c not in df:
            df[c] = np.nan
    for c in ["OB_Type", "OB_Quality", "OB_Has_FVG_Confirmation"]:
        if c not in df:
            df[c] = _new_object_column(df)
    if "OB_Event_Time" not in df:
        df["OB_Event_Time"] = _new_time_column(df)

    for i in range(1, len(df) - 1):
        atr = df["ATR"].iloc[i + 1]
        if pd.isna(atr) or atr <= 0:
            continue
        next_range = df["Candle_Range"].iloc[i + 1]
        displacement_ratio = next_range / atr
        origin = i - 1

        if (df["Close"].iloc[origin] < df["Open"].iloc[origin]
                and displacement_ratio >= OB_DISPLACEMENT_MULT
                and df["Close"].iloc[i + 1] > df["High"].iloc[origin]):
            df.at[origin, "OB_Type"] = "Bullish"
            df.at[origin, "OB_Top"] = df["High"].iloc[origin]
            df.at[origin, "OB_Bottom"] = df["Low"].iloc[origin]
            df.at[origin, "OB_Event_Time"] = df["time"].iloc[i + 1]
            df.at[origin, "OB_Displacement_Ratio"] = displacement_ratio
            future_fvg = df["FVG_Type"].iloc[i + 1] == "Bullish" if "FVG_Type" in df else False
            df.at[origin, "OB_Has_FVG_Confirmation"] = bool(future_fvg)
            df.at[origin, "OB_Quality"] = "Primary" if future_fvg else "Secondary"

        if (df["Close"].iloc[origin] > df["Open"].iloc[origin]
                and displacement_ratio >= OB_DISPLACEMENT_MULT
                and df["Close"].iloc[i + 1] < df["Low"].iloc[origin]):
            df.at[origin, "OB_Type"] = "Bearish"
            df.at[origin, "OB_Top"] = df["High"].iloc[origin]
            df.at[origin, "OB_Bottom"] = df["Low"].iloc[origin]
            df.at[origin, "OB_Event_Time"] = df["time"].iloc[i + 1]
            df.at[origin, "OB_Displacement_Ratio"] = displacement_ratio
            future_fvg = df["FVG_Type"].iloc[i + 1] == "Bearish" if "FVG_Type" in df else False
            df.at[origin, "OB_Has_FVG_Confirmation"] = bool(future_fvg)
            df.at[origin, "OB_Quality"] = "Primary" if future_fvg else "Secondary"

    return df


def tag_session(df: pd.DataFrame, broker_utc_offset_hours: float = BROKER_UTC_OFFSET_HOURS) -> pd.DataFrame:
    """Unchanged from v5. TODO: broker_utc_offset_hours=3 was for the old
    broker and is NOT yet confirmed against wmmarket's demo server time."""
    df = df.copy()
    local = df["time"] + pd.Timedelta(hours=broker_utc_offset_hours)
    hour = local.dt.hour

    conditions = [hour.between(12, 14), hour.between(7, 9), hour.between(18, 19), hour.between(0, 5)]
    choices = ["NY_AM_Preferred", "London_Preferred", "NYPM_Acceptable", "Asian_Low"]
    df["Session_Priority"] = np.select(conditions, choices, default="Other_Acceptable")

    df["Session_Quality"] = np.select(
        [df["Session_Priority"].isin(["NY_AM_Preferred", "London_Preferred"]),
         df["Session_Priority"].eq("NYPM_Acceptable"),
         df["Session_Priority"].eq("Asian_Low")],
        ["Preferred", "Acceptable", "Low_Priority"],
        default="Acceptable",
    )
    return df


def add_premium_discount(df: pd.DataFrame) -> pd.DataFrame:
    """Unchanged from v5. Uses confirmed pivots only."""
    df = df.copy()
    last_high = df["Confirmed_Pivot_High"].ffill()
    last_low = df["Confirmed_Pivot_Low"].ffill()
    eq = (last_high + last_low) / 2.0
    valid = last_high.notna() & last_low.notna() & last_high.gt(last_low)

    df["Dealing_Range_High"] = last_high.where(valid, np.nan)
    df["Dealing_Range_Low"] = last_low.where(valid, np.nan)
    df["Equilibrium"] = eq.where(valid, np.nan)

    zone = np.full(len(df), "Uncertain", dtype=object)
    close = df["Close"]
    zone[valid & close.lt(eq)] = "Discount"
    zone[valid & close.gt(eq)] = "Premium"
    zone[valid & close.eq(eq)] = "Equilibrium"
    df["PD_Zone"] = zone
    return df


def detect_structure_events(df: pd.DataFrame) -> pd.DataFrame:
    """Unchanged from v5. Row-by-row (not vectorized) but genuinely
    causal — Sweep -> MSS -> BOS state is only ever built from
    df.iloc[i] and earlier accumulated pivots, so unlike detect_fvg()
    this one has no future-leakage concern even when reused across
    multiple simulated points in time."""
    df = df.copy().sort_values("time").reset_index(drop=True)

    event_cols = [
        "Sweep_Event", "Sweep_Direction", "Sweep_Level", "Sweep_Time",
        "MSS_Event", "MSS_Direction", "MSS_Level", "MSS_Time",
        "BOS_Event", "BOS_Direction", "BOS_Level", "BOS_Time",
    ]
    for c in event_cols:
        if c.endswith("Time"):
            df[c] = _new_time_column(df)
        elif c.endswith("Level"):
            df[c] = np.nan
        else:
            df[c] = ""

    # CRITICAL FIX (found via real-data testing, confirmed present in v5
    # unchanged): v5 checked `Pivot_High_Confirmation_Time.iloc[i] <= t`
    # where t IS df["time"].iloc[i] — but a pivot's confirmation time
    # (recorded at its own origin row) is always PIVOT_PERIOD bars in
    # the future relative to that same row's own time. That comparison
    # is mathematically always False, so confirmed_highs/confirmed_lows
    # were NEVER populated, so Sweep (and therefore MSS and BOS, which
    # depend on it) NEVER fired — on any data, ever, in the original
    # file. Fixed here by indexing pivots by confirmation time up front
    # and admitting each one once the loop's current row reaches it.
    ph_queue = sorted(
        (t_, float(v)) for t_, v in
        zip(df.loc[df["Pivot_High"].notna(), "Pivot_High_Confirmation_Time"],
            df.loc[df["Pivot_High"].notna(), "Pivot_High"])
        if pd.notna(t_)
    )
    pl_queue = sorted(
        (t_, float(v)) for t_, v in
        zip(df.loc[df["Pivot_Low"].notna(), "Pivot_Low_Confirmation_Time"],
            df.loc[df["Pivot_Low"].notna(), "Pivot_Low"])
        if pd.notna(t_)
    )
    ph_ptr = pl_ptr = 0

    confirmed_highs: list[tuple[pd.Timestamp, float]] = []
    confirmed_lows: list[tuple[pd.Timestamp, float]] = []
    pending_sweep = None
    pending_mss = None

    for i in range(len(df)):
        t = df["time"].iloc[i]
        close = df["Close"].iloc[i]
        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]

        while ph_ptr < len(ph_queue) and ph_queue[ph_ptr][0] <= t:
            confirmed_highs.append(ph_queue[ph_ptr])
            ph_ptr += 1
        while pl_ptr < len(pl_queue) and pl_queue[pl_ptr][0] <= t:
            confirmed_lows.append(pl_queue[pl_ptr])
            pl_ptr += 1

        prior_high = confirmed_highs[-1][1] if confirmed_highs else np.nan
        prior_low = confirmed_lows[-1][1] if confirmed_lows else np.nan

        if pd.notna(prior_high) and high > prior_high and close < prior_high:
            df.at[i, "Sweep_Event"] = "Confirmed"
            df.at[i, "Sweep_Direction"] = "Bearish"
            df.at[i, "Sweep_Level"] = prior_high
            df.at[i, "Sweep_Time"] = t
            pending_sweep = {"direction": "Bearish", "time": t, "level": prior_high}
        elif pd.notna(prior_low) and low < prior_low and close > prior_low:
            df.at[i, "Sweep_Event"] = "Confirmed"
            df.at[i, "Sweep_Direction"] = "Bullish"
            df.at[i, "Sweep_Level"] = prior_low
            df.at[i, "Sweep_Time"] = t
            pending_sweep = {"direction": "Bullish", "time": t, "level": prior_low}

        # MSS after sweep (FIX vs v5 — see primitives.py module note below):
        # a Bullish sweep (sell-side liquidity swept, bullish reversal
        # thesis) is confirmed by MSS breaking ABOVE a confirmed high
        # (Bullish MSS) — the SAME bias the sweep implied, per prompt
        # Sec. 7 ("break of a relevant opposing swing" continuing the
        # thesis). v5 had this inverted (bearish sweep -> bullish MSS),
        # which — combined with the confirmation-timing bug above — meant
        # no setup could ever reach S3 on any real data.
        if pending_sweep:
            if pending_sweep["direction"] == "Bullish" and pd.notna(prior_high):
                opposing = [x for x in confirmed_highs if x[0] <= pending_sweep["time"]]
                if opposing and close > opposing[-1][1]:
                    df.at[i, "MSS_Event"] = "Confirmed"
                    df.at[i, "MSS_Direction"] = "Bullish"
                    df.at[i, "MSS_Level"] = opposing[-1][1]
                    df.at[i, "MSS_Time"] = t
                    pending_mss = {"direction": "Bullish", "time": t}
                    pending_sweep = None
            elif pending_sweep["direction"] == "Bearish" and pd.notna(prior_low):
                opposing = [x for x in confirmed_lows if x[0] <= pending_sweep["time"]]
                if opposing and close < opposing[-1][1]:
                    df.at[i, "MSS_Event"] = "Confirmed"
                    df.at[i, "MSS_Direction"] = "Bearish"
                    df.at[i, "MSS_Level"] = opposing[-1][1]
                    df.at[i, "MSS_Time"] = t
                    pending_mss = {"direction": "Bearish", "time": t}
                    pending_sweep = None

        if pending_mss:
            if pending_mss["direction"] == "Bullish" and pd.notna(prior_high):
                if close > prior_high:
                    df.at[i, "BOS_Event"] = "Confirmed"
                    df.at[i, "BOS_Direction"] = "Bullish"
                    df.at[i, "BOS_Level"] = prior_high
                    df.at[i, "BOS_Time"] = t
                    pending_mss = None
            elif pending_mss["direction"] == "Bearish" and pd.notna(prior_low):
                if close < prior_low:
                    df.at[i, "BOS_Event"] = "Confirmed"
                    df.at[i, "BOS_Direction"] = "Bearish"
                    df.at[i, "BOS_Level"] = prior_low
                    df.at[i, "BOS_Time"] = t
                    pending_mss = None

    return df


def latest_closed_row(df: pd.DataFrame):
    """Unchanged from v5. Relies on the explicit Is_Current_Bar flag
    (set by DataAgent._fetch_rates) as the single source of truth for
    "last closed candle", rather than assuming position -2."""
    if df is None or df.empty:
        return None
    if "Is_Current_Bar" in df.columns:
        closed = df[~df["Is_Current_Bar"].astype(bool)]
        if not closed.empty:
            return closed.iloc[-1]
        return df.iloc[-1]
    if len(df) >= 2:
        return df.iloc[-2]
    return df.iloc[-1]


def build_weekly_bias(w1: pd.DataFrame) -> pd.DataFrame:
    """Unchanged from v5. NOT part of the uniform per-timeframe stack —
    in v5 this is called separately, only on the W1 frame, after
    process_timeframe() already ran (make_symbol_summary(), line ~1582).
    Requires Pivot_High/Pivot_Low, which detect_pivots() above provides."""
    df = w1.copy().sort_values("time").reset_index(drop=True)

    confirmed_high = df["Pivot_High"].shift(PIVOT_PERIOD).ffill()
    confirmed_low = df["Pivot_Low"].shift(PIVOT_PERIOD).ffill()

    bullish_break = confirmed_high.notna() & df["Close"].gt(confirmed_high)
    bearish_break = confirmed_low.notna() & df["Close"].lt(confirmed_low)

    event = pd.Series(np.nan, index=df.index, dtype=object)
    event.loc[bullish_break] = "BUY"
    event.loc[bearish_break & ~bullish_break] = "SELL"

    bias = event.ffill().fillna("NEUTRAL")
    df["Weekly_Bias"] = bias
    df["Weekly_Bias_State"] = np.where(bias.eq("NEUTRAL"), "Unconfirmed", "Confirmed")

    bos = bullish_break | bearish_break
    df["Weekly_BOS"] = bos
    df["Weekly_BOS_Direction"] = ""
    df.loc[bullish_break, "Weekly_BOS_Direction"] = "Bullish"
    df.loc[bearish_break & ~bullish_break, "Weekly_BOS_Direction"] = "Bearish"
    return df


def enrich_frame(df: pd.DataFrame, tf_name: str) -> pd.DataFrame:
    """The uniform per-timeframe enrichment stack from v5's
    process_timeframe() (everything after fetch_rates+add_indicators,
    which DataAgent already does), PLUS build_weekly_bias() for W1 —
    folded in here (rather than kept as a separate late call like v5)
    so DataAgent's output is fully self-contained per timeframe and
    ContextAgent never has to know which frame needs an extra step.
    """
    lookback = REGIME_RANGE_ATR_LOOKBACK_BY_TF.get(tf_name, REGIME_RANGE_ATR_LOOKBACK)
    df = classify_regime(df, lookback)
    df = detect_pivots(df, PIVOT_PERIOD)
    df = build_confirmed_liquidity_levels(df)
    df = detect_fvg(df)
    df = detect_order_blocks(df)
    df = tag_session(df)
    df = add_premium_discount(df)
    if tf_name == "W1":
        df = build_weekly_bias(df)
    if tf_name in ("H4", "M30", "M15", "M5"):
        df = detect_structure_events(df)
    return df
