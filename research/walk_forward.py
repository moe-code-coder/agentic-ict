"""
research/walk_forward.py
==========================
True chronological walk-forward optimization, ported from v5 unchanged
in structure (it was already correct): optimize inside Train only,
freeze parameters, evaluate once on unseen Test, advance without ever
looking back at Test performance. Builds on backtest.py's leak-fixed
evaluate_setup_asof, so the leak fix applies here too automatically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import (
    evaluate_setup_asof, simulate_trade, calculate_backtest_performance, BACKTEST_HORIZON_BARS,
)

WFO_TRAIN_BARS = 600
WFO_TEST_BARS = 150
WFO_STEP_BARS = 150
WFO_MIN_TRAIN_TRADES = 30
WFO_MIN_OOS_TRADES = 10

# Deliberately small grid — WFO here is for robustness, not brute-force
# parameter mining. These are the two hard execution gates already in
# the strategy (see agents/execution_agent.py MIN_RR / EXECUTION_MIN).
WFO_MIN_RR_GRID = [2.0, 2.5, 3.0, 3.5, 4.0]
WFO_EXECUTION_MIN_GRID = [6, 7, 8, 9]


def _slice_frames_asof(frames: dict, start_time: pd.Timestamp, end_time: pd.Timestamp) -> dict:
    """Retain all pre-window history (needed to establish confirmed
    structure) but never a bar after end_time."""
    out = {}
    for tf, df in frames.items():
        if df is None or df.empty or "time" not in df.columns:
            continue
        out[tf] = df.loc[df["time"] <= end_time].copy().sort_values("time").reset_index(drop=True)
    return out


def backtest_in_period(symbol: str, frames: dict, start_time: pd.Timestamp, end_time: pd.Timestamp,
                        min_rr: float, execution_min: int) -> pd.DataFrame:
    """Runs the same engine as backtest.backtest_symbol but only accepts
    signals whose signal candle falls inside [start_time, end_time] —
    used to isolate Train vs. Test performance within one continuous
    history."""
    local = _slice_frames_asof(frames, start_time, end_time)
    h4 = local.get("H4")
    if h4 is None or h4.empty:
        return pd.DataFrame()

    h4_idx = h4.index[h4["time"] >= start_time]
    if len(h4_idx) == 0:
        return pd.DataFrame()

    w1, d1 = local.get("W1"), local.get("D1")
    m30, m15, m5 = local.get("M30"), local.get("M15"), local.get("M5")
    rows = []

    for i in h4_idx:
        i = int(i)
        if i >= len(h4) - 1:
            continue
        setup = evaluate_setup_asof(h4, w1, d1, m30, m15, m5, i, min_rr, execution_min)
        if setup is None or not setup["eligible"]:
            continue

        next_i = i + 1
        if h4.iloc[next_i]["time"] > end_time:
            continue

        entry = float(h4.iloc[next_i]["Open"])
        plan = setup["plan"].copy()
        stop, target, direction = plan["Stop"], plan["Target"], setup["weekly_bias"]

        invalid_open = (direction == "BUY" and (entry <= stop or entry >= target)) or \
                       (direction == "SELL" and (entry >= stop or entry <= target))
        if invalid_open:
            result = {"Outcome": "INVALIDATED_AT_ENTRY", "Exit_Time": h4.iloc[next_i]["time"],
                      "Exit_Price": entry, "Bars_Held": 0, "MFE": np.nan, "MAE": np.nan, "R_Multiple": np.nan}
        else:
            result = simulate_trade(h4, next_i, direction, entry, stop, target, BACKTEST_HORIZON_BARS)

        rows.append({
            "Symbol": symbol, "Signal_Time": setup["time"], "Entry_Time": h4.iloc[next_i]["time"],
            "Direction": direction, "Sweep_Time": setup["sweep_time"], "MSS_Time": setup["mss_time"],
            "BOS_Time": setup["bos_time"], "Zone_Confirmed_Time": setup["zone_time"],
            "D1_Regime": setup["d1_regime"], "PD_Zone_D1": setup["pd_zone"],
            "M30_Context": setup["m30_state"], "M15_Execution": setup["m15_state"],
            "M5_Precision": setup["m5_state"], "Quality_Score": setup["quality_score"],
            "Execution_Quality": setup["execution_score"], "Liquidity": setup["liquidity"],
            "Planned_Entry": plan["Entry"], "Entry_Price": entry, "Stop": stop, "Target": target,
            "Planned_RR": plan["RR"], "WFO_Min_RR": min_rr, "WFO_Execution_Min": execution_min,
            **result,
        })

    return pd.DataFrame(rows)


def _wfo_score(summary: pd.DataFrame) -> float:
    """Conservative training objective: rewards expectancy and profit
    factor, penalizes drawdown, and hard-floors on minimum sample size
    (caller enforces WFO_MIN_TRAIN_TRADES) — not Sharpe-maximization."""
    if summary is None or summary.empty:
        return -np.inf
    row = summary.iloc[0]
    trades = float(row.get("Trades", 0) or 0)
    expectancy = float(row.get("Expectancy_R", row.get("Average_R", np.nan)))
    pf = float(row.get("Profit_Factor", np.nan))
    dd = abs(float(row.get("Max_Drawdown_R", np.nan)))

    if not np.isfinite(expectancy) or trades < WFO_MIN_TRAIN_TRADES:
        return -np.inf

    pf_term = np.log1p(max(pf, 0.0)) if np.isfinite(pf) else 0.0
    dd_penalty = dd / max(trades, 1.0)
    return expectancy * (1.0 + 0.25 * pf_term) - 0.10 * dd_penalty


def optimize_wfo_window(symbol: str, frames: dict, train_start: pd.Timestamp, train_end: pd.Timestamp):
    candidates = []
    for min_rr in WFO_MIN_RR_GRID:
        for execution_min in WFO_EXECUTION_MIN_GRID:
            trades = backtest_in_period(symbol, frames, train_start, train_end, min_rr, execution_min)
            perf = calculate_backtest_performance(trades)
            score = _wfo_score(perf)
            candidates.append({
                "Min_RR": min_rr, "Execution_Min": execution_min,
                "Train_Trades": int(perf.iloc[0].get("Trades", 0)),
                "Train_Expectancy_R": float(perf.iloc[0].get("Expectancy_R", np.nan)),
                "Train_Profit_Factor": float(perf.iloc[0].get("Profit_Factor", np.nan)),
                "Train_Max_Drawdown_R": float(perf.iloc[0].get("Max_Drawdown_R", np.nan)),
                "Objective": score,
            })

    grid = pd.DataFrame(candidates).sort_values(["Objective", "Train_Trades"], ascending=[False, False]).reset_index(drop=True)
    if grid.empty or not np.isfinite(grid.iloc[0]["Objective"]):
        return None, grid
    return grid.iloc[0].to_dict(), grid


def walk_forward_optimize_symbol(symbol: str, frames: dict, train_bars: int = WFO_TRAIN_BARS,
                                  test_bars: int = WFO_TEST_BARS, step_bars: int = WFO_STEP_BARS):
    h4 = frames.get("H4")
    if h4 is None or h4.empty:
        return pd.DataFrame(), pd.DataFrame()

    h4 = h4.sort_values("time").reset_index(drop=True)
    if len(h4) < train_bars + test_bars:
        return pd.DataFrame(), pd.DataFrame()

    window_rows, oos_trades = [], []
    start, window_id = 0, 0

    while start + train_bars + test_bars <= len(h4):
        train_start, train_end = h4.iloc[start]["time"], h4.iloc[start + train_bars - 1]["time"]
        test_start, test_end = h4.iloc[start + train_bars]["time"], h4.iloc[start + train_bars + test_bars - 1]["time"]

        best, _grid = optimize_wfo_window(symbol, frames, train_start, train_end)
        if best is None:
            window_rows.append({
                "Symbol": symbol, "Window": window_id, "Train_Start": train_start, "Train_End": train_end,
                "Test_Start": test_start, "Test_End": test_end, "Status": "NO_VALID_TRAIN_CONFIGURATION",
            })
            start += step_bars
            window_id += 1
            continue

        oos = backtest_in_period(symbol, frames, test_start, test_end,
                                  min_rr=float(best["Min_RR"]), execution_min=int(best["Execution_Min"]))
        oos_perf = calculate_backtest_performance(oos)
        if not oos.empty:
            oos = oos.copy()
            oos.insert(1, "WFO_Window", window_id)
            oos_trades.append(oos)

        oos_row = oos_perf.iloc[0].to_dict() if not oos_perf.empty else {}
        window_rows.append({
            "Symbol": symbol, "Window": window_id, "Train_Start": train_start, "Train_End": train_end,
            "Test_Start": test_start, "Test_End": test_end,
            "Selected_Min_RR": float(best["Min_RR"]), "Selected_Execution_Min": int(best["Execution_Min"]),
            "Train_Trades": int(best["Train_Trades"]), "Train_Expectancy_R": float(best["Train_Expectancy_R"]),
            "Train_Profit_Factor": float(best["Train_Profit_Factor"]), "Train_Max_Drawdown_R": float(best["Train_Max_Drawdown_R"]),
            "OOS_Trades": int(oos_row.get("Trades", 0)), "OOS_Win_Rate": float(oos_row.get("Win_Rate", np.nan)),
            "OOS_Expectancy_R": float(oos_row.get("Expectancy_R", np.nan)), "OOS_Profit_Factor": float(oos_row.get("Profit_Factor", np.nan)),
            "OOS_Total_R": float(oos_row.get("Total_R", np.nan)), "OOS_Max_Drawdown_R": float(oos_row.get("Max_Drawdown_R", np.nan)),
            "Status": "OOS_OK" if int(oos_row.get("Trades", 0)) >= WFO_MIN_OOS_TRADES else "OOS_UNDERPOWERED",
        })
        start += step_bars
        window_id += 1

    windows = pd.DataFrame(window_rows)
    trades = pd.concat(oos_trades, ignore_index=True) if oos_trades else pd.DataFrame()
    return windows, trades


def summarize_wfo_windows(windows: pd.DataFrame) -> pd.DataFrame:
    """NOTE: only the first ~15 lines of v5's summarize_wfo_windows were
    visible during porting; the aggregation formulas below (weighted
    OOS expectancy, mean win rate/profit factor, % positive windows) are
    my own reconstruction of a reasonable summary, not a verified
    line-for-line port. Treat the numbers this produces as directional
    until checked against the original file if you still have it."""
    empty = {"Windows": 0, "Valid_OOS_Windows": 0, "OOS_Trades": 0, "OOS_Total_R": 0.0,
             "OOS_Expectancy_R": np.nan, "OOS_Win_Rate": np.nan, "OOS_Profit_Factor": np.nan,
             "Positive_OOS_Windows_Pct": np.nan}
    if windows is None or windows.empty:
        return pd.DataFrame([empty])

    w = windows.copy()
    valid = w[w["Status"].eq("OOS_OK")].copy()
    if valid.empty:
        out = dict(empty)
        out["Windows"] = int(len(w))
        return pd.DataFrame([out])

    trades = pd.to_numeric(valid["OOS_Trades"], errors="coerce").fillna(0)
    total_r = pd.to_numeric(valid["OOS_Total_R"], errors="coerce").fillna(0)
    n = int(trades.sum())
    total = float(total_r.sum())
    pos_windows_pct = float((pd.to_numeric(valid["OOS_Total_R"], errors="coerce") > 0).mean() * 100.0)

    return pd.DataFrame([{
        "Windows": int(len(w)), "Valid_OOS_Windows": int(len(valid)), "OOS_Trades": n,
        "OOS_Total_R": total, "OOS_Expectancy_R": (total / n) if n else np.nan,
        "OOS_Win_Rate": float(pd.to_numeric(valid["OOS_Win_Rate"], errors="coerce").mean()),
        "OOS_Profit_Factor": float(pd.to_numeric(valid["OOS_Profit_Factor"], errors="coerce").replace([np.inf, -np.inf], np.nan).mean()),
        "Positive_OOS_Windows_Pct": pos_windows_pct,
    }])
