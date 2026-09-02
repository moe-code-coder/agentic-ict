"""
scripts/run_backtest_from_files.py
=====================================
Loads the Parquet files produced by export_history.py and runs the
backtest + walk-forward optimization against REAL wmmarket history
instead of synthetic data (see README's "Backtest validity caveat" —
synthetic random-walk data cannot properly exercise the Sweep detector).

Usage:
    python scripts/run_backtest_from_files.py XAUUSD
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agentic_ict.research.backtest import backtest_symbol, calculate_backtest_performance, summarize_backtest
from agentic_ict.research.walk_forward import walk_forward_optimize_symbol, summarize_wfo_windows

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REQUIRED_TFS = ["M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"]


def load_frames(symbol: str) -> dict:
    symbol_dir = DATA_DIR / symbol
    frames = {}
    for tf in REQUIRED_TFS:
        path = symbol_dir / f"{tf}.parquet"
        if not path.exists():
            print(f"  WARNING: missing {path} — run export_history.py first.")
            continue
        df = pd.read_parquet(path)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        frames[tf] = df
    return frames


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_backtest_from_files.py <SYMBOL>  (e.g. XAUUSD)")
        sys.exit(1)

    symbol = sys.argv[1]
    frames = load_frames(symbol)
    if "H4" not in frames:
        print("No H4 data — cannot backtest. Run export_history.py first.")
        sys.exit(1)

    print(f"\n=== Backtest: {symbol} ({len(frames['H4'])} H4 bars) ===")
    trades = backtest_symbol(symbol, frames)
    print(f"Trades: {len(trades)}")
    if not trades.empty:
        print(trades[["Signal_Time", "Direction", "Outcome", "R_Multiple", "Planned_RR"]])
    perf = calculate_backtest_performance(trades)
    print("\nPerformance:")
    for k, v in perf.iloc[0].to_dict().items():
        print(f"  {k}: {v}")

    print(f"\n=== Walk-forward: {symbol} ===")
    windows, oos_trades = walk_forward_optimize_symbol(symbol, frames)
    if windows.empty:
        print("Not enough H4 history for a full WFO window "
              "(needs >= WFO_TRAIN_BARS + WFO_TEST_BARS = 750 H4 bars, ~4-5 months).")
    else:
        print(windows.to_string(index=False))
        summary = summarize_wfo_windows(windows)
        print("\nWFO summary:")
        for k, v in summary.iloc[0].to_dict().items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
