"""
scripts/export_history.py
===========================
Run this ON THE WINDOWS MACHINE where the MT5 terminal + wmmarket demo
login are active. Connects via DataAgent (the same class the live
system will eventually use), fetches every configured timeframe for
each symbol, and writes each enriched frame to Parquet under ./data/.

Usage:
    python scripts/export_history.py XAUUSD@ EURUSD@ USDJPY@ AUDUSD@ US500CASH

Output:
    data/<symbol>/<timeframe>.parquet   (one file per symbol per timeframe)

These Parquet files are what research/backtest.py and
research/walk_forward.py should be pointed at — see
scripts/run_backtest_from_files.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agentic_ict.agents.data_agent import DataAgent, BACKTEST_TIMEFRAMES_CONFIG

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


def export_symbol(agent: DataAgent, symbol: str) -> None:
    print(f"Fetching {symbol} ...")
    snapshot = agent.fetch(symbol)  # broker_symbol defaults to symbol (wmmarket uses identical names, incl. "@")
    out_dir = OUTPUT_DIR / symbol.replace("@", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    for tf_name, df in snapshot.frames.items():
        path = out_dir / f"{tf_name}.csv"
        df.to_csv(path, index=False)
        print(f"  {tf_name}: {len(df)} bars -> {path}")


def main():
    args = sys.argv[1:]
    backtest_mode = "--backtest" in args
    symbols = [a for a in args if a != "--backtest"]
    if not symbols:
        symbols = ["XAUUSD@", "EURUSD@", "USDJPY@", "AUDUSD@", "US500CASH"]
        print(f"No symbols given on the command line — using the default list: {symbols}")

    config = BACKTEST_TIMEFRAMES_CONFIG if backtest_mode else None
    if backtest_mode:
        print("Using the deep-history backtest profile (H4 up to 8000 bars) — this will take longer.")

    with DataAgent(timeframes_config=config) as agent:
        for symbol in symbols:
            try:
                export_symbol(agent, symbol)
            except Exception as exc:
                print(f"  FAILED for {symbol}: {exc}")

    print(f"\nDone. Files are under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
