"""
agents/data_agent.py
=====================
The only module in this project allowed to import MetaTrader5.

Ported from fetch_forex_data_v5_FINAL.py: setup_mt5(), ensure_symbol_enabled(),
fetch_rates(), add_indicators(). Logic is unchanged (it was already
correct — see the old file's changelog for the bugs that were already
fixed here in earlier iterations); what changed is packaging:
    - wrapped in a class so it can be constructed once (one MT5 session)
      and reused across symbols instead of relying on module-level state.
    - returns a typed MarketSnapshot instead of a bare dict of DataFrames,
      so downstream agents get an explicit contract (see schemas.py).
    - broad `except Exception: df[col] = np.nan` blocks kept as-is for the
      indicator calculations (safe: a failed indicator degrades to NaN,
      it doesn't hide a fetch failure) but fetch-level failures now raise
      DataFetchError instead of silently returning None, so a bad symbol
      name shows up immediately instead of quietly producing an empty
      run at the very end of the pipeline.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:  # allows importing this module on non-Windows dev
    # machines for type-checking / unit tests that mock the client.
    mt5 = None

from ta.volatility import AverageTrueRange
from ta.trend import ADXIndicator, EMAIndicator

from ..schemas import MarketSnapshot
from ..primitives import enrich_frame

ATR_PERIOD = 14
ADX_PERIOD = 14
EMA_REGIME_PERIOD = 20
EMA_SLOPE_LOOKBACK = 5

# Confirmed against wmmarket demo Market Watch (see /areas/agentic-trading-system.md).
# All symbols below were Visible=True on the account this was checked against.
DEFAULT_TIMEFRAMES_CONFIG = {
    "M5":  {"tf": "TIMEFRAME_M5",  "candles": 3000},
    "M15": {"tf": "TIMEFRAME_M15", "candles": 2500},
    "M30": {"tf": "TIMEFRAME_M30", "candles": 2000},
    "H1":  {"tf": "TIMEFRAME_H1",  "candles": 1500},
    "H4":  {"tf": "TIMEFRAME_H4",  "candles": 1200},
    "D1":  {"tf": "TIMEFRAME_D1",  "candles": 800},
    "W1":  {"tf": "TIMEFRAME_W1",  "candles": 400},
    "MN1": {"tf": "TIMEFRAME_MN1", "candles": 60},
}


class DataFetchError(RuntimeError):
    """Raised on a real fetch failure (bad symbol, MT5 not connected, empty
    response) — deliberately NOT silently swallowed, unlike v5's fetch_rates()
    which returned None and let the caller quietly skip the timeframe."""


class DataAgent:
    def __init__(self, timeframes_config: dict | None = None):
        self.timeframes_config = timeframes_config or DEFAULT_TIMEFRAMES_CONFIG
        self._connected = False

    # -- connection lifecycle ------------------------------------------------

    def connect(self) -> None:
        if mt5 is None:
            raise RuntimeError("MetaTrader5 package not available in this environment.")
        if not mt5.initialize():
            raise DataFetchError(f"MT5 initialize failed: {mt5.last_error()}")
        self._connected = True

    def disconnect(self) -> None:
        if mt5 is not None and self._connected:
            mt5.shutdown()
        self._connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.disconnect()

    # -- symbol handling -------------------------------------------------

    def ensure_symbol_enabled(self, broker_symbol: str) -> None:
        info = mt5.symbol_info(broker_symbol)
        if info is None:
            raise DataFetchError(f"Symbol not found on this account: {broker_symbol}")
        if not info.visible and not mt5.symbol_select(broker_symbol, True):
            raise DataFetchError(f"Could not enable symbol: {broker_symbol}")

    # -- fetch -------------------------------------------------------------

    def _fetch_rates(self, broker_symbol: str, tf_name: str, candles: int) -> pd.DataFrame:
        tf_value = getattr(mt5, self.timeframes_config[tf_name]["tf"])
        rates = mt5.copy_rates_from_pos(broker_symbol, tf_value, 0, candles)
        if rates is None or len(rates) == 0:
            raise DataFetchError(f"No data returned for {broker_symbol} {tf_name}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low", "close": "Close",
            "tick_volume": "Volume", "spread": "Spread", "real_volume": "RealVolume",
        })
        df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

        now_utc = pd.Timestamp.now(tz="UTC")
        df["Is_Current_Bar"] = False
        if len(df):
            df["LastBarAgeSeconds"] = (now_utc - df["time"]).dt.total_seconds()
            df.loc[df.index[-1], "Is_Current_Bar"] = True

        df["Symbol"] = broker_symbol
        return df

    @staticmethod
    def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        try:
            df["ATR"] = AverageTrueRange(
                high=df["High"], low=df["Low"], close=df["Close"], window=ATR_PERIOD
            ).average_true_range()
        except Exception:
            df["ATR"] = np.nan

        try:
            if len(df) >= ADX_PERIOD * 2 + 1:
                df["ADX"] = ADXIndicator(
                    high=df["High"], low=df["Low"], close=df["Close"], window=ADX_PERIOD
                ).adx()
            else:
                df["ADX"] = np.nan
        except Exception:
            df["ADX"] = np.nan

        try:
            df["EMA20"] = EMAIndicator(close=df["Close"], window=EMA_REGIME_PERIOD).ema_indicator()
            df["EMA20_Slope"] = df["EMA20"].diff(EMA_SLOPE_LOOKBACK)
        except Exception:
            df["EMA20"] = np.nan
            df["EMA20_Slope"] = np.nan

        df["Candle_Range"] = df["High"] - df["Low"]
        df["Range_ATR_Ratio"] = df["Candle_Range"] / df["ATR"].replace(0, np.nan)
        df["Return_1"] = df["Close"].pct_change()
        return df

    def fetch(self, symbol: str, broker_symbol: Optional[str] = None,
              knowledge_time: Optional[datetime] = None) -> MarketSnapshot:
        """Fetch all configured timeframes for one symbol.

        symbol: canonical name used internally by other agents/logs.
        broker_symbol: exact MT5 ticker if it differs from `symbol`
            (on wmmarket these are currently identical, incl. the "@" suffix).
        """
        broker_symbol = broker_symbol or symbol
        if not self._connected:
            raise RuntimeError("DataAgent.connect() must be called before fetch().")

        self.ensure_symbol_enabled(broker_symbol)

        frames: dict[str, pd.DataFrame] = {}
        for tf_name, cfg in self.timeframes_config.items():
            raw = self._fetch_rates(broker_symbol, tf_name, cfg["candles"])
            with_indicators = self._add_indicators(raw)
            frames[tf_name] = enrich_frame(with_indicators, tf_name)

        return MarketSnapshot(
            symbol=symbol,
            broker_symbol=broker_symbol,
            knowledge_time=knowledge_time or datetime.now(timezone.utc),
            frames=frames,
        )
