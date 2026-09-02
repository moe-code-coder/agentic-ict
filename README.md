# Agentic ICT Trading System — v6 (rewrite)

## Setup

```
pip install -r requirements.txt
```

MetaTrader5 only installs/works on Windows — that's where the terminal
and wmmarket demo login live, so `DataAgent`, `scripts/export_history.py`,
and eventually the live orchestrator all run there. `research/` (backtest,
walk-forward) has no MT5 dependency and can run anywhere once you have
exported Parquet files (see below).

## Workflow so far

```
# On the Windows machine with MT5 + wmmarket demo running:
python scripts/export_history.py XAUUSD@ EURUSD@ USDJPY@ AUDUSD@ US500CASH

# Anywhere, using the exported data:
python scripts/run_backtest_from_files.py XAUUSD
```

## Why a rewrite instead of refactoring v5

`fetch_forex_data_v5_FINAL.py` (8368 lines) has real, tested ICT/SMC logic,
but ~30% of it (portfolio risk, position sizing, shadow trading, and a
whole self-auditing subsystem built with `ast`) is never called from the
actual run path, and everything lives in one file with a hard dependency
on `MetaTrader5`. The core ICT logic is worth keeping; the file is not.

## Agent boundaries (mirrors the 4-layer model in the ICT prompt)

```
DataAgent → ContextAgent → SetupAgent → ExecutionAgent → RiskAgent → BrokerAgent
                                                              ↑
                                                      NarratorAgent (optional, explains — never decides)
```

Each arrow is a typed object from `schemas.py`. An agent may only read the
object(s) handed to it — never reach into a lower layer's internals — so
the "lower layer can't override a higher layer" rule from the prompt is
enforced by the type signatures, not just by discipline.

## Port map: old function → new home

| Old function (fetch_forex_data_v5_FINAL.py) | New home |
|---|---|
| `setup_mt5`, `ensure_symbol_enabled`, `fetch_rates`, `add_indicators` | `agents/data_agent.py` |
| `classify_regime`, `build_weekly_bias` | `agents/context_agent.py` |
| `detect_pivots`, `build_confirmed_liquidity_levels`, `detect_fvg`, `detect_order_blocks`, `detect_structure_events`, `build_setup_events`, `find_latest_zone` | `agents/setup_agent.py` |
| `infer_ltf_context`, `tag_session`, `assess_liquidity`, `execution_score`, `estimate_trade_plan`, `structural_rr` | `agents/execution_agent.py` |
| `calculate_position_size`, `validate_risk_plan`, `portfolio_risk_snapshot`, `approve_portfolio_entry`, `correlation_adjusted_risk` (previously dead code — now actually wired in) | `agents/risk_agent.py` |
| *(new — old file had no real broker execution)* | `agents/broker_agent.py` |
| *(new)* | `agents/narrator_agent.py` — Claude API call that turns a `RiskVerdict` into a plain-language rationale for the human reviewer; must never change the decision, only narrate it |
| `backtest_symbol`, `run_end_to_end_backtest`, `walk_forward_optimize_symbol`, `statistical_validation` | kept, but as an offline `research/` module, not part of the live agent loop |
| Everything under phases 19–25 (self-audit via `ast`, release manifests, shadow trading never wired up) | **dropped** — reintroduce only if a specific need shows up |

## Not yet decided (will need your input, not urgent)

- Orchestration mechanism: current `orchestrator.py` is plain Python
  (easiest to debug, no new dependency). A framework (LangGraph etc.)
  only earns its cost if we need parallel multi-symbol runs with retries —
  can add later without changing `schemas.py`.
- Exact wmmarket symbol names and server UTC offset (old file has
  `XAUUSD@`, `US500CASH`, offset=3 — these were for a different broker
  and need to be confirmed against your MT5 Market Watch).

## Status

All core agents are implemented and wired end-to-end (verified with a
synthetic-data smoke test, `smoke_test/run_full_chain.py` — not part of
the deliverable, just a sanity check):

- `primitives.py` — shared enrichment (regime, pivots, FVG, OB, sweep/MSS/BOS, session, PD-zone, weekly bias)
- `agents/data_agent.py` — MT5 fetch + indicators + enrichment
- `agents/context_agent.py` — D1 regime + Weekly Bias -> directional permission
- `agents/setup_agent.py` — H4 Sweep -> MSS -> BOS -> entry zone, quality score
- `agents/execution_agent.py` — M30/M15/M5 trigger hierarchy, RR gate, trade plan
- `agents/risk_agent.py` — position sizing + portfolio admission gate (the previously-dead v5 risk engine, now actually called)
- `agents/broker_agent.py` — stages orders for human review; a *separate, explicit* method is required to actually send an order — Orchestrator never calls it
- `orchestrator.py` — wires all of the above in the prompt's gate order
- `research/backtest.py` — event-driven H4 backtest, with the FVG
  mitigation look-ahead leak (found during code review, see below)
  fixed via `mitigation_status_asof()`
- `research/walk_forward.py` — chronological train/freeze/test WFO with
  a small (Min_RR x Execution_Min) parameter grid

## Backtest validity caveat (important — read before trusting any numbers)

The backtest/WFO modules were only smoke-tested against **synthetic
random-walk data** (pure geometric Brownian motion), which produced
**zero Sweep events across 2200 H4 bars** — FVG/OB fired normally (147/109),
but a Sweep requires a genuine fakeout (wick through a level, then close
back inside it), a behavior pure GBM rarely produces on its own. So the
"0 trades" result proves the pipeline doesn't crash, NOT that the
strategy has no edge or that the code is broken. Any real performance
read (win rate, expectancy, WFO out-of-sample results) requires running
this against **actual wmmarket historical bars** exported from MT5, not
synthetic data — that's the necessary next step before this module's
output means anything.

## Known gaps / next steps (in rough priority order)

1. **wmmarket broker offset not confirmed.** `BROKER_UTC_OFFSET_HOURS = 3`
   in `primitives.py` is inherited from the old broker and affects
   session tagging (a real ICT input) — needs verification against the
   demo account's actual server time.
2. **Weekly Bias only has binary Confirmed/Unconfirmed**, not the
   prompt's 3-state Confirmed/Provisional/Invalidated model — flagged
   in `context_agent.py`.
3. **No correlation-adjusted portfolio risk yet** — `RiskAgent` runs
   with `correlation_matrix=None` (v5's own explicit fallback), since
   that needs a cross-symbol pass across the whole universe before a
   single-symbol risk check can use it.
4. **`value_per_price_unit_map`** (needed for correct position sizing)
   is only stubbed for XAUUSD in the smoke test — needs one real,
   broker-confirmed value per traded symbol before this can size a real
   position.
5. **No live run against the actual wmmarket MT5 terminal yet** —
   everything has only been exercised against synthetic data. First
   live run will likely surface integration issues (symbol lookup,
   empty-frame edge cases, timezone assumptions).
6. **`agents/narrator_agent.py`** (optional LLM rationale layer) and
   the actual human-review UI (something has to show `BrokerAgent.pending_orders()`
   to a person) are not built yet.
7. **Next concrete step: run the backtest against real wmmarket history**,
   not synthetic data (see validity caveat above). This means exporting
   enough H4 (+ D1/W1/M30/M15/M5) history from the MT5 demo terminal —
   either live via `DataAgent` once connected, or from MT5's own
   History Center export — and running `research/backtest.backtest_symbol()`
   / `research/walk_forward.walk_forward_optimize_symbol()` against it.
8. `statistical_validation()` (v5 had this — deeper significance testing
   beyond win rate/expectancy) has not been ported yet; worth adding
   once real-data backtest results exist to actually validate.
