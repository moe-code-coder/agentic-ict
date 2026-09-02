"""
orchestrator.py
================
Implements Section 1 of Professional_ICT_Trading_Prompt_v4 ("OPTIMIZED
EXECUTION ORDER") as actual control flow, instead of leaving it as a
diagram a human has to remember to follow.

Non-negotiable rule (from the prompt, enforced here structurally, not
just by convention):
    A lower layer can delay or reject a trade, but cannot override a
    higher-layer invalidation.

That's why each step below short-circuits to WATCHLIST/WAIT/INVALID the
moment a gate fails, and never lets a later agent see a symbol whose
context already rejected it.

This file intentionally contains NO ICT logic itself (no FVG/OB/sweep
math, no scoring formulas). That logic is being ported from
fetch_forex_data_v5_FINAL.py into agents/*.py one function at a time —
see the TODO markers. This file only encodes the *sequencing*.
"""

from __future__ import annotations
from datetime import datetime, timezone

from .schemas import (
    MarketSnapshot, ContextView, SetupState, ExecutionSignal, RiskVerdict,
    Decision,
)

# Agents are imported lazily inside run() bodies for now, since they're
# still stubs — this keeps the orchestrator importable/testable before
# every agent exists.


class Orchestrator:
    def __init__(self, data_agent, context_agent, setup_agent,
                 execution_agent, risk_agent, broker_agent, narrator_agent=None):
        self.data_agent = data_agent
        self.context_agent = context_agent
        self.setup_agent = setup_agent
        self.execution_agent = execution_agent
        self.risk_agent = risk_agent
        self.broker_agent = broker_agent
        self.narrator_agent = narrator_agent   # optional: LLM commentary layer, never a decision-maker

    def run_symbol(self, symbol: str) -> RiskVerdict | ContextView | SetupState:
        """Run one symbol through the full gate hierarchy.

        Returns whatever layer the symbol got stopped at, so the caller
        (and the human) can always see exactly where a candidate died.
        """
        now = datetime.now(timezone.utc)

        # --- Layer 0: data --------------------------------------------------
        snapshot: MarketSnapshot = self.data_agent.fetch(symbol, knowledge_time=now)

        # --- Layer 1: context (D1 regime, weekly bias, directional permission)
        context: ContextView = self.context_agent.evaluate(snapshot)
        if not context.directional_permission:
            return context  # hard stop — nothing below may run. TODO: log to watchlist store.

        # --- Layer 2: setup (H4 liquidity map, sweep -> MSS -> BOS, entry zone)
        setup: SetupState = self.setup_agent.evaluate(context, snapshot)
        if setup.decision in (Decision.WATCHLIST, Decision.INVALID):
            return setup

        # --- Layer 3: execution (M30 -> M15 -> M5, session, liquidity, RR)
        signal: ExecutionSignal = self.execution_agent.evaluate(setup, snapshot)
        if signal.decision in (Decision.WATCHLIST, Decision.WAIT, Decision.INVALID):
            return signal

        # --- Layer 4: risk (position size, RR gate, portfolio correlation)
        verdict: RiskVerdict = self.risk_agent.evaluate(signal)

        # --- Human-in-the-loop gate (mandatory during the supervised phase) --
        if verdict.approved and verdict.requires_human_confirmation:
            if self.narrator_agent is not None:
                verdict.reasons.append(self.narrator_agent.explain(verdict))
            # BrokerAgent must NOT place a live order here on its own.
            # It only ever prepares the order + waits for explicit human ack.
            self.broker_agent.stage_for_human_review(verdict)
        elif verdict.approved:
            # Autonomous execution path — intentionally not wired up yet.
            # Do not enable until the human-supervised phase has enough
            # tracked outcomes to justify it (this is a project decision,
            # not a technical one — see /areas/agentic-trading-system.md).
            raise NotImplementedError(
                "Autonomous (non-human-reviewed) execution is disabled by design."
            )

        return verdict

    def run_universe(self, symbols: list[str]) -> dict:
        return {s: self.run_symbol(s) for s in symbols}
