"""
agents/context_agent.py
=========================
Layer 1 of the ICT prompt's decision model: "What direction and market
behavior are permitted?" This is the ONLY agent allowed to set
directional_permission=False — every other agent downstream must respect
that hard stop.

Reads pre-enriched D1/W1 frames from the MarketSnapshot (Regime and
Weekly_Bias/Weekly_Bias_State columns are already computed by
primitives.enrich_frame() inside DataAgent — see primitives.py). This
agent does no computation of its own beyond picking the latest closed
row and applying the prompt's Sec. 3/4 gating rule.
"""

from __future__ import annotations

from ..schemas import MarketSnapshot, ContextView, Direction


def _latest_closed_row(df):
    closed = df[~df["Is_Current_Bar"]]
    return closed.iloc[-1] if not closed.empty else df.iloc[-1]


class ContextAgent:
    def evaluate(self, snapshot: MarketSnapshot) -> ContextView:
        reasons: list[str] = []

        d1 = snapshot.frames.get("D1")
        w1 = snapshot.frames.get("W1")

        if d1 is None or d1.empty:
            return ContextView(
                symbol=snapshot.symbol, knowledge_time=snapshot.knowledge_time,
                d1_regime="Indeterminate", weekly_bias=Direction.NEUTRAL,
                directional_permission=False, reasons=["No D1 data available."],
            )
        if w1 is None or w1.empty:
            return ContextView(
                symbol=snapshot.symbol, knowledge_time=snapshot.knowledge_time,
                d1_regime="Indeterminate", weekly_bias=Direction.NEUTRAL,
                directional_permission=False, reasons=["No W1 data available."],
            )

        last_d1 = _latest_closed_row(d1)
        last_w1 = _latest_closed_row(w1)

        regime = last_d1["Regime"]
        bias_raw = last_w1["Weekly_Bias"]
        weekly_state = last_w1["Weekly_Bias_State"]
        weekly_bias = {"BUY": Direction.BULLISH, "SELL": Direction.BEARISH}.get(
            bias_raw, Direction.NEUTRAL
        )

        # Prompt Sec. 3: D1 regime changes EXPECTATIONS ONLY (expected
        # retracement depth, confirmation strictness) — it does not
        # create direction and must not gate permission. Only Weekly
        # Bias (Sec. 4) gates permission.
        #
        # KNOWN GAP (inherited from v5): the prompt wants three
        # confirmation states (Confirmed / Provisional / Invalidated) —
        # only Confirmed may authorize a Trade Signal, Provisional
        # supports Watchlist only. v5's build_weekly_bias() only tracks
        # binary Confirmed/Unconfirmed, so this agent is currently
        # slightly more permissive than the prompt intends for the
        # Provisional case. Revisit once Provisional is modeled.
        if weekly_bias == Direction.NEUTRAL or weekly_state != "Confirmed":
            reasons.append("Weekly bias is NEUTRAL/unconfirmed — no directional permission (Sec. 4).")
        if regime == "Parabolic":
            reasons.append("D1 regime is Parabolic — informational only (Sec. 3): expect deeper "
                            "retracements and stricter confirmation downstream, not a permission gate.")

        directional_permission = (weekly_bias != Direction.NEUTRAL) and (weekly_state == "Confirmed")

        return ContextView(
            symbol=snapshot.symbol,
            knowledge_time=snapshot.knowledge_time,
            d1_regime=str(regime),
            weekly_bias=weekly_bias,
            directional_permission=directional_permission,
            reasons=reasons,
        )
