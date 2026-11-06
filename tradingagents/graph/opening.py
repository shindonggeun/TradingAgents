"""Opening statements made at the same time instead of one after another.

In the first round of a debate no speaker has heard the others yet when it is
the first to speak, so the bull and the bear (and the three risk debaters) can
open side by side from the same state. Their openings are merged into the one
debate state the serial graph would have built, in the serial order, with the
count and the last speaker set so the existing routers pick the next round's
first speaker (or the manager) unchanged.

What this gives up: in the serial graph the second and third speakers rebut the
ones before them in round one. With more than one round the rebuttals come back
from round two, where every speaker has read every opening.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context


def _together(state, nodes) -> list[dict]:
    """Call the nodes on the same state, each on its own thread.

    Each thread runs in a copy of this context, so the run config (output
    language, instrument) the nodes read through context variables goes with
    it. A node that raises fails the opening, like a failed serial node.
    """
    with ThreadPoolExecutor(max_workers=len(nodes), thread_name_prefix="opening") as pool:
        futures = [pool.submit(copy_context().run, node, state) for node in nodes]
        return [f.result() for f in futures]


def create_debate_opening(bull_node, bear_node):
    """The bull and the bear open the research debate at the same time."""

    def debate_opening(state) -> dict:
        before = state["investment_debate_state"]
        bull, bear = (r["investment_debate_state"] for r in _together(state, (bull_node, bear_node)))
        return {"investment_debate_state": {
            # the serial order: the bull speaks first
            "history": bull["history"] + "\n" + bear["current_response"],
            "bull_history": bull["bull_history"],
            "bear_history": bear["bear_history"],
            # the bear spoke last, so a second round starts with the bull
            "current_response": bear["current_response"],
            "count": before["count"] + 2,
        }}

    return debate_opening


def create_risk_opening(aggressive_node, conservative_node, neutral_node):
    """The aggressive, conservative and neutral debaters open at the same time."""

    def risk_opening(state) -> dict:
        before = state["risk_debate_state"]
        agg, con, neu = (r["risk_debate_state"]
                         for r in _together(state, (aggressive_node, conservative_node, neutral_node)))
        return {"risk_debate_state": {
            "history": agg["history"] + "\n" + con["current_conservative_response"]
                       + "\n" + neu["current_neutral_response"],
            "aggressive_history": agg["aggressive_history"],
            "conservative_history": con["conservative_history"],
            "neutral_history": neu["neutral_history"],
            # the neutral debater spoke last, so a second round starts with the aggressive one
            "latest_speaker": "Neutral",
            "current_aggressive_response": agg["current_aggressive_response"],
            "current_conservative_response": con["current_conservative_response"],
            "current_neutral_response": neu["current_neutral_response"],
            "count": before["count"] + 3,
        }}

    return risk_opening
