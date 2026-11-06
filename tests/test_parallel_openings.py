"""The first round's speakers can open at the same time (debate / risk).

The merged state must be the one the serial graph would have built after its
first round (same history order, prefixes, count), so the routers pick the next
speaker unchanged and readers of the state (managers, UIs diffing it) see the
same fields. Off by default: the graph is then exactly the serial one.
"""
from __future__ import annotations

import contextvars
import threading
from unittest.mock import MagicMock

import pytest

from tests.test_graph_end_to_end import TRADE_DATE, ScriptedModel, _graph, offline  # noqa: F401
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.opening import create_debate_opening, create_risk_opening

_REPORTS = {
    "company_of_interest": "AAPL", "asset_type": "stock",
    "market_report": "m", "sentiment_report": "s",
    "news_report": "n", "fundamentals_report": "f",
}


def _llm(answer: str, prompts: list):
    llm = MagicMock()
    llm.invoke.side_effect = lambda prompt: prompts.append(prompt) or MagicMock(content=answer)
    return llm


def _debate_state():
    return {**_REPORTS, "investment_debate_state": {
        "history": "", "bull_history": "", "bear_history": "", "current_response": "", "count": 0}}


def _risk_state():
    return {**_REPORTS, "trader_investment_plan": "plan", "risk_debate_state": {
        "history": "", "aggressive_history": "", "conservative_history": "", "neutral_history": "",
        "latest_speaker": "", "current_aggressive_response": "", "current_conservative_response": "",
        "current_neutral_response": "", "count": 0}}


def _serial(state, key, nodes):
    for node in nodes:
        state = {**state, **node(state)}
    return state[key]


@pytest.mark.unit
def test_the_debate_opening_merges_like_the_serial_round_without_the_rebuttal():
    prompts: list = []
    bull, bear = create_bull_researcher(_llm("up", prompts)), create_bear_researcher(_llm("down", prompts))

    merged = create_debate_opening(bull, bear)(_debate_state())["investment_debate_state"]
    serial = _serial(_debate_state(), "investment_debate_state", (bull, bear))

    assert merged == serial == {
        "history": "\nBull Analyst: up\nBear Analyst: down", "bull_history": "\nBull Analyst: up",
        "bear_history": "\nBear Analyst: down", "current_response": "Bear Analyst: down", "count": 2}
    # ...but in parallel both speakers opened: neither heard the other.
    assert sum("has not spoken yet" in p for p in prompts[:2]) == 2


@pytest.mark.unit
def test_the_risk_opening_merges_like_the_serial_round():
    prompts: list = []
    nodes = (create_aggressive_debator(_llm("a", prompts)), create_conservative_debator(_llm("c", prompts)),
             create_neutral_debator(_llm("n", prompts)))

    merged = create_risk_opening(*nodes)(_risk_state())["risk_debate_state"]
    serial = _serial(_risk_state(), "risk_debate_state", nodes)

    for key in ("history", "aggressive_history", "conservative_history", "neutral_history",
                "latest_speaker", "count", "current_aggressive_response",
                "current_conservative_response", "current_neutral_response"):
        assert merged[key] == serial[key], key
    assert merged["history"] == "\nAggressive Analyst: a\nConservative Analyst: c\nNeutral Analyst: n"
    assert (merged["latest_speaker"], merged["count"]) == ("Neutral", 3)


@pytest.mark.unit
@pytest.mark.parametrize("rounds,debate_next,risk_next", [
    (1, "Research Manager", "Portfolio Manager"),
    (2, "Bull Researcher", "Aggressive Analyst"),     # round two runs serially, in the usual order
])
def test_the_routers_continue_from_an_opening_unchanged(rounds, debate_next, risk_next):
    logic = ConditionalLogic(max_debate_rounds=rounds, max_risk_discuss_rounds=rounds)
    prompts: list = []
    debate = create_debate_opening(create_bull_researcher(_llm("up", prompts)),
                                   create_bear_researcher(_llm("down", prompts)))(_debate_state())
    risk = create_risk_opening(create_aggressive_debator(_llm("a", prompts)),
                               create_conservative_debator(_llm("c", prompts)),
                               create_neutral_debator(_llm("n", prompts)))(_risk_state())

    assert logic.should_continue_debate(debate) == debate_next
    assert logic.should_continue_risk_analysis(risk) == risk_next


@pytest.mark.unit
def test_the_openers_run_at_the_same_time_and_keep_the_run_s_context():
    """Each opener waits for the others: run one after another they would time out.
    A context variable set by the run (as the output language is) reaches each thread."""
    run_value = contextvars.ContextVar("run_value", default="unset")
    run_value.set("Korean")
    barrier, seen = threading.Barrier(3, timeout=5), []

    def node(name):
        def call(state):
            barrier.wait()
            seen.append(run_value.get())
            return {"risk_debate_state": {**state["risk_debate_state"], f"{name}_history": "\n" + name,
                                          "history": "\n" + name, f"current_{name}_response": name}}
        return call

    merged = create_risk_opening(node("aggressive"), node("conservative"), node("neutral"))(_risk_state())

    assert seen == ["Korean"] * 3
    assert merged["risk_debate_state"]["count"] == 3


@pytest.mark.unit
def test_a_failed_opener_fails_the_opening():
    def broken(state):
        raise RuntimeError("provider unavailable")

    ok = create_bull_researcher(_llm("up", []))
    with pytest.raises(RuntimeError, match="provider unavailable"):
        create_debate_opening(ok, broken)(_debate_state())


def _edges(graph):
    return {(e.source, e.target) for e in graph.graph.get_graph().edges}


@pytest.mark.unit
def test_off_by_default_the_graph_is_the_serial_one(tmp_path, monkeypatch, offline):  # noqa: F811
    graph = _graph(tmp_path, monkeypatch, ScriptedModel())
    nodes = set(graph.graph.get_graph().nodes)

    assert not {"Debate Opening", "Risk Opening"} & nodes
    assert ("Trader", "Aggressive Analyst") in _edges(graph)
    assert ("Memory Log", "Bull Researcher") in _edges(graph)


@pytest.mark.unit
def test_with_both_openings_a_run_reaches_its_decision(tmp_path, monkeypatch, offline):  # noqa: F811
    model = ScriptedModel()
    graph = _graph(tmp_path, monkeypatch, model,
                   debate_parallel_first_round=True, risk_parallel_first_round=True)
    edges = _edges(graph)
    assert ("Trader", "Risk Opening") in edges and ("Memory Log", "Debate Opening") in edges

    state, signal = graph.propagate("NVDA", TRADE_DATE)

    assert signal == "Overweight"
    debate, risk = state["investment_debate_state"], state["risk_debate_state"]
    assert debate["count"] == 2 and debate["bull_history"].strip() and debate["bear_history"].strip()
    assert risk["count"] == 3 and risk["latest_speaker"] == "Judge"
    for key in ("aggressive_history", "conservative_history", "neutral_history"):
        assert risk[key].strip(), key


@pytest.mark.unit
def test_the_options_change_the_checkpoint_signature():
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    g = object.__new__(TradingAgentsGraph)
    g.selected_analysts = ("market",)
    g.config = {"max_debate_rounds": 1, "max_risk_discuss_rounds": 1}
    base = g._run_signature("stock")
    for key in ("debate_parallel_first_round", "risk_parallel_first_round"):
        g.config = {"max_debate_rounds": 1, "max_risk_discuss_rounds": 1, key: True}
        assert g._run_signature("stock") != base, key
