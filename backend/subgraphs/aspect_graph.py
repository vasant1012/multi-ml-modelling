"""Independently testable aspects subgraph."""
from langgraph.graph import END, START, StateGraph

from backend.nodes.aspects import analyze_aspects
from backend.state import ReviewState


def build_graph():
    builder = StateGraph(ReviewState)
    builder.add_node("analyze_aspects", analyze_aspects)
    builder.add_edge(START, "analyze_aspects")
    builder.add_edge("analyze_aspects", END)
    return builder.compile()
