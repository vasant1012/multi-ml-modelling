"""Independently testable aspects subgraph."""
from langgraph.graph import StateGraph, START, END
from app.state import ReviewState
from app.nodes.aspects import analyze_aspects

def build_graph():
    builder = StateGraph(ReviewState)
    builder.add_node("analyze_aspects", analyze_aspects)
    builder.add_edge(START, "analyze_aspects")
    builder.add_edge("analyze_aspects", END)
    return builder.compile()
