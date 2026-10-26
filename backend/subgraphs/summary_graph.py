"""Independently testable summarization subgraph."""
from langgraph.graph import END, START, StateGraph

from backend.nodes.summarization import summarize_review
from backend.state import ReviewState


def build_graph():
    builder = StateGraph(ReviewState)
    builder.add_node("summarize_review", summarize_review)
    builder.add_edge(START, "summarize_review")
    builder.add_edge("summarize_review", END)
    return builder.compile()
