"""Independently testable summarization subgraph."""
from langgraph.graph import StateGraph, START, END
from app.state import ReviewState
from app.nodes.summarization import summarize_review

def build_graph():
    builder = StateGraph(ReviewState)
    builder.add_node("summarize_review", summarize_review)
    builder.add_edge(START, "summarize_review")
    builder.add_edge("summarize_review", END)
    return builder.compile()
