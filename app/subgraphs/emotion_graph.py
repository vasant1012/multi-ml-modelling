"""Independently testable emotion subgraph."""
from langgraph.graph import StateGraph, START, END
from app.state import ReviewState
from app.nodes.emotion import detect_emotions

def build_graph():
    builder = StateGraph(ReviewState)
    builder.add_node("detect_emotions", detect_emotions)
    builder.add_edge(START, "detect_emotions")
    builder.add_edge("detect_emotions", END)
    return builder.compile()
