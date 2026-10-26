"""Independently testable rating subgraph."""
from langgraph.graph import END, START, StateGraph

from backend.nodes.rating import predict_rating
from backend.state import ReviewState


def build_graph():
    builder = StateGraph(ReviewState)
    builder.add_node("predict_rating", predict_rating)
    builder.add_edge(START, "predict_rating")
    builder.add_edge("predict_rating", END)
    return builder.compile()
