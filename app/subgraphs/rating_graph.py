"""Independently testable rating subgraph."""
from langgraph.graph import StateGraph, START, END
from app.state import ReviewState
from app.nodes.rating import predict_rating

def build_graph():
    builder = StateGraph(ReviewState)
    builder.add_node("predict_rating", predict_rating)
    builder.add_edge(START, "predict_rating")
    builder.add_edge("predict_rating", END)
    return builder.compile()
