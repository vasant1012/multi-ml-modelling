"""Independently testable sentiment subgraph."""
from langgraph.graph import END, START, StateGraph

from backend.nodes.sentiment import predict_sentiment
from backend.state import ReviewState


def build_graph():
    builder = StateGraph(ReviewState)
    builder.add_node("predict_sentiment", predict_sentiment)
    builder.add_edge(START, "predict_sentiment")
    builder.add_edge("predict_sentiment", END)
    return builder.compile()
