"""Independently testable sentiment subgraph."""
from langgraph.graph import StateGraph, START, END
from app.state import ReviewState
from app.nodes.sentiment import predict_sentiment

def build_graph():
    builder = StateGraph(ReviewState)
    builder.add_node("predict_sentiment", predict_sentiment)
    builder.add_edge(START, "predict_sentiment")
    builder.add_edge("predict_sentiment", END)
    return builder.compile()
