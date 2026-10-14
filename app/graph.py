"""Parent LangGraph workflow that fans out to five independent subgraphs."""
from langgraph.graph import StateGraph, START, END
from app.state import ReviewState
from app.nodes.validation import validate_input, route_after_validation
from app.nodes.preprocessing import preprocess_text
from app.nodes.reconciliation import reconcile_results
from app.nodes.aggregation import aggregate_results
from app.nodes.sentiment import predict_sentiment
from app.nodes.rating import predict_rating
from app.nodes.emotion import detect_emotions
from app.nodes.aspects import analyze_aspects
from app.nodes.summarization import summarize_review

def build_review_graph():
    builder = StateGraph(ReviewState)
    builder.add_node("validate", validate_input)
    builder.add_node("preprocess", preprocess_text)
    builder.add_node("sentiment", predict_sentiment)
    builder.add_node("rating", predict_rating)
    builder.add_node("emotion", detect_emotions)
    builder.add_node("aspects", analyze_aspects)
    builder.add_node("summary", summarize_review)
    builder.add_node("reconcile", reconcile_results)
    builder.add_node("aggregate", aggregate_results)
    builder.add_edge(START, "validate")
    builder.add_conditional_edges("validate", route_after_validation, {"valid": "preprocess", "invalid": "aggregate"})
    branches = ["sentiment", "rating", "emotion", "aspects", "summary"]
    for branch in branches:
        builder.add_edge("preprocess", branch)
    # List-based fan-in makes reconciliation wait for all five branches.
    builder.add_edge(branches, "reconcile")
    builder.add_edge("reconcile", "aggregate")
    builder.add_edge("aggregate", END)
    return builder.compile()

review_graph = build_review_graph()
