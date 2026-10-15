"""
Main LangGraph workflow for hotel-review analysis.
"""

from langgraph.graph import END, START, StateGraph

from app.nodes.aggregation import aggregate_response
from app.nodes.emotion import detect_emotions
from app.nodes.preprocessing import preprocess_text
from app.nodes.rating import predict_rating
from app.nodes.reconciliation import reconcile_results
from app.nodes.sentiment import predict_sentiment
from app.nodes.validation import validate_input
from app.state import ReviewState


def build_review_graph():
    """
    Build and compile the complete hotel-review workflow.

    Current workflow:

        validate
            |
        preprocess
            |
        +---+--------+
        |   |        |
      sentiment   rating   emotion
        |   |        |
        +---+--------+
            |
        reconcile
            |
        aggregate
    """

    builder = StateGraph(
        ReviewState
    )

    # Register workflow nodes.
    builder.add_node(
        "validate",
        validate_input,
    )

    builder.add_node(
        "preprocess",
        preprocess_text,
    )

    builder.add_node(
        "sentiment",
        predict_sentiment,
    )

    builder.add_node(
        "rating",
        predict_rating,
    )

    builder.add_node(
        "emotion",
        detect_emotions,
    )

    builder.add_node(
        "reconcile",
        reconcile_results,
    )

    builder.add_node(
        "aggregate",
        aggregate_response,
    )

    # Workflow entry.
    builder.add_edge(
        START,
        "validate",
    )

    builder.add_edge(
        "validate",
        "preprocess",
    )

    # Parallel model branches.
    builder.add_edge(
        "preprocess",
        "sentiment",
    )

    builder.add_edge(
        "preprocess",
        "rating",
    )

    builder.add_edge(
        "preprocess",
        "emotion",
    )

    # Wait for all model branches before reconciliation.
    builder.add_edge(
        [
            "sentiment",
            "rating",
            "emotion",
        ],
        "reconcile",
    )

    builder.add_edge(
        "reconcile",
        "aggregate",
    )

    builder.add_edge(
        "aggregate",
        END,
    )

    return builder.compile()


review_graph = build_review_graph()