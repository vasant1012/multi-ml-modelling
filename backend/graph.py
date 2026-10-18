from langgraph.graph import END, START, StateGraph

from backend.nodes.aggregation import aggregate_response
from backend.nodes.aspects import analyze_aspects
from backend.nodes.emotion import detect_emotions
from backend.nodes.preprocessing import preprocess_text
from backend.nodes.rating import predict_rating
from backend.nodes.reconciliation import reconcile_results
from backend.nodes.sentiment import predict_sentiment
from backend.nodes.validation import validate_input
from backend.state import ReviewState


def build_review_graph():
    """
    Build and compile the complete hotel-review workflow.

    Sentiment, rating, emotion, and ABSA run independently after
    preprocessing. LangGraph waits for all four branches before
    running reconciliation.
    """

    builder = StateGraph(ReviewState)

    # --------------------------------------------------------------
    # Register nodes
    # --------------------------------------------------------------

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
        "aspects",
        analyze_aspects,
    )

    builder.add_node(
        "reconcile",
        reconcile_results,
    )

    builder.add_node(
        "aggregate",
        aggregate_response,
    )

    # --------------------------------------------------------------
    # Main sequential flow
    # --------------------------------------------------------------

    builder.add_edge(
        START,
        "validate",
    )

    builder.add_edge(
        "validate",
        "preprocess",
    )

    # --------------------------------------------------------------
    # Parallel analytical branches
    # --------------------------------------------------------------

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

    builder.add_edge(
        "preprocess",
        "aspects",
    )

    # --------------------------------------------------------------
    # Fan-in
    #
    # Reconciliation runs only after all four component nodes finish.
    # --------------------------------------------------------------

    builder.add_edge(
        [
            "sentiment",
            "rating",
            "emotion",
            "aspects",
        ],
        "reconcile",
    )

    # --------------------------------------------------------------
    # Final response creation
    # --------------------------------------------------------------

    builder.add_edge(
        "reconcile",
        "aggregate",
    )

    builder.add_edge(
        "aggregate",
        END,
    )

    return builder.compile()


# Reusable compiled graph for FastAPI requests.
review_graph = build_review_graph()