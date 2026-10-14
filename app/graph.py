from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from app.nodes.aggregation import aggregate_response
from app.nodes.preprocessing import preprocess_text
from app.nodes.rating import predict_rating
from app.nodes.sentiment import predict_sentiment
from app.nodes.validation import validate_input
from app.state import ReviewState

builder = StateGraph(
    ReviewState
)

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
    "aggregate",
    aggregate_response,
)

builder.add_edge(
    START,
    "validate",
)

builder.add_edge(
    "validate",
    "preprocess",
)

builder.add_edge(
    "preprocess",
    "sentiment",
)

builder.add_edge(
    "preprocess",
    "rating",
)

builder.add_edge(
    ["sentiment", "rating"],
    "aggregate",
)

builder.add_edge(
    "aggregate",
    END,
)

review_graph = builder.compile()