"""
Independently executable emotion-detection subgraph.
"""

from langgraph.graph import END, START, StateGraph

from app.nodes.emotion import detect_emotions
from app.state import ReviewState


def build_emotion_graph():
    """Build and compile the emotion-detection subgraph."""

    builder = StateGraph(ReviewState)

    builder.add_node(
        "detect_emotions",
        detect_emotions,
    )

    builder.add_edge(
        START,
        "detect_emotions",
    )

    builder.add_edge(
        "detect_emotions",
        END,
    )

    return builder.compile()


emotion_graph = build_emotion_graph()