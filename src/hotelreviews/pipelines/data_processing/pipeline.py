from kedro.pipeline import Pipeline, node  # type: ignore

from .nodes import (
    analyze_sentiment,
    load_data,
    preprocess_data,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=load_data,
                inputs="hotel_reviews",
                outputs="raw_data",
                name="load_data_node",
            ),
            node(
                func=preprocess_data,
                inputs="raw_data",
                outputs="cleaned_reviews",
                name="preprocess_data_node",
            ),
            node(
                func=analyze_sentiment,
                inputs="cleaned_reviews",
                outputs="sentiment_analysis",
                name="sentiment_analysis_node"
            ),
        ]
    )
