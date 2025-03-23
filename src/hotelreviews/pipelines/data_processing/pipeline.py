from kedro.pipeline import Pipeline, node  # type: ignore

from .nodes import (
    analyze_sentiment,
    # create_model_input_table,
    load_data,
    # preprocess_companies,
    preprocess_data,
    # preprocess_shuttles,
)

# def create_pipeline(**kwargs) -> Pipeline:
#     return pipeline(
#         [
#             node(
#                 func=preprocess_companies,
#                 inputs="companies",
#                 outputs="preprocessed_companies",
#                 name="preprocess_companies_node",
#             ),
#             node(
#                 func=preprocess_shuttles,
#                 inputs="shuttles",
#                 outputs="preprocessed_shuttles",
#                 name="preprocess_shuttles_node",
#             ),
#             node(
#                 func=create_model_input_table,
#                 inputs=["preprocessed_shuttles", "preprocessed_companies", "reviews"],
#                 outputs="model_input_table",
#                 name="create_model_input_table_node",
#             ),
#         ]
#     )


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
