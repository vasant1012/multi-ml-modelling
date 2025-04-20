"""Project pipelines."""

from kedro.pipeline import Pipeline

from hotelreviews.pipelines.data_processing import pipeline as feature_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
         A mapping from pipeline names to ``Pipeline`` objects.
    """
    return {
        "__default__": feature_pipeline.create_pipeline(),
        "feature_pipeline": feature_pipeline.create_pipeline(),  # <-- this is the key Kedro is complaining about
    }
