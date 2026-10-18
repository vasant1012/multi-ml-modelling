"""Central configuration loaded from environment variables."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[1]

class Settings(BaseSettings):
    data_path: str = "data/raw/hotel_reviews.csv"
    text_column: str = "review"
    rating_column: str = "rating"
    summary_min_words: int = 35
    max_review_chars: int = 20_000
    cors_origins: str = "http://localhost:5173"
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    def path(self, value: str) -> Path:
        """Resolve project-relative paths consistently."""
        path = Path(value)
        return path if path.is_absolute() else BASE_DIR / path

settings = Settings()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Existing data settings
    data_path: str = "data/raw/hotel_reviews.csv"
    text_column: str = "review"
    rating_column: str = "rating"

    # Existing API/application settings
    max_review_chars: int = 20_000
    cors_origins: str = "http://localhost:5173"

    # Emotion-model settings
    emotion_model_name: str = (
        "j-hartmann/emotion-english-distilroberta-base"
    )
    emotion_confidence_threshold: float = 0.20
    emotion_max_results: int = 3

    # False downloads the model when it is not cached.
    # True forces Transformers to use locally cached files only.
    hf_local_files_only: bool = False

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        extra="ignore",
    )

    def path(self, value: str) -> Path:
        """Resolve a path relative to the project root."""

        path = Path(value)

        if path.is_absolute():
            return path

        return BASE_DIR / path


settings = Settings()
