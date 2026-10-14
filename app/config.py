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
