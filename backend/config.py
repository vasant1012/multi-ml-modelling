from pathlib import Path

from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

# ==========================================================
# Project Paths
# ==========================================================

"""
Central application configuration.

All paths are resolved from the project root.

Project Structure

Hotel-Review-Intelligence/

├── Backend/
│   ├── backend/
│   ├── models/
│   ├── training/
│   ├── notebooks/
│   └── tests/
│
├── Frontend/
├── Data/
└── Logs/
"""

# Backend/backend/config.py
#            ↑
# parents[0] = backend
# parents[1] = Backend
# parents[2] = Project Root


# ==========================================================
# Project Root Detection
# ==========================================================

PROJECT_ROOT = Path.cwd()

BACKEND_DIR = (
    PROJECT_ROOT
    / "backend"
)

FRONTEND_DIR = (
    PROJECT_ROOT
    / "frontend"
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

LOGS_DIR = (
    PROJECT_ROOT
    / "logs"
)

MODELS_DIR = (
    BACKEND_DIR
    / "models"
)

TRAINING_DIR = (
    BACKEND_DIR
    / "training"
)

TESTS_DIR = (
    BACKEND_DIR
    / "tests"
)

NOTEBOOKS_DIR = (
    BACKEND_DIR
    / "notebooks"
)


# ==========================================================
# Application Settings
# ==========================================================

class Settings(BaseSettings):
    """
    Application settings loaded
    from the .env file.
    """

    # ------------------------------------------------------
    # Data
    # ------------------------------------------------------

    data_path: str = (
        "data/raw/hotel_reviews.csv"
    )

    text_column: str = "review"

    rating_column: str = "rating"

    # ------------------------------------------------------
    # Review Processing
    # ------------------------------------------------------

    max_review_chars: int = 20_000

    summary_min_words: int = 35

    # ------------------------------------------------------
    # FastAPI / CORS
    # ------------------------------------------------------

    cors_origins: str = (
        "http://localhost:5173"
    )

    # ------------------------------------------------------
    # Emotion Detection
    # ------------------------------------------------------

    emotion_model_name: str = (
        "j-hartmann/emotion-english-distilroberta-base"
    )

    emotion_confidence_threshold: float = (
        0.20
    )

    emotion_max_results: int = 3

    hf_local_files_only: bool = True

    # ------------------------------------------------------
    # Environment
    # ------------------------------------------------------

    model_config = (
        SettingsConfigDict(
            env_file=[
                PROJECT_ROOT / ".env",
                BACKEND_DIR / ".env",
            ],
            extra="ignore",
        )
    )

    # ------------------------------------------------------
    # Path Resolver
    # ------------------------------------------------------

    def path(
        self,
        value: str,
    ) -> Path:
        """
        Resolve any project-relative path.

        The repository layout is a mix of legacy and current naming,
        so we resolve from the project root first and then fall back to
        the backend directory for model/training resources.
        """

        path = Path(value)

        if path.is_absolute():

            return path

        candidates = [
            PROJECT_ROOT / path,
        ]

        if path.parts and path.parts[0] == "backend":
            candidates.append(
                PROJECT_ROOT / path.relative_to("backend")
            )
        elif path.parts and path.parts[0] in {
            "models",
            "training",
            "tests",
            "notebooks",
        }:
            candidates.append(
                BACKEND_DIR / path
            )
        elif path.parts and path.parts[0] in {"data", "logs"}:
            candidates.append(
                PROJECT_ROOT / path
            )

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[0]


settings = Settings()


if __name__ == "__main__":
    settings = Settings()