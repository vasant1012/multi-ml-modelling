"""Pydantic API request and response schemas."""
from typing import Any
from pydantic import BaseModel, Field

class ReviewRequest(BaseModel):
    review_id: str | None = None
    text: str = Field(min_length=1)
    rating: int | None = Field(default=None, ge=1, le=5)
    components: list[str] | None = None

class ReviewResponse(BaseModel):
    result: dict[str, Any]
