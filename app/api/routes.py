"""HTTP routes that invoke the compiled graph."""
from uuid import uuid4
from fastapi import APIRouter
from app.api.schemas import ReviewRequest, ReviewResponse
from app.graph import review_graph

router = APIRouter(prefix="/api/v1", tags=["reviews"])

@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@router.post("/analyze", response_model=ReviewResponse)
def analyze(payload: ReviewRequest) -> ReviewResponse:
    state = {
        "review_id": payload.review_id or str(uuid4()),
        "original_text": payload.text,
        "original_rating": payload.rating,
    }
    if payload.components is not None:
        state["requested_components"] = payload.components
    output = review_graph.invoke(state)
    return ReviewResponse(result=output["final_result"])
