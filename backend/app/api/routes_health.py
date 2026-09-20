from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    # Readiness checks (DB/Qdrant/Redis/Flagsmith/Langfuse) added in Phase 6
    return {"status": "ok"}
