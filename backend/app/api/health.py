from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    """Liveness check. Returns service identity only — no simulated metrics."""
    return {"status": "ok", "service": "ugv-backend"}
