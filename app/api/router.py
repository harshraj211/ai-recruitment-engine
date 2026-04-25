from fastapi import APIRouter

from app.api.routes.matching import router as matching_router
from app.api.routes.system import router as system_router

api_router = APIRouter()
api_router.include_router(system_router, tags=["system"])
api_router.include_router(matching_router, tags=["matching"])
