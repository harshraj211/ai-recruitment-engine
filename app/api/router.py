from fastapi import APIRouter

from app.api.routes.data_sources import router as data_sources_router
from app.api.routes.generate_jd import router as generate_jd_router
from app.api.routes.matching import router as matching_router
from app.api.routes.system import router as system_router

api_router = APIRouter()
api_router.include_router(system_router, tags=["system"])
api_router.include_router(data_sources_router, tags=["data sources"])
api_router.include_router(matching_router, tags=["matching"])
api_router.include_router(generate_jd_router, tags=["generate"])
