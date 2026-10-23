from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.app.core.config import settings
from backend.app.api.endpoints import router as api_router
from backend.app.services.model_service import model_service
from backend.app.services.dataset_service import dataset_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warming verification on startup
    print(f"[NASA-RUL API] Initializing backend services...")
    print(f"[NASA-RUL API] Loaded models: {model_service.get_loaded_models_status()}")
    print(f"[NASA-RUL API] Scaler ready: {model_service.scaler is not None}")
    print(f"[NASA-RUL API] Dataset ready: {dataset_service.is_loaded()}")
    yield
    print("[NASA-RUL API] Shutting down backend services.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production-grade predictive maintenance API for turbofan engine Remaining Useful Life estimation on NASA C-MAPSS FD001.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration — supports all origins (*) by default, or configured FRONTEND_ORIGIN
if settings.FRONTEND_ORIGIN == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = [o.strip() for o in settings.FRONTEND_ORIGIN.split(",") if o.strip()]
    for local_origin in settings.CORS_ORIGINS:
        if local_origin not in allowed_origins:
            allowed_origins.append(local_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,  # Wildcard origins cannot be combined with allow_credentials=True in CORS spec
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global safe exception handler: never leak local system paths or raw stack traces
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = str(exc)
    # Strip any accidental local path occurrences
    for sensitive in ["K:\\", "k:\\", "C:\\", "c:\\", "Users"]:
        if sensitive in error_msg:
            error_msg = "An internal processing error occurred."
            break
    return JSONResponse(
        status_code=500,
        content={"detail": error_msg, "status": "error"}
    )

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {
        "project": settings.PROJECT_NAME,
        "docs": "/docs",
        "health": "/api/health",
        "status": "operational"
    }

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0" if os.getenv("RENDER") else "127.0.0.1")
    uvicorn.run("backend.app.main:app", host=host, port=port, reload=False if os.getenv("RENDER") else True)
