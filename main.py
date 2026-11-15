from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.errors import register_exception_handlers
from app.modules.events.router import router as events_router

app = FastAPI(title="SAR Change Watch")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_methods=["GET"],
    allow_headers=["*"],
)

register_exception_handlers(app)
app.include_router(events_router, prefix="/api/v1")

settings.events_dir.mkdir(parents=True, exist_ok=True)
# Rendered layers (before.png / after.png / change.png) are plain files — no reason to proxy them.
app.mount("/files", StaticFiles(directory=settings.data_dir), name="files")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
