from contextlib import asynccontextmanager
from pathlib import Path

from core.db import engine
from core.jobs import metadata
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from .routers import batches, trainings, models

STATIC_DIR = Path(__file__).parent / "static"
OPENAPI_URL = "/openapi.json"

@asynccontextmanager
async def lifespan(_: FastAPI):
    metadata.create_all(engine)
    yield

app = FastAPI(
    title="MRX Asynchronous API",
    version="0.1.0",
    lifespan=lifespan,
    openapi_url=OPENAPI_URL,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/docs", include_in_schema=False)
def swagger_ui():
    return get_swagger_ui_html(
        openapi_url=OPENAPI_URL,
        title=f"{app.title} - Swagger UI",
        swagger_js_url="/static/swagger/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger/swagger-ui.css",
        swagger_favicon_url="",
    )

app.include_router(batches.router)
app.include_router(trainings.router)
app.include_router(models.router)