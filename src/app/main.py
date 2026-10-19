from fastapi import FastAPI
from .routers import batches, trainings, models

app = FastAPI(
    title="MRX Asynchronous API",
    version="0.1.0"
)

app.include_router(batches.router)
app.include_router(trainings.router)
app.include_router(models.router)