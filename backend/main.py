"""FastAPI application entry point for the digital media forensics backend."""

from fastapi import FastAPI

from backend.api.routes import router


app = FastAPI(
    title="Digital Media Forensics API",
    version="0.1.0",
)

app.include_router(router)
