"""
FastAPI Application Entry Point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from hacktronix.api.routes import router

app = FastAPI(
    title="HackModel-AI – World Modeling for Autonomous Agents",
    description="Persistent Self-Correcting World Model | HackTronix 2.0 Track B",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "name": "HackModel-AI",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }
