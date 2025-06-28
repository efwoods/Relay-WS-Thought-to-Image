# This file defines the complete MVP architecture for Neural Nexus: Simulation API, Relay API, and Frontend structure.

# ============================
# SIMULATION API - WebSocket Server
# ============================
# Accepts image, encodes it, decodes to waveform, sends waveform_latent to Relay API

# app/main.py
import asyncio
import websockets
import torch
import json
import base64
import numpy as np
from torchvision import transforms
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from core.config import settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

import uvicorn
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from fastapi import Request

from contextlib import asynccontextmanager

# Configurations & Metrics
from core.config import settings
from core.monitoring import metrics
from core.logging import logger

import redis

# API Routes
from api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize Whisper model
    load_dotenv()
    redis_client = redis.Redis(
        host=settings.REDIS_HOST,  # e.g. "localhost" or a container name like "redis"
        port=settings.REDIS_PORT,  # usually 6379
        password=settings.REDIS_PASSWORD,
        decode_responses=True,  # Automatically decode UTF-8 strings
    )

    app.state.redis = redis_client  # ✅ make available globally

    yield  # Application runs here

    # Shutdown: (optional cleanup)
    # e.g., release resources or shutdown thread pools


app = FastAPI(
    title="Relay Neural Waveform Latents to Reconstructed Image Websocket",
    root_path="/relay-waveform-latent-to-image-reconstruction-api",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/reconstruct", tags=["Reconstruct"])


@app.get("/")
async def root(request: Request):
    return RedirectResponse(url=request.scope.get("root_path", "") + "/docs")


@app.get("/health")
async def health():
    metrics.health_requests.inc()
    return {"status": "healthy"}


@app.router.get("/metrics")
def metrics_endpoint():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.FASTAPI_PORT)
