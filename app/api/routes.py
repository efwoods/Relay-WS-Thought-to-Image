# api/routes.py

from fastapi import APIRouter
from fastapi import WebSocket, WebSocketDisconnect
from core.monitoring import metrics
from core.config import settings
from core.logging import logger
import json
import torch
import base64
import io
import websockets
import transforms
from io import BytesIO

from core.config import settings

from service.reconstruct import (
    preprocess_image_from_websocket,
    reconstruct_image_from_waveform_latents,
)

router = APIRouter()


@router.websocket("/ws/reconstruct-image-from-waveform-latent")
async def simulate(websocket: WebSocket):
    redis_client = websocket.app.state.redis
    await websocket.accept()
    try:
        async for message in websocket:
            waveform_latent, skip_connections, request = (
                preprocess_image_from_websocket(message)
            )

            reconstructed_image = reconstruct_image_from_waveform_latents(
                waveform_latent, skip_connections
            )

            # Convert to base64 to send back to client
            image_pil = transforms.ToPILImage()(reconstructed_image.squeeze().cpu())
            buf = BytesIO()
            image_pil.save(buf, format="PNG")
            image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            # Respond to simulation client
            await websocket.send_json(
                {
                    "status": "success",
                }
            )

            # forward to frontend if available (place in Redis Cache)
            # settings.THOUGHT_TO_IMAGE_REDIS_KEY
            redis_key = f"reconstructed:{settings.THOUGHT_TO_IMAGE_REDIS_KEY}"
            redis_value = json.dumps(
                {
                    "type": "reconstructed_image",
                    "session_id": request.get("session_id", "anonymous"),
                    "image_base64": f"data:image/png;base64,{image_base64}",
                }
            )
            redis_client.set(
                redis_key,
                redis_value,
                ex=600,
            )
            metrics.visual_thoughts_rendered.inc()

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.exception("WebSocket error in image reconstruction:")
        metrics.websocket_errors.inc()


@router.get("/ws-info", tags=["Reconstruct"])
async def websocket_info():
    return {
        "endpoint": "/simulate/ws/simulate-image-to-waveform-latent",
        "full_url": "ws://localhost:8000/image-simulation-to-synthetic-waveform-api/simulate/ws/simulate-image-to-waveform-latent",
        "protocol": "WebSocket",
        "description": "Real-time simulation of image → synthetic waveform → waveform latent.",
        "input_format": {
            "type": "simulate",
            "session_id": "string (optional)",
            "image_base64": "data:image/png;base64,...",
        },
        "output_format": {
            "type": "waveform_latent",
            "session_id": "copied from input",
            "payload": "[float list representing latent]",
        },
    }
