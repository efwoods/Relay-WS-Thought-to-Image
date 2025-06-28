# service/transform.py
import torch
import json
import base64
from io import BytesIO
from PIL import Image
from service.loader import load_models, get_image_resize_transform
from core.config import settings

image_decoder = load_models()
image_resize_transform = get_image_resize_transform()


# Load normalization stats
import json

with open(settings.NORMALIZATION_CONFIG, "r") as f:
    electrode_stats = json.load(f)

# Convert to tensors for batch use
means = (
    torch.tensor([electrode_stats[str(i)]["mean"] for i in range(len(electrode_stats))])
    .float()
    .to(settings.DEVICE)
)
stds = (
    torch.tensor([electrode_stats[str(i)]["std"] for i in range(len(electrode_stats))])
    .float()
    .to(settings.DEVICE)
)


def preprocess_image_from_websocket(message):
    request = json.loads(message)

    # Parse waveform_latent and convert to tensor
    waveform_latent = torch.tensor(
        request["payload"], dtype=torch.float32, device=settings.DEVICE
    ).unsqueeze(0)

    # Deserialize skip connections
    serialized_skip = request["skip_connections"]
    buffer = BytesIO(base64.b64decode(serialized_skip))
    skip_connections = torch.load(buffer, map_location=settings.DEVICE)

    return waveform_latent, skip_connections, request


@torch.no_grad()
def reconstruct_image_from_waveform_latents(waveform_latent, skip_connections):
    reconstructed_image = image_decoder(waveform_latent, skip_connections)
    return reconstructed_image
