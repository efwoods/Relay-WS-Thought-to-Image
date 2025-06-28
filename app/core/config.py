from pydantic_settings import BaseSettings

import torch


class Settings(BaseSettings):
    # Redis (Docker-managed)
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_PASSWORD: str
    THOUGHT_TO_IMAGE_REDIS_KEY: str

    # Torch
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Models
    IMAGE_DECODER_PATH: str
    RESIZED_IMAGE_SIZE: int
    LATENT_DIM: int

    class Config:
        env_file = ".env"


settings = Settings()
