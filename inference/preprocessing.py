from io import BytesIO

from PIL import Image
import torch

from configs.resnet_config import config as resnet_config
from src.data.transforms import get_test_transforms


def load_image(image_bytes: bytes) -> Image.Image:
    """
    Load an uploaded image from raw bytes and convert it to RGB.
    """
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Convert uploaded image bytes into a batched tensor ready for inference.
    Uses the same preprocessing as validation for the pretrained ResNet18.
    """
    image = load_image(image_bytes)
    transform = get_test_transforms(
        image_size=resnet_config.image_size,
        pretrained=True,
    )
    tensor = transform(image)
    return tensor.unsqueeze(0)