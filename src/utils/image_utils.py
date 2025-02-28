from typing import Any
from PIL import Image
from urllib.request import urlopen


def load_image(image_path: str) -> Any:
    image = Image.open(image_path).convert("RGB")
    return image


def load_image_url(image_url: str) -> Any:
    image = Image.open(urlopen(image_url)).convert("RGB")
    return image
