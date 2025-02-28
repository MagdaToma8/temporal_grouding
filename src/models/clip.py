from dataclasses import dataclass, field
from typing import Any, Dict

from transformers import (
    CLIPModel,
    CLIPProcessor
)
import torch

from src.models.vlm_wrapper import VLMWrapper


@dataclass
class CLIPWrapper(VLMWrapper):
    model: Any = field(
        default_factory=lambda: CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            device_map={"": 0},
            torch_dtype=torch.float16
        )
    )
    processor: Any = field(
        default_factory=lambda: CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
    )

    def process_inputs(self, **kwargs) -> Dict[str, Any]:
        required_keys = {'image', 'prompt'}
        if not required_keys.issubset(kwargs.keys()):
            raise ValueError(f"Missing required arguments: {required_keys - set(kwargs.keys())}")
        return self.processor(
            images=kwargs['image'],
            text=kwargs['prompt'],
            return_tensors="pt",
            padding=True
        ).to(self.model.device)

    def get_embeddings(self, inputs: Dict[str, Any], **kwargs) -> Any:
        outputs = self.model(**inputs)
        return {
            'image_embeds': outputs.image_embeds,
            'text_embeds': outputs.text_embeds,
            'logits_per_image': outputs.logits_per_image,
            'logits_per_text': outputs.logits_per_text,
            'vision_model_output': outputs.vision_model_output.last_hidden_state,
            'text_model_output': outputs.text_model_output.last_hidden_state
        }

    def generate(self, *args, **kwargs) -> Any:
        raise NotImplementedError("CLIP does not support text generation")

    def decode(self, *args, **kwargs) -> Any:
        raise NotImplementedError("CLIP does not support decoding")
