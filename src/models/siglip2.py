from dataclasses import dataclass, field
from typing import Any, Dict

import torch
from transformers import AutoProcessor, SiglipModel

from src.models.vlm_wrapper import VLMWrapper


@dataclass
class SigLip2Wrapper(VLMWrapper):
    model: Any = field(
        default_factory=lambda: SiglipModel.from_pretrained(
            "google/siglip2-base-patch16-224", device_map={"": 0}, torch_dtype=torch.float16
        )
    )
    processor: Any = field(default_factory=lambda: AutoProcessor.from_pretrained("google/siglip2-base-patch16-224"))

    def process_inputs(self, images=None, text=None) -> Dict[str, Any]:
        assert images is not None or text is not None
        # It's very important to lowercase the text with siglip2 model
        if text is not None:
            text = [t.lower() for t in text]
        return self.processor(
            images=images,
            text=text,
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True,
        ).to(self.model.device)

    def get_embeddings(self, inputs: Dict[str, Any], **kwargs) -> Any:
        outputs = self.model(**inputs)
        return {
            "image_embeds": outputs.image_embeds,
            "text_embeds": outputs.text_embeds,
            "logits_per_image": outputs.logits_per_image,
            "logits_per_text": outputs.logits_per_text,
            "vision_model_output": outputs.vision_model_output.last_hidden_state,
            "text_model_output": outputs.text_model_output.last_hidden_state,
        }

    def generate(self, *args, **kwargs) -> Any:
        raise NotImplementedError("CLIP does not support text generation")

    def decode(self, *args, **kwargs) -> Any:
        raise NotImplementedError("CLIP does not support decoding")
