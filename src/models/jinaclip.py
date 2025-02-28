from dataclasses import dataclass, field
from typing import Any, Dict

from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from src.models.vlm_wrapper import VLMWrapper


class JinaProcessor:
    def __init__(self, model_id: str):
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )

    @classmethod
    def from_pretrained(cls, model_id: str):
        return cls(model_id)

    def __call__(self, images=None, text=None, return_tensors="pt", padding=True, truncation=True):
        img_inputs = {}
        text_inputs = {}
        if images is not None:
            img_inputs = self.image_processor(
                images,
                return_tensors=return_tensors,
            )
        if text is not None:
            text_inputs = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=padding,
                truncation=truncation,
            )
        assert img_inputs or text_inputs
        return {
            **img_inputs,
            **text_inputs
        }


@dataclass
class JinaVLMWrapper(VLMWrapper):
    model: Any = field(
        default_factory=lambda: AutoModel.from_pretrained(
            'jinaai/jina-clip-v2', trust_remote_code=True
        )
    )

    image_processor: Any = field(
        default_factory=lambda: AutoImageProcessor.from_pretrained(
            'jinaai/jina-clip-v2', trust_remote_code=True
        )
    )

    tokenizer: Any = field(
        default_factory=lambda: AutoTokenizer.from_pretrained(
            'jinaai/jina-clip-v2', trust_remote_code=True
        )
    )

    def process_inputs(self, **kwargs) -> Dict[str, Any]:
        required_keys = {'image', 'prompt'}
        if not required_keys.issubset(kwargs.keys()):
            raise ValueError(f"Missing required arguments: {required_keys - set(kwargs.keys())}")

        image_inputs = self.image_processor(
            images=kwargs['image'],
            return_tensors="pt",
        )
        text_inputs = self.tokenizer(
            kwargs['prompt'],
            return_tensors="pt",
            task='retrieval.query',
        )
        return {
            'pixel_values': image_inputs['pixel_values'],
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask']
        }

    def get_embeddings(self, inputs):
        outputs = self.model(**inputs)

        return outputs

    def generate(self, *args, **kwargs) -> Any:
        raise NotImplementedError("CLIP does not support text generation")

    def decode(self, *args, **kwargs) -> Any:
        raise NotImplementedError("CLIP does not support decoding")
