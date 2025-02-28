from dataclasses import dataclass, field
from typing import Any, Dict

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from src.models.vlm_wrapper import VLMWrapper

@dataclass
class LLaVaWrapper(VLMWrapper):
    model: Any = field(
        default_factory=lambda: LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",
            device_map={"": 0},
            torch_dtype=torch.float16
        )
    )
    processor: Any = field(
        default_factory=lambda: AutoProcessor.from_pretrained(
            "llava-hf/llava-1.5-7b-hf"
        )
    )

    def __post_init__(self):
        self.processor.tokenizer.padding_side = "left"

    def process_inputs(self, apply_template=True, **kwargs):
        required_keys = {'image', 'prompt'}
        if not required_keys.issubset(kwargs.keys()):
            raise ValueError(f"Missing required arguments: {required_keys - set(kwargs.keys())}")

        if apply_template:
            prompts = [
                f"USER: <image>\n{prompt} ASSISTANT:" for prompt in kwargs['prompt']
            ]
        else:
            prompts = kwargs['prompt']

        return self.processor(
            images=kwargs['image'],
            text=prompts,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

    def decode(self, outputs, **kwargs):
        skip_special_tokens = kwargs.get('skip_special_tokens', True)
        clean_up_tokenization_spaces = kwargs.get('clean_up_tokenization_spaces', False)
        return self.processor.batch_decode(
            outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )

    def generate(self, inputs: Dict[str, Any], **kwargs) -> Any:
        # max_len = kwargs.get('max_len', 1000)
        max_new_tokens = kwargs.get('max_new_tokens', 100)
        return self.model.generate(**inputs, max_new_tokens=max_new_tokens)

    def get_embeddings(self, *args, **kwargs) -> Any:
        raise NotImplementedError("`VideoLlavaWrapper` does not support embeddings")

