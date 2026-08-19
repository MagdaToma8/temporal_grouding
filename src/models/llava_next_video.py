from dataclasses import dataclass, field
from typing import Any, Dict

import torch
from transformers import AutoProcessor, LlavaNextVideoForConditionalGeneration

from src.models.vlm_wrapper import VLMWrapper


@dataclass
class LlavaNextVideoWrapper(VLMWrapper):
    """
    Video-captioning counterpart to LLaVaWrapper, for GRF on video: captions generated from
    single frames (LLaVaWrapper applied to one middle frame) can't describe motion/events that
    only show up across frames, the same limitation the ViCLIP-B backbone work was about. This
    wrapper instead feeds the model num_frames real frames per video, sampled the same way
    (MSRVTTDataset) as everywhere else in this codebase.

    Like LLaVaWrapper, "image" is reused as the kwarg name for the visual input even though it
    holds a list of frames here, not a single image -- matching the existing convention (see
    MSRVTTDataset.__getitem__) of keeping the same field name so calling code doesn't need to
    special-case video.
    """
    model: Any = field(
        default_factory=lambda: LlavaNextVideoForConditionalGeneration.from_pretrained(
            "llava-hf/LLaVA-NeXT-Video-7B-hf",
            device_map={"": 0},
            torch_dtype=torch.float16
        )
    )
    processor: Any = field(
        default_factory=lambda: AutoProcessor.from_pretrained(
            "llava-hf/LLaVA-NeXT-Video-7B-hf"
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
                f"USER: <video>\n{prompt} ASSISTANT:" for prompt in kwargs['prompt']
            ]
        else:
            prompts = kwargs['prompt']

        return self.processor(
            videos=kwargs['image'],
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
        max_new_tokens = kwargs.get('max_new_tokens', 100)
        return self.model.generate(**inputs, max_new_tokens=max_new_tokens)

    def get_embeddings(self, *args, **kwargs) -> Any:
        raise NotImplementedError("LlavaNextVideoWrapper does not support embeddings")
