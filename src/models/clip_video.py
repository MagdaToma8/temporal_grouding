from dataclasses import dataclass, field
from typing import Any, Dict

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

from src.models.vlm_wrapper import VLMWrapper


@dataclass
class CLIPVideoWrapper(VLMWrapper):
    """
    Adapts a frozen image CLIP model for video-text retrieval (the "MeanP" approach
    from CLIP4Clip): each sampled frame is encoded independently through the exact
    same vision tower a single-image CLIPWrapper would use, then frame embeddings are
    mean-pooled into one video embedding.

    This is an intentional first step, not the final backbone: we plan to replace this
    with a tubelet-based (spatiotemporal patch) encoder such as VideoMAE, paired with a
    trained text alignment head. Building this simpler path first isolates whether any
    future issues are in the (still largely untested-on-video) pipeline plumbing versus
    the eventual backbone/alignment itself, and unblocks PRF/GRF experiments and
    first-stage retrieval-result generation (needed before AFS training can even start)
    without waiting on a newly-trained alignment.

    Deliberately NOT using a native pretrained video-text model (e.g. X-CLIP): those
    typically cross-attend text embeddings against a specific video's features (X-CLIP's
    "prompts_generator"), which makes text embeddings non-precomputable ahead of
    retrieval -- incompatible with this codebase's "encode everything once, compare via
    one similarity matrix" design. Keeping text and video encoding fully independent
    (as here) preserves that design.
    """
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
        """
        Args:
            inputs: dict with
                pixel_values: [batch, num_frames, C, H, W]
                input_ids, attention_mask: standard text inputs, one caption per example
                    (unaffected by video -- text is never conditioned on video features)
        """
        pixel_values = inputs['pixel_values']
        assert pixel_values.ndim == 5, (
            "CLIPVideoWrapper expects pixel_values shaped [batch, num_frames, C, H, W], "
            f"got shape {tuple(pixel_values.shape)}"
        )
        batch_size, num_frames, num_channels, height, width = pixel_values.shape

        # Encode all frames of all videos in one batched pass through the same vision
        # tower a single-image CLIPWrapper would use -- this is exactly
        # CLIPModel.forward's image branch, just with batch*num_frames images at once.
        flattened_pixel_values = pixel_values.reshape(batch_size * num_frames, num_channels, height, width)
        vision_outputs = self.model.vision_model(pixel_values=flattened_pixel_values)
        frame_embeds = self.model.visual_projection(vision_outputs.pooler_output)  # [batch*num_frames, dim]

        # Regroup by video and mean-pool over the frame dimension ("MeanP"), then
        # normalize once at the end -- matching how CLIPModel.forward normalizes
        # image_embeds only after projection, not per intermediate step.
        frame_embeds = frame_embeds.view(batch_size, num_frames, -1)
        video_embeds = frame_embeds.mean(dim=1)
        video_embeds = F.normalize(video_embeds, p=2, dim=-1)

        # Per-frame patch tokens, kept around (unpooled, unprojected) for a future local-mode
        # AFS summarizer -- not consumed anywhere yet. Shape: [batch, num_frames, num_patches+1, dim]
        local_tokens = vision_outputs.last_hidden_state.view(
            batch_size, num_frames, -1, vision_outputs.last_hidden_state.shape[-1]
        )

        text_outputs = self.model.text_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask', None)
        )
        text_embeds = self.model.text_projection(text_outputs.pooler_output)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        return {
            'image_embeds': video_embeds,
            'text_embeds': text_embeds,
            'vision_model_output': local_tokens,
            'text_model_output': text_outputs.last_hidden_state,
        }

    def generate(self, *args, **kwargs) -> Any:
        raise NotImplementedError("CLIPVideoWrapper does not support text generation")

    def decode(self, *args, **kwargs) -> Any:
        raise NotImplementedError("CLIPVideoWrapper does not support decoding")
