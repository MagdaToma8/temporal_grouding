import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoModel

from src.models.vlm_wrapper import VLMWrapper
from src.models.viclip_assets.simple_tokenizer import SimpleTokenizer

VICLIP_MODEL_ID = "OpenGVLab/ViCLIP-B-16-hf"
_VENDORED_BPE_VOCAB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "viclip_assets", "bpe_simple_vocab_16e6.txt.gz"
)


class ViCLIPModelLoader:
    """
    Thin shim so `model_class.from_pretrained(model_id, trust_remote_code=True)` keeps
    working unmodified from retrieval_pipeline.py / run_embeddings_and_retrieval.py /
    train_backbone.py. OpenGVLab/ViCLIP-B-16-hf's config.json ships a `tokenizer_path`
    of "./bpe_simple_vocab_16e6.txt.gz" -- a bare relative path that only resolves if
    the process's cwd happens to be the HF cache's dynamic-module directory, which it
    never is here. We instead point it at our own vendored copy of the identical file
    before constructing the model.
    """

    @staticmethod
    def from_pretrained(model_id: str, **kwargs):
        # trust_remote_code is required for this model family (its modeling code lives in
        # the HF repo, not in `transformers` itself) -- forced True here rather than left to
        # the caller, since not every caller passes it (e.g. train_backbone.py doesn't).
        kwargs.pop("trust_remote_code", None)
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.tokenizer_path = _VENDORED_BPE_VOCAB_PATH
        return AutoModel.from_pretrained(model_id, config=config, trust_remote_code=True, **kwargs)

# ViCLIP was initialized from CLIP but preprocesses frames with plain ImageNet
# normalization (not CLIP's own mean/std), per its reference demo.ipynb.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
_INPUT_RESOLUTION = 224
# ViCLIP's own ViCLIP.__init__ hardcodes max_txt_l=32 and re-tokenizes at that context
# length (not CLIP's usual 77) when building the text encoder from the checkpoint.
_CONTEXT_LENGTH = 32


class ViCLIPProcessor:
    """
    ViCLIP has no HuggingFace AutoProcessor -- its own reference code (frames2tensor
    for video, CLIP_TEXT.tokenize for text, both in the OpenGVLab/ViCLIP-B-16-hf repo)
    does its own preprocessing outside the model class. This reimplements both, as a
    processor exposing the same `__call__(images=, text=, return_tensors=, ...)`
    interface CaptioningDataCollator already expects, so no pipeline code needs to
    change to support this model family.
    """

    def __init__(self):
        self.tokenizer = SimpleTokenizer()
        self.context_length = _CONTEXT_LENGTH

    @classmethod
    def from_pretrained(cls, model_id: str):
        return cls()

    def _preprocess_frame(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB").resize((_INPUT_RESOLUTION, _INPUT_RESOLUTION))
        frame = np.asarray(image, dtype=np.float32) / 255.0
        frame = (frame - _IMAGENET_MEAN) / _IMAGENET_STD
        return frame.transpose(2, 0, 1)  # HWC -> CHW

    def _tokenize(self, texts: List[str]) -> torch.Tensor:
        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        all_tokens = [
            [sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts
        ]
        result = torch.zeros(len(all_tokens), self.context_length, dtype=torch.long)
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > self.context_length:
                tokens = tokens[: self.context_length]
                tokens[-1] = eot_token
            result[i, : len(tokens)] = torch.tensor(tokens)
        return result

    def __call__(
        self,
        images: Optional[List[Image.Image]] = None,
        text: Optional[List[str]] = None,
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, Any]:
        outputs = {}
        if images is not None:
            frames = np.stack([self._preprocess_frame(img) for img in images], axis=0)
            outputs["pixel_values"] = torch.from_numpy(frames).float()
        if text is not None:
            outputs["input_ids"] = self._tokenize(text)
        assert outputs, "ViCLIPProcessor requires images and/or text"
        return outputs


@dataclass
class ViCLIPWrapper(VLMWrapper):
    """
    ViCLIP (OpenGVLab/InternVid): a tubelet/spatiotemporal-attention video backbone,
    unlike CLIPVideoWrapper's per-frame-then-mean-pool approach. Its vision tower is
    CLIP's own ViT with plain spatial attention replaced by joint spatiotemporal
    attention over all frame patches at once (see ViCLIPWrapper's get_embeddings),
    so -- unlike mean-pooling -- a frame's patches can attend to other frames' patches,
    letting the model represent motion instead of just averaging independent per-frame
    encodings.

    Like CLIPVideoWrapper, text is encoded fully independently of any video (ViCLIP's
    text tower is a plain CLIP text transformer, not cross-attended against video
    features), preserving the "encode everything once, compare via one similarity
    matrix" retrieval design.
    """
    model: Any = field(
        default_factory=lambda: ViCLIPModelLoader.from_pretrained(
            VICLIP_MODEL_ID, trust_remote_code=True
        )
    )
    processor: Any = field(default_factory=lambda: ViCLIPProcessor.from_pretrained(VICLIP_MODEL_ID))

    def process_inputs(self, **kwargs) -> Dict[str, Any]:
        required_keys = {'image', 'prompt'}
        if not required_keys.issubset(kwargs.keys()):
            raise ValueError(f"Missing required arguments: {required_keys - set(kwargs.keys())}")
        return self.processor(
            images=kwargs['image'],
            text=kwargs['prompt'],
        )

    def get_embeddings(self, inputs: Dict[str, Any], **kwargs) -> Any:
        """
        Args:
            inputs: dict with
                pixel_values: [batch, num_frames, C, H, W]
                input_ids: [batch, 32] token ids from ViCLIPProcessor (ViCLIP's own
                    context length; padding/attention_mask are unused -- its text
                    encoder pools from the EOT token position directly)
        """
        pixel_values = inputs['pixel_values']
        assert pixel_values.ndim == 5, (
            "ViCLIPWrapper expects pixel_values shaped [batch, num_frames, C, H, W], "
            f"got shape {tuple(pixel_values.shape)}"
        )
        # ViCLIP's own encode_vision(image, test=True) accepts [B,T,C,H,W] directly
        # (it permutes to [B,C,T,H,W] internally before the Conv3d patch embedding)
        # and, with test=True, skips the random token masking used during training.
        video_embeds = self.model.encode_vision(pixel_values, test=True)
        video_embeds = F.normalize(video_embeds, p=2, dim=-1)

        text_embeds = self.model.text_encoder(inputs['input_ids'])
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        return {
            'image_embeds': video_embeds,
            'text_embeds': text_embeds,
            # ViCLIP's vision/text towers only expose pooled outputs in this call path
            # (unlike CLIP's HF wrapper, which also returns last_hidden_state), and
            # retrieval_pipeline.py unconditionally calls .detach().cpu() on these two
            # fields even though nothing consumes them for the video backbone yet (they
            # exist for a future local-mode AFS summarizer). Placeholder pooled tensors
            # here, not real per-patch/per-token features -- revisit if ViCLIP is ever
            # used with local-mode AFS.
            'vision_model_output': video_embeds,
            'text_model_output': text_embeds,
        }

    def generate(self, *args, **kwargs) -> Any:
        raise NotImplementedError("ViCLIPWrapper does not support text generation")

    def decode(self, *args, **kwargs) -> Any:
        raise NotImplementedError("ViCLIPWrapper does not support decoding")
