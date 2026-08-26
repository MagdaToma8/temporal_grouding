import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
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
VICLIP_CONTEXT_LENGTH = 32


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
        self.context_length = VICLIP_CONTEXT_LENGTH

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

    def _encode_vision_full(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Mirrors ViCLIP's own VisionTransformer.forward (vendored viclip_vision.py, at
        OpenGVLab/ViCLIP-B-16-hf's HF dynamic-module cache) up through ln_post, computed
        once so both the pooled (global) and full-sequence (local) outputs can be derived
        from the same activations -- avoids running the transformer twice per call the way
        calling encode_vision() separately for the pooled output would. This is a direct
        translation of that forward pass, not a call into it, since it always takes the
        `self.proj is None` branch (full sequence, unprojected) regardless of whether this
        model actually has a projection -- the real proj is applied separately in
        get_embeddings for the pooled output.

        Returns: [1 + num_patches_per_frame * num_frames, batch, vision_width] (NBD,
        ViCLIP's own pre-permute layout) -- class token at index 0, then every patch
        across every frame.
        """
        vt = self.model.vision_encoder
        # [B,T,C,H,W] -> [B,C,T,H,W], mirrors encode_vision's own permute before conv1
        x = pixel_values.permute(0, 2, 1, 3, 4).contiguous()
        x = vt.conv1(x)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        x = torch.cat([
            vt.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)
        x = x + vt.positional_embedding.to(x.dtype)

        cls_tokens = x[:B, :1, :]
        x = x[:, 1:]
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        if hasattr(vt, 'temporal_positional_embedding'):
            if x.size(1) == 1:
                x = x + vt.temporal_positional_embedding.mean(1)
            else:
                x = x + vt.temporal_positional_embedding
        x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)

        x = torch.cat((cls_tokens, x), dim=1)
        x = vt.ln_pre(x)

        x = x.permute(1, 0, 2)  # BND -> NBD
        x = vt.transformer(x)
        return vt.ln_post(x)  # NBD

    def _encode_text_full(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Mirrors ViCLIP's own CLIP_TEXT.forward (vendored viclip_text.py) up through
        ln_final, computed once so both the pooled (global, EOT-position-only) and
        full-sequence (local) outputs can be derived from the same activations --
        avoids running the transformer twice per call.

        Returns: [batch, context_length, text_width] (BLD), unprojected.
        """
        te = self.model.text_encoder
        x = te.token_embedding(input_ids)
        x = x + te.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = te.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        return te.ln_final(x)

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

        # Local mode (AFS): per-patch/per-token detail, mirroring how CLIP's HF wrapper
        # exposes last_hidden_state -- vision keeps the class token (index 0) plus every
        # patch across every frame, unprojected; text keeps every token position,
        # unprojected. See _encode_vision_full/_encode_text_full docstrings for why these
        # reimplement ViCLIP's own forward passes rather than calling encode_vision/
        # text_encoder directly: those only ever return the pooled output.
        vt = self.model.vision_encoder
        vision_seq = self._encode_vision_full(pixel_values)  # NBD
        vision_local = vision_seq.permute(1, 0, 2)  # NBD -> BND
        video_embeds = vt.dropout(vision_seq[0]) @ vt.proj
        video_embeds = F.normalize(video_embeds, p=2, dim=-1)

        te = self.model.text_encoder
        text_local = self._encode_text_full(inputs['input_ids'])  # BLD
        text_embeds = text_local[
            torch.arange(text_local.shape[0]), inputs['input_ids'].argmax(dim=-1)
        ] @ te.text_projection
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        return {
            'image_embeds': video_embeds,
            'text_embeds': text_embeds,
            'vision_model_output': vision_local,
            'text_model_output': text_local,
        }

    def generate(self, *args, **kwargs) -> Any:
        raise NotImplementedError("ViCLIPWrapper does not support text generation")

    def decode(self, *args, **kwargs) -> Any:
        raise NotImplementedError("ViCLIPWrapper does not support decoding")
