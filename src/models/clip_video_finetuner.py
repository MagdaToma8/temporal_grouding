import math
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from src.models.vlm_wrapper import VLMWrapper


def load_finetuned_clip_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Extract the underlying CLIPModel's weights from a CLIPVideoFineTuner Lightning
    checkpoint, for loading back into a fresh CLIPModel at inference/evaluation time.

    A CLIPVideoFineTuner checkpoint's state_dict keys are prefixed with
    "vlm_wrapper_model." (the attribute name the fine-tuner registers the CLIP model
    under -- see CLIPVideoFineTuner.__init__) plus a "criterion.logit_scale" entry for
    the contrastive loss's learnable temperature, which isn't part of the CLIP model
    itself and is dropped here.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    prefix = "vlm_wrapper_model."
    state_dict = {
        key[len(prefix):]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith(prefix)
    }
    if not state_dict:
        raise ValueError(
            f"No '{prefix}*' keys found in {checkpoint_path} -- "
            "is this actually a CLIPVideoFineTuner checkpoint?"
        )
    return state_dict


class ContrastiveLoss(nn.Module):
    """
    CLIP's own training objective (symmetric InfoNCE): within a batch, every
    (video, caption) pair on the diagonal is the correct match, and every
    off-diagonal pair is a negative. Unlike CosineSimilarityLoss (attentive_summarizer.py),
    which only pulls positive pairs together, this explicitly pushes negatives apart too --
    needed when fine-tuning the backbone itself, since a "pull together only" loss risks
    collapsing all embeddings to look alike.
    """
    def __init__(self, temperature: float = 0.07, learnable_temperature: bool = True):
        super().__init__()
        if learnable_temperature:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / temperature)))
        else:
            self.logit_scale = None
            self.fixed_scale = 1.0 / temperature

    def forward(self, video_embeds: torch.Tensor, text_embeds: torch.Tensor) -> Dict[str, Any]:
        # video_embeds/text_embeds are assumed already L2-normalized (CLIPVideoWrapper does this).
        scale = self.logit_scale.exp() if self.logit_scale is not None else self.fixed_scale

        logits_per_video = scale * video_embeds @ text_embeds.t()
        logits_per_text = logits_per_video.t()

        batch_size = video_embeds.shape[0]
        targets = torch.arange(batch_size, device=video_embeds.device)

        loss_video_to_text = F.cross_entropy(logits_per_video, targets)
        loss_text_to_video = F.cross_entropy(logits_per_text, targets)
        loss = (loss_video_to_text + loss_text_to_video) / 2

        with torch.no_grad():
            batch_hits_1 = (logits_per_video.argmax(dim=1) == targets).float().mean()

        return {"loss": loss, "batch_hits@1": batch_hits_1}


class CLIPVideoFineTuner(LightningModule):
    """
    Contrastive fine-tuning of CLIPVideoWrapper's underlying CLIP model on (video, caption)
    pairs. Two modes, controlled by `freeze_backbone`:
      - False (full fine-tuning): every CLIP parameter (vision tower, text tower, both
        projections) is trainable. Adapts the model most thoroughly, but is the most
        expensive and the most prone to overfitting/forgetting on a comparatively small
        dataset like MSR-VTT's ~8.5k training videos.
      - True (partial fine-tuning): the vision and text towers are frozen exactly as
        pretrained; only the final visual/text projection layers are trainable. Much
        cheaper and lower-risk, but a more limited adaptation.
    """
    def __init__(
            self,
            vlm_wrapper: VLMWrapper,
            learning_rate: float = 1e-6,
            weight_decay: float = 0.01,
            max_epochs: int = 10,
            freeze_backbone: bool = False,
            temperature: float = 0.07,
    ):
        super().__init__()
        self.vlm_wrapper = vlm_wrapper
        # Assigning the underlying nn.Module directly (not just vlm_wrapper, which is a plain
        # dataclass) is what makes PyTorch register it as a proper submodule -- required for
        # Trainer's .to(device) and parameter discovery to actually reach it. Same pattern
        # already used by AttentiveSummarizer (attentive_summarizer.py) for the same reason.
        self.vlm_wrapper_model = vlm_wrapper.model

        self.criterion = ContrastiveLoss(temperature=temperature)
        self.save_hyperparameters(ignore=["vlm_wrapper"])

        # Some backbones freeze parts of themselves at construction time regardless of what
        # we want here -- e.g. ViCLIP's own __init__ freezes its text encoder by default
        # (freeze_text=True), unlike a freshly-loaded CLIPModel where everything already
        # requires grad. Force everything trainable first so "full fine-tuning" (the
        # freeze_backbone=False default) actually means all parameters, for every backbone.
        for param in self.vlm_wrapper_model.parameters():
            param.requires_grad = True

        if freeze_backbone:
            for param in self.vlm_wrapper_model.parameters():
                param.requires_grad = False
            if hasattr(self.vlm_wrapper_model, "visual_projection"):
                # CLIP-style: projections are separate nn.Linear submodules.
                for param in self.vlm_wrapper_model.visual_projection.parameters():
                    param.requires_grad = True
                for param in self.vlm_wrapper_model.text_projection.parameters():
                    param.requires_grad = True
            elif hasattr(self.vlm_wrapper_model, "vision_encoder"):
                # ViCLIP-style: projections are raw nn.Parameter tensors on the vision/text
                # encoders, not separate submodules with their own .parameters().
                self.vlm_wrapper_model.vision_encoder.proj.requires_grad = True
                self.vlm_wrapper_model.text_encoder.text_projection.requires_grad = True
            else:
                raise NotImplementedError(
                    f"freeze_backbone partial fine-tuning isn't implemented for "
                    f"{type(self.vlm_wrapper_model).__name__} -- add a branch here."
                )

    def forward(self, batch: Dict[str, Any]):
        outputs = self.vlm_wrapper.get_embeddings(inputs={
            "pixel_values": batch["image"],
            "input_ids": batch["caption_0"],
            "attention_mask": batch["caption_0_attention_mask"],
        })
        return outputs["image_embeds"], outputs["text_embeds"]

    def _shared_step(self, batch, split: str):
        video_embeds, text_embeds = self(batch)
        loss_dict = self.criterion(video_embeds, text_embeds)
        self.log(f"{split}/loss", loss_dict["loss"], prog_bar=True, batch_size=video_embeds.shape[0])
        self.log(f"{split}/batch_hits@1", loss_dict["batch_hits@1"], prog_bar=True, batch_size=video_embeds.shape[0])
        return loss_dict["loss"]

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        trainable_params = [p for p in self.vlm_wrapper_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
