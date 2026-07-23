import pytest
import torch
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

from src.datasets.msrvtt import load_msrvtt_data, MSRVTTDataset
from src.models.clip_video import CLIPVideoWrapper


MODEL_ID = "openai/clip-vit-base-patch32"


@pytest.fixture(scope="module")
def clip_model():
    # No device_map/fp16 here (unlike the wrapper's own default_factory) so this test
    # runs on CPU-only machines too, matching how retrieval_pipeline.py actually loads
    # models (.from_pretrained(...) then .to(device) explicitly, not via the dataclass default).
    return CLIPModel.from_pretrained(MODEL_ID)


@pytest.fixture(scope="module")
def clip_processor():
    return CLIPProcessor.from_pretrained(MODEL_ID)


@pytest.fixture(scope="module")
def wrapper(clip_model, clip_processor):
    return CLIPVideoWrapper(model=clip_model, processor=clip_processor)


@pytest.fixture
def data_dir():
    return "data/msrvtt"


def test_video_embeds_match_per_frame_meanpool(data_dir, clip_model, clip_processor, wrapper):
    """
    The core correctness proof: encoding 12 frames of ONE video through
    CLIPVideoWrapper (batched, reshaped internally) must give bit-close results to
    manually running the same 12 frames through the exact same CLIP vision tower
    one at a time, averaging, and normalizing once -- i.e. the batching/reshaping
    logic doesn't silently mix up which frames belong to which video.
    """
    dataset = MSRVTTDataset(data_dir=data_dir, split="test", num_frames=12)
    frames = dataset[0]["image"]  # 12 PIL frames from one real video
    assert len(frames) == 12

    # Reference: process all 12 frames as independent images (not video-batched),
    # replicating exactly what CLIPModel.forward's image branch does per frame.
    inputs = clip_processor(images=frames, return_tensors="pt")
    with torch.no_grad():
        vision_outputs = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        frame_embeds = clip_model.visual_projection(vision_outputs.pooler_output)  # [12, dim]
    reference = torch.nn.functional.normalize(frame_embeds.mean(dim=0, keepdim=True), p=2, dim=-1)

    # CLIPVideoWrapper's path: same 12 frames, but reshaped into [1, 12, C, H, W]
    # and run through get_embeddings' batched reshape/pool logic.
    video_pixel_values = inputs["pixel_values"].unsqueeze(0)  # [1, 12, C, H, W]
    dummy_text = clip_processor(text=["a placeholder caption"], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = wrapper.get_embeddings(inputs={
            "pixel_values": video_pixel_values,
            "input_ids": dummy_text["input_ids"],
            "attention_mask": dummy_text["attention_mask"],
        })

    assert torch.allclose(outputs["image_embeds"], reference, atol=1e-5)


def test_video_embeds_shape_and_normalization(data_dir, clip_processor, wrapper):
    dataset, collator = load_msrvtt_data(
        config={"data_dir": data_dir, "num_frames": 12},
        split="test",
        processor=clip_processor,
        process_images=True,
    )
    loader = DataLoader(dataset, batch_size=4, collate_fn=collator)
    batch = next(iter(loader))

    assert batch["image"].shape == (4, 12, 3, 224, 224)

    with torch.no_grad():
        outputs = wrapper.get_embeddings(inputs={
            "pixel_values": batch["image"],
            "input_ids": batch["caption_0"],
            "attention_mask": batch["caption_0_attention_mask"],
        })

    assert outputs["image_embeds"].shape == (4, 512)
    assert outputs["text_embeds"].shape == (4, 512)
    assert torch.allclose(outputs["image_embeds"].norm(dim=-1), torch.ones(4), atol=1e-4)
    assert torch.allclose(outputs["text_embeds"].norm(dim=-1), torch.ones(4), atol=1e-4)


def test_local_tokens_shape(data_dir, clip_processor, wrapper):
    dataset, collator = load_msrvtt_data(
        config={"data_dir": data_dir, "num_frames": 12},
        split="test",
        processor=clip_processor,
        process_images=True,
    )
    loader = DataLoader(dataset, batch_size=2, collate_fn=collator)
    batch = next(iter(loader))

    with torch.no_grad():
        outputs = wrapper.get_embeddings(inputs={
            "pixel_values": batch["image"],
            "input_ids": batch["caption_0"],
            "attention_mask": batch["caption_0_attention_mask"],
        })

    # CLIP ViT-B/32 on 224x224 images: (224/32)^2 = 49 patches + 1 CLS token = 50
    assert outputs["vision_model_output"].shape == (2, 12, 50, 768)


def test_collator_video_batch_shape(data_dir, clip_processor):
    dataset, collator = load_msrvtt_data(
        config={"data_dir": data_dir, "num_frames": 12},
        split="test",
        processor=clip_processor,
        process_images=True,
    )
    loader = DataLoader(dataset, batch_size=3, collate_fn=collator)
    batch = next(iter(loader))

    assert batch["image"].shape == (3, 12, 3, 224, 224)
    assert batch["class_label"].shape == (3,)
