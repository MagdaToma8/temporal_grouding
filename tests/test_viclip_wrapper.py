import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.datasets.msrvtt import load_msrvtt_data, MSRVTTDataset
from src.models.viclip import ViCLIPModelLoader, ViCLIPProcessor, ViCLIPWrapper, VICLIP_MODEL_ID


# Reference preprocessing constants, copied from OpenGVLab/ViCLIP-B-16-hf's own demo.ipynb
# (frames2tensor), to prove ViCLIPProcessor's reimplementation matches it exactly.
_REF_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
_REF_STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)


def _reference_frames_to_tensor(frames):
    """Mirrors demo.ipynb's frames2tensor, given already-RGB PIL frames (no BGR
    reversal needed, unlike the demo's cv2-sourced frames)."""
    arrays = []
    for frame in frames:
        arr = np.asarray(frame.convert("RGB").resize((224, 224)), dtype=np.float64)
        arr = (arr / 255.0 - _REF_MEAN) / _REF_STD
        arrays.append(arr)
    tube = np.stack(arrays, axis=0)  # [T, H, W, C]
    tube = np.transpose(tube, (0, 3, 1, 2))  # [T, C, H, W]
    return torch.from_numpy(tube).unsqueeze(0).float()  # [1, T, C, H, W]


@pytest.fixture(scope="module")
def viclip_model():
    return ViCLIPModelLoader.from_pretrained(VICLIP_MODEL_ID, trust_remote_code=True)


@pytest.fixture(scope="module")
def viclip_processor():
    return ViCLIPProcessor.from_pretrained(VICLIP_MODEL_ID)


@pytest.fixture(scope="module")
def wrapper(viclip_model, viclip_processor):
    return ViCLIPWrapper(model=viclip_model, processor=viclip_processor)


@pytest.fixture
def data_dir():
    return "data/msrvtt"


def test_video_embeds_match_reference_preprocessing(data_dir, viclip_model, wrapper):
    """
    Core correctness proof: our ViCLIPProcessor's frame normalization must produce
    bit-close results to the model's own reference preprocessing (demo.ipynb's
    frames2tensor) -- i.e. no accidental mismatch in resize/normalize/axis order.
    """
    dataset = MSRVTTDataset(data_dir=data_dir, split="test", num_frames=8)
    frames = dataset[0]["image"]
    assert len(frames) == 8

    with torch.no_grad():
        reference = viclip_model.get_vid_features(_reference_frames_to_tensor(frames))

        processed = ViCLIPProcessor()(images=frames)
        pixel_values = processed["pixel_values"].unsqueeze(0)  # [1, 8, C, H, W]
        outputs = wrapper.get_embeddings(inputs={
            "pixel_values": pixel_values,
            "input_ids": ViCLIPProcessor()(text=["a placeholder caption"])["input_ids"],
        })

    assert torch.allclose(outputs["image_embeds"], reference, atol=1e-4)


def test_text_embeds_match_reference_tokenization(viclip_model, wrapper):
    caption = "a woman is cooking food in a kitchen"

    with torch.no_grad():
        reference = torch.nn.functional.normalize(
            viclip_model.encode_text([caption]), p=2, dim=-1
        )

        processed = ViCLIPProcessor()(text=[caption])
        dummy_video = torch.randn(1, 8, 3, 224, 224)
        outputs = wrapper.get_embeddings(inputs={
            "pixel_values": dummy_video,
            "input_ids": processed["input_ids"],
        })

    assert torch.allclose(outputs["text_embeds"], reference, atol=1e-5)


def test_video_embeds_shape_and_normalization(data_dir, viclip_processor, wrapper):
    dataset, collator = load_msrvtt_data(
        config={"data_dir": data_dir, "num_frames": 8},
        split="test",
        processor=viclip_processor,
        process_images=True,
    )
    loader = DataLoader(dataset, batch_size=4, collate_fn=collator)
    batch = next(iter(loader))

    assert batch["image"].shape == (4, 8, 3, 224, 224)

    with torch.no_grad():
        outputs = wrapper.get_embeddings(inputs={
            "pixel_values": batch["image"],
            "input_ids": batch["caption_0"],
        })

    assert outputs["image_embeds"].shape == (4, 512)
    assert outputs["text_embeds"].shape == (4, 512)
    assert torch.allclose(outputs["image_embeds"].norm(dim=-1), torch.ones(4), atol=1e-4)
    assert torch.allclose(outputs["text_embeds"].norm(dim=-1), torch.ones(4), atol=1e-4)
