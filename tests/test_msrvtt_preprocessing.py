import numpy as np
import pytest
import torch
from PIL import Image

from src.datasets.msrvtt import MSRVTTDataset


@pytest.fixture
def data_dir():
    return "data/msrvtt"


def _to_tensor(frame: Image.Image) -> torch.Tensor:
    # Minimal, torchvision-free transform: this repo preprocesses images via the HF
    # processor inside the collator rather than torchvision, so a real pipeline would
    # normally never exercise this constructor argument -- this just verifies the
    # dataset stacks per-frame transform outputs correctly, whatever the transform is.
    array = np.array(frame)  # [H, W, C]
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0


def test_val_test_size(data_dir):
    assert len(MSRVTTDataset(data_dir=data_dir, split="train")) == 8500
    assert len(MSRVTTDataset(data_dir=data_dir, split="val")) == 500
    assert len(MSRVTTDataset(data_dir=data_dir, split="test")) == 1000


def test_captions_per_video(data_dir):
    train_ds = MSRVTTDataset(data_dir=data_dir, split="train")
    item = train_ds[0]
    assert len([k for k in item if k.startswith("caption_")]) == 20

    test_ds = MSRVTTDataset(data_dir=data_dir, split="test")
    item = test_ds[0]
    assert len([k for k in item if k.startswith("caption_")]) == 1


def test_frames_without_transform_are_pil_images(data_dir):
    dataset = MSRVTTDataset(data_dir=data_dir, split="test", num_frames=12)
    item = dataset[0]
    assert len(item["image"]) == 12
    assert all(isinstance(frame, Image.Image) for frame in item["image"])


def test_frames_with_transform_are_stacked_tensor(data_dir):
    dataset = MSRVTTDataset(data_dir=data_dir, split="test", num_frames=12, transform=_to_tensor)
    item = dataset[0]
    assert isinstance(item["image"], torch.Tensor)
    assert item["image"].shape[0] == 12
    assert item["image"].shape[1] == 3


def test_eval_mode_frame_sampling_is_deterministic(data_dir):
    dataset = MSRVTTDataset(data_dir=data_dir, split="test", num_frames=12)
    video_path = dataset.data[0]["video_path"]
    frames_a = dataset._load_video_frames(video_path)
    frames_b = dataset._load_video_frames(video_path)
    assert all(
        np.array_equal(np.array(a), np.array(b))
        for a, b in zip(frames_a, frames_b)
    )


def test_train_mode_frame_sampling_stays_in_bounds(data_dir):
    dataset = MSRVTTDataset(data_dir=data_dir, split="train", num_frames=12)
    item = dataset[0]
    assert len(item["image"]) == 12
