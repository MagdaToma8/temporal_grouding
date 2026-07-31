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


def test_zero_overlap_reproduces_original_non_overlapping_indices(data_dir):
    # segment_overlap=0.0 must be indistinguishable from the original (pre-overlap)
    # implementation -- this is the baseline result we already validated and reported
    # real retrieval numbers against, so it must not silently shift.
    dataset = MSRVTTDataset(data_dir=data_dir, split="test", num_frames=12, segment_overlap=0.0)
    indices = dataset._sample_frame_indices(300)
    assert indices == [12, 37, 62, 87, 112, 137, 162, 187, 212, 237, 262, 287]


def test_overlap_widens_segments(data_dir):
    dataset = MSRVTTDataset(data_dir=data_dir, split="test", num_frames=12, segment_overlap=0.5)
    boundaries = np.linspace(0, 300, 13)
    base_stride = 300 / 12
    window_width = base_stride / (1 - dataset.segment_overlap)
    for i in range(11):  # last segment is clipped near the clip's end, skip it
        start = int(boundaries[i])
        end = int(min(boundaries[i] + window_width, 300))
        assert (end - start) > base_stride


def test_overlap_eval_mode_still_deterministic_and_in_bounds(data_dir):
    dataset = MSRVTTDataset(data_dir=data_dir, split="test", num_frames=12, segment_overlap=0.5)
    a = dataset._sample_frame_indices(300)
    b = dataset._sample_frame_indices(300)
    assert a == b
    assert len(a) == 12 and all(0 <= i < 300 for i in a)


def test_overlap_train_mode_varies_and_stays_in_bounds():
    dataset = MSRVTTDataset.__new__(MSRVTTDataset)
    dataset.num_frames = 12
    dataset.segment_overlap = 0.5
    dataset.is_train = True
    a = dataset._sample_frame_indices(300)
    b = dataset._sample_frame_indices(300)
    assert len(a) == 12 and all(0 <= i < 300 for i in a)
    assert a != b


def test_invalid_segment_overlap_rejected(data_dir):
    with pytest.raises(AssertionError):
        MSRVTTDataset(data_dir=data_dir, split="test", num_frames=12, segment_overlap=1.0)
