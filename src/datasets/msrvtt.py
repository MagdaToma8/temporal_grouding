import os
import random

import numpy as np
from PIL import Image
# torch must be imported before decord: on Windows, decord's bundled native DLLs
# get put on the DLL search path first and break torch's own DLL loading otherwise.
import torch
from torch.utils.data import Dataset
import decord

from src.utils.utils import load_json_lines_file
from src.datasets.data_collator import CaptioningDataCollator


def load_msrvtt_data(config, split, processor, process_images=False, num_frames=None, siglip2=False):
    assert split in ["train", "val", "test"]

    data_dir = config.get("data_dir", None)
    assert data_dir is not None, "data_dir is required"

    transform = config.get("transform", None)
    num_frames = num_frames or config.get("num_frames", 12)
    segment_overlap = config.get("segment_overlap", 0.0)

    num_captions_to_use = config.get("num_captions_to_use", 20)
    assert num_captions_to_use > 0, "num_captions_to_use must be greater than 0"

    process_images = config.get("process_images", False) or process_images

    dataset = MSRVTTDataset(
        data_dir=data_dir,
        split=split,
        transform=transform,
        num_captions_to_use=num_captions_to_use,
        num_frames=num_frames,
        segment_overlap=segment_overlap,
    )

    # is_video=True tells the collator that each example's "image" is a list of num_frames
    # PIL frames (not a single image): frames get flattened across the batch for one processor
    # call, then reshaped back to [batch, num_frames, C, H, W] -- the shape CLIPVideoWrapper
    # (and any future video wrapper) expects.
    collator = CaptioningDataCollator(
        processor=processor,
        process_images=process_images,
        num_captions=dataset.num_captions_to_use,
        siglip2=siglip2,
        is_video=True,
    )

    return dataset, collator


class MSRVTTDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            split: str = None,
            transform=None,
            num_captions_to_use: int = 20,
            num_frames: int = 12,
            segment_overlap: float = 0.0,
    ):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.num_frames = num_frames
        assert 0.0 <= segment_overlap < 1.0, "segment_overlap must be in [0, 1)"
        self.segment_overlap = segment_overlap
        # TSN-style sampling: random frame per segment at train time, center frame at eval time.
        # "val" behaves like "test" here (deterministic) since it's used for checkpoint selection,
        # not for augmentation.
        self.is_train = split == "train"

        assert num_captions_to_use > 0
        self.data = self._load_data()

        # Every item in a given split is expected to carry the same number of captions
        # (20 for MSR-VTT train/val, 1 for the JSFusion test split). Clamp defensively so that
        # CaptioningDataCollator's fixed caption_0..caption_{num_captions-1} keys always exist
        # across the whole split, instead of silently KeyError-ing on a batch item with fewer
        # captions than requested.
        min_captions_available = min(len(item["captions"]) for item in self.data)
        self.num_captions_to_use = min(num_captions_to_use, min_captions_available)

    def _load_data(self):
        data_file = os.path.join(self.data_dir, "annotations", f"{self.split}.json")
        data = load_json_lines_file(data_file)
        dataset = []
        for item in data:
            video_path = os.path.join(
                self.data_dir,
                item["filepath"],
                item["filename"]
            )
            dataset.append({
                "video_path": video_path,
                "captions": item["sentences"],
                "sentids": item["sentids"],
                "imgid": item["imgid"],
            })
        return dataset

    def __len__(self):
        return len(self.data)

    def _sample_frame_indices(self, num_available_frames: int):
        """TSN-style uniform segment sampling, with optional overlap between segments.

        Segments start at the same evenly-spaced points regardless of overlap (so
        segment_overlap=0.0 reproduces the original non-overlapping behavior exactly).
        `segment_overlap` widens each segment beyond its non-overlapping width, so it
        reaches into the next segment's range: at 0.0, segment width == the spacing
        between segment starts (no overlap); at 0.5, each segment is twice as wide as
        the non-overlapping case, so it shares half its range with its neighbor.

        Pick a random frame from within each (possibly widened) segment at train time,
        or the center frame at eval time. If a segment has no frames (a clip shorter
        than num_frames), repeat the last available index.
        """
        boundaries = np.linspace(0, num_available_frames, self.num_frames + 1)
        base_stride = num_available_frames / self.num_frames
        window_width = base_stride / (1 - self.segment_overlap)  # == base_stride when segment_overlap == 0

        indices = []
        for i in range(self.num_frames):
            start = int(boundaries[i])
            end = int(min(boundaries[i] + window_width, num_available_frames))
            if end <= start:
                idx = start
            elif self.is_train:
                idx = random.randint(start, end - 1)
            else:
                idx = (start + end - 1) // 2
            indices.append(min(max(idx, 0), num_available_frames - 1))
        return indices

    def _load_video_frames(self, video_path: str):
        video_reader = decord.VideoReader(video_path, num_threads=1)    #open the video 
        num_available_frames = len(video_reader)        #num of frames in the video
        indices = self._sample_frame_indices(num_available_frames)  #get the 12 index numbers
        frames = video_reader.get_batch(indices).asnumpy()  # [num_frames, H, W, C], uint8
        return [Image.fromarray(frame) for frame in frames]

    def __getitem__(self, idx):
        item = self.data[idx]
        frames = self._load_video_frames(item["video_path"])
        if self.transform is not None:
            frames = torch.stack([self.transform(frame) for frame in frames])

        if self.num_captions_to_use < len(item["captions"]):
            captions = random.sample(item["captions"], self.num_captions_to_use)
        else:
            captions = item["captions"]
        captions_dict = {}
        for i, caption in enumerate(captions):
            captions_dict[f"caption_{i}"] = caption

        return {
            # field name kept as "image"/"img_path" (not renamed to "video"/"video_path") so the
            # existing collators and pipeline scripts (retrieval_pipeline.py etc.) keep working
            # unchanged -- the value is a list/stack of num_frames frames instead of a single image.
            "image": frames,
            "img_path": item["video_path"],
            "class_label": item["imgid"],  # we want to retrieve the correct video id based on caption
            **captions_dict,
        }
