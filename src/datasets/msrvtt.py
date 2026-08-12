import os
import random

import numpy as np
from PIL import Image
# torch must be imported before decord: on Windows, decord's bundled native DLLs
# get put on the DLL search path first and break torch's own DLL loading otherwise.
import torch
from torch.utils.data import Dataset
import decord

from src.utils.utils import load_json_lines_file, load_json_file
from src.datasets.data_collator import CaptioningDataCollator, SummarizerDatasetCollator


def load_msrvtt_data(config, split, processor, process_images=False, num_frames=None, siglip2=False, summarizer=False):
    assert split in ["train", "val", "test"]

    data_dir = config.get("data_dir", None)
    assert data_dir is not None, "data_dir is required"

    transform = config.get("transform", None)
    num_frames = num_frames or config.get("num_frames", 12)
    segment_overlap = config.get("segment_overlap", 0.0)

    process_images = config.get("process_images", False) or process_images

    if summarizer:
        embeddings_path = config.get("embeddings_path", None)
        assert embeddings_path is not None, "embeddings_path is required"
        embeddings_path = os.path.join(embeddings_path, split)

        topk = config.get("topk", 5)
        use_generated_captions = config.get("use_generated_captions", False)

        dataset = MSRVTTDatasetSummarizer(
            data_dir=data_dir,
            split=split,
            transform=transform,
            num_frames=num_frames,
            segment_overlap=segment_overlap,
            embeddings_path=embeddings_path,
            topk=topk,
            use_generated_captions=use_generated_captions,
        )

        # is_video=True tells the collator both "image" (the query video) and each of the
        # topk entries in "retrieval_results_images" are lists of num_frames PIL frames,
        # not single images -- see MSRVTTDatasetSummarizer.__getitem__.
        collator = SummarizerDatasetCollator(
            processor=processor,
            process_images=process_images,
            siglip2=siglip2,
            is_video=True,
        )
    else:
        num_captions_to_use = config.get("num_captions_to_use", 20)
        assert num_captions_to_use > 0, "num_captions_to_use must be greater than 0"

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


class MSRVTTDatasetSummarizer(MSRVTTDataset):
    """
    AFS training data for video: mirrors FlickrDatasetSummarizer (src/datasets/flickr.py),
    but both the query item and its top-k retrieved neighbors are videos (sampled frames),
    not single images. Requires embeddings/retrieval results already generated by
    run_embeddings_and_retrieval.py -- see README's "Toward AFS on video" section.
    """

    def __init__(
            self,
            data_dir: str,
            split: str = None,
            transform=None,
            num_frames: int = 12,
            segment_overlap: float = 0.0,
            embeddings_path: str = None,
            topk: int = 5,
            use_generated_captions: bool = False,
    ):
        # Summarizer training holds one caption out per __getitem__ call as the query and
        # uses the rest as ground truth, so it wants every caption the split has -- pass a
        # large num_captions_to_use so MSRVTTDataset's own clamping picks up however many
        # the split actually has (20 for MSR-VTT train/val) instead of sampling a subset.
        super().__init__(
            data_dir=data_dir,
            split=split,
            transform=transform,
            num_captions_to_use=10 ** 9,
            num_frames=num_frames,
            segment_overlap=segment_overlap,
        )
        assert embeddings_path is not None, "embeddings_path is required"
        self.embeddings_path = embeddings_path
        self.topk = topk
        # Retrieved neighbors are always from this same split's own candidate pool (that's
        # what run_embeddings_and_retrieval.py ranks against), so a basename lookup scoped
        # to self.data is sufficient to recover a retrieved video's full path.
        self._basename_to_path = {
            os.path.basename(item["video_path"]): item["video_path"] for item in self.data
        }
        self.retrieval_results = self._load_retrieval_results()

        self.use_generated_captions = use_generated_captions
        self.generated_captions = self._load_generated_captions() if use_generated_captions else None

    def _load_retrieval_results(self):
        embeddings_files = [
            f for f in os.listdir(self.embeddings_path)
            if f.startswith("caption_") and f.endswith(".pt")
        ]
        # {caption_idx: {video_basename: [topk retrieved video basenames]}}
        retrieval_results_dict = {}
        for file in embeddings_files:
            caption_idx = int(file.split("_")[1])
            embeddings = torch.load(os.path.join(self.embeddings_path, file), weights_only=False)
            retrieval_results_dict[caption_idx] = {
                video_basename: entry["retrieval_results"][:self.topk]
                for video_basename, entry in embeddings.items()
            }
        return retrieval_results_dict

    def _load_generated_captions(self):
        generated_captions_file = os.path.join(self.data_dir, "captions", f"captions_{self.split}.json")
        return load_json_file(generated_captions_file)

    def _get_generated_captions(self, retrieved_basenames):
        return [self.generated_captions[basename] for basename in retrieved_basenames]

    def __getitem__(self, idx):
        item = self.data[idx]
        frames = self._load_video_frames(item["video_path"])
        if self.transform is not None:
            frames = torch.stack([self.transform(frame) for frame in frames])

        captions = item["captions"]
        random_idx = random.randint(0, len(captions) - 1)
        query = captions[random_idx]
        ground_truth = [caption for i, caption in enumerate(captions) if i != random_idx]

        video_basename = os.path.basename(item["video_path"])
        retrieved_basenames = self.retrieval_results[random_idx][video_basename]

        retrieval_results_images = []
        retrieval_results_video_paths = []
        for basename in retrieved_basenames:
            retrieved_path = self._basename_to_path[basename]
            retrieved_frames = self._load_video_frames(retrieved_path)
            if self.transform is not None:
                retrieved_frames = torch.stack([self.transform(frame) for frame in retrieved_frames])
            retrieval_results_images.append(retrieved_frames)
            retrieval_results_video_paths.append(retrieved_path)

        generated_text = self._get_generated_captions(retrieved_basenames) if self.use_generated_captions else None

        return {
            "image": frames,
            "img_path": item["video_path"],
            "class_label": item["imgid"],
            "query": query,
            "ground_truth": ground_truth,
            # List[List[PIL.Image]]: topk retrieved videos, each num_frames PIL frames.
            "retrieval_results_images": retrieval_results_images,
            "retrieval_results_img_paths": retrieval_results_video_paths,
            "text_feedback": None,
            "generated_text": generated_text,
        }
