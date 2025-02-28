import pytest
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from src.datasets.flickr import FlickrDataset, FlickrDatasetSummarizer, FlickrDatasetSummarizerCollator


@pytest.fixture
def data_dir():
    return "data/flickr30k"


@pytest.fixture
def split():
    return "test"

@pytest.fixture
def processor():
    return AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")


@pytest.fixture
def topk():
    return 7


@pytest.fixture
def dataset_yolo(data_dir, split, topk):
    dataset = FlickrDatasetSummarizer(
        data_dir=data_dir,
        data_file=os.path.join(data_dir, "dataset_flickr30k.json"),
        split=split,
        embeddings_path="embeddings/flickr30k/clip-vit-base-patch32/test",
        use_embeddings=False,
        use_detected_objects=True,
        use_classified_objects=True,
        use_generated_captions=False,
        topk=topk
    )
    return dataset


@pytest.fixture
def dataset_generated_captions(data_dir, split, topk):
    dataset = FlickrDatasetSummarizer(
        data_dir=data_dir,
        data_file=os.path.join(data_dir, "dataset_flickr30k.json"),
        split=split,
        embeddings_path="embeddings/flickr30k/clip-vit-base-patch32/test",
        use_embeddings=False,
        use_detected_objects=False,
        use_classified_objects=False,
        use_generated_captions=True,
        topk=topk
    )
    return dataset


def test_val_test_size(data_dir, split):
    dataset = FlickrDataset(
        data_dir=data_dir,
        split=split,
        data_file=os.path.join(data_dir, "dataset_flickr30k.json")
    )

    assert len(dataset) == 1000

    dataset = FlickrDataset(
        data_dir=data_dir,
        split="val",
        data_file=os.path.join(data_dir, "dataset_flickr30k.json")
    )

    assert len(dataset) == 1014


def test_5_captions_per_image(data_dir, split):
    dataset = FlickrDataset(
        data_dir=data_dir,
        split=split,
        data_file=os.path.join(data_dir, "dataset_flickr30k.json")
    )

    for i in range(len(dataset)):
        assert len([key for key in dataset[i].keys() if "caption" in key]) == 5


def test_flickr_dataset_for_summarizer__yolo(dataset_yolo):
    item = dataset_yolo.__getitem__(28)
    assert item["generated_text"] is None
    assert item["text_feedback"] is not None


def test_flickr_dataset_for_summarizer__generated_captions(dataset_generated_captions):
    item = dataset_generated_captions.__getitem__(28)
    assert item["generated_text"] is not None
    assert item["text_feedback"] is None


def test_flickr_dataset_for_summarizer__collator_generated_captions(dataset_generated_captions, processor, topk):
    dataset_collator = FlickrDatasetSummarizerCollator(
        processor=processor,
        process_images=True
    )

    dataloader = DataLoader(
        dataset_generated_captions,
        batch_size=8,
        collate_fn=dataset_collator
    )

    batch = next(iter(dataloader))

    assert batch["image"].shape == (8, 3, 224, 224)
    assert batch["query_input_ids"].shape[0] == 8
    assert batch["ground_truth_input_ids"].shape[0] == 8 * (dataset_generated_captions.num_all_captions - 1)
    assert batch["retrieval_results_images"].shape == (8 * topk, 3, 224, 224)
    assert batch["generated_text_input_ids"].shape[0] == 8 * topk
