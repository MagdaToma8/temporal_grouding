import pytest
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from src.datasets.coco import COCODataset, COCODatasetSummarizer
from src.datasets.data_collator import SummarizerDatasetCollator


@pytest.fixture
def data_dir():
    return "data/coco"


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
def dataset_generated_captions(data_dir, split, topk):
    dataset = COCODatasetSummarizer(
        data_dir=data_dir,
        split=split,
        embeddings_path="embeddings/coco/clip-vit-base-patch32/test",
        use_embeddings=False,
        use_generated_captions=True,
        topk=topk
    )
    return dataset


def test_val_test_size(data_dir, split):
    dataset = COCODataset(
        data_dir=data_dir,
        split=split,
    )

    assert len(dataset) == 5000

    dataset = COCODataset(
        data_dir=data_dir,
        split="val",
    )

    assert len(dataset) == 5000


def test_5_captions_per_image(data_dir, split):
    dataset = COCODataset(
        data_dir=data_dir,
        split=split,
    )

    for i in range(len(dataset)):
        assert len([key for key in dataset[i].keys() if "caption" in key]) == 5


def test_coco_dataset_for_summarizer__generated_captions(dataset_generated_captions):
    item = dataset_generated_captions.__getitem__(28)
    assert item["generated_text"] is not None


def test_coco_dataset_for_summarizer__collator_generated_captions(dataset_generated_captions, processor, topk):
    dataset_collator = SummarizerDatasetCollator(
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
