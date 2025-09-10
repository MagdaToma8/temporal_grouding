import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset

from src.utils.utils import load_json_file, load_json_lines_file
from src.datasets.data_collator import CaptioningDataCollator, SummarizerDatasetCollator


COCO_NUM_ALL_CAPTIONS = 5


def load_coco_data(config, split, processor, process_images=True, summarizer=False, siglip2=False):
    assert split in ["train", "val", "test"]

    data_dir = config.get("data_dir", None)
    assert data_dir is not None, "data_dir is required"

    transform = config.get("transform", None)

    num_captions_to_use = config.get("num_captions_to_use", 5)
    assert num_captions_to_use > 0, "num_captions_to_use must be greater than 0"

    process_images = config.get("process_images", False) or process_images

    if summarizer:
        embeddings_path = config.get("embeddings_path", None)
        assert embeddings_path is not None, "embeddings_path is required"
        embeddings_path = os.path.join(embeddings_path, split)

        topk = config.get("topk", 5)
        use_embeddings = config.get("use_embeddings", True)
        use_generated_captions = config.get("use_generated_captions", True)
        dataset = COCODatasetSummarizer(
            data_dir=data_dir,
            split=split,
            transform=transform,
            embeddings_path=embeddings_path,
            topk=topk,
            use_embeddings=use_embeddings,
            use_generated_captions=use_generated_captions,
        )

        collator = SummarizerDatasetCollator(
            processor=processor,
            process_images=process_images,
            siglip2=siglip2
        )

    else:
        dataset = COCODataset(
            data_dir=data_dir,
            split=split,
            transform=transform,
            num_captions_to_use=num_captions_to_use
        )

        collator = CaptioningDataCollator(
            processor=processor,
            process_images=process_images,
            num_captions=num_captions_to_use,
            siglip2=siglip2
        )

    return dataset, collator


class COCODataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            split: str = None,
            transform=None,
            num_captions_to_use: int = 5,
    ):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        assert num_captions_to_use > 0
        self.num_captions_to_use = num_captions_to_use
        self.data = self._load_data()

    def _load_data(self):
        data_file = os.path.join(self.data_dir, "annotations", f"{self.split}.json")
        data = load_json_lines_file(data_file)
        dataset = []
        for item in data:
            image_path = os.path.join(
                self.data_dir,
                item["filepath"],
                item["filename"]
            )
            captions = item["sentences"]
            dataset.append({
                "image_path": image_path,
                "captions": captions,
                "sentids": item["sentids"],
                "imgid": item["imgid"],
            })
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.num_captions_to_use < len(item["captions"]):
            captions = random.sample(item["captions"], self.num_captions_to_use)
        else:
            captions = item["captions"]
        captions_dict = {}
        for i, caption in enumerate(captions):
            captions_dict[f"caption_{i}"] = caption
        return {
            "image": image,
            "img_path": item["image_path"],
            "class_label": item["imgid"], # we want to retrieve the correct image id based on caption
            **captions_dict,
        }


class COCODatasetSummarizer(Dataset):
    def __init__(
            self,
            data_dir: str,
            split: str = None,
            transform=None,
            embeddings_path: str = None,
            topk: int = 5,
            use_embeddings: bool = False,
            use_generated_captions: bool = True,
    ):
        self.num_all_captions = COCO_NUM_ALL_CAPTIONS

        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        self.data = self._load_data()
        self.embeddings_path = embeddings_path
        self.use_embeddings = use_embeddings
        self.topk = topk
        self.caption_embeddings, self.retrieval_results = self._load_caption_embeddings_and_retrieval_results()

        self.use_generated_captions = use_generated_captions
        self.generated_captions = self._load_generated_captions() if use_generated_captions else None

    def _load_data(self):
        data_file = os.path.join(self.data_dir, "annotations", f"{self.split}.json")
        data = load_json_lines_file(data_file)
        dataset = []
        for item in data:
            image_path = os.path.join(
                self.data_dir,
                item["filepath"],
                item["filename"]
            )
            captions = item["sentences"]
            dataset.append({
                "image_path": image_path,
                "filepath": item["filepath"],
                "captions": captions,
                "sentids": item["sentids"],
                "imgid": item["imgid"],
            })
        return dataset

    def _load_caption_embeddings_and_retrieval_results(self):
        embeddings_files = os.listdir(self.embeddings_path)
        embeddings_files = [file for file in embeddings_files if file.startswith("caption_") and file.endswith(".pt")]
        embeddings_dict = {}

        retrieval_results_dict = {} # {caption_idx: {img_path: [topk_retrieval_image_paths]}}

        for i, file in enumerate(embeddings_files):
            retrieval_results_dict[i] = {}
            embeddings = torch.load(os.path.join(self.embeddings_path, file))
            for img_path in embeddings:
                # Whether to use the caption embeddings (global) which were used to retrieve the images
                if self.use_embeddings:
                    if i not in embeddings_dict:
                        embeddings_dict[i] = {}
                    embeddings_dict[i][img_path] = embeddings[img_path]["text_embeds"]
                retrieval_results_dict[i][img_path] = embeddings[img_path]["retrieval_results"][:self.topk]
        return embeddings_dict, retrieval_results_dict

    def _load_generated_captions(self):
        generated_captions_file = os.path.join(self.data_dir, "captions", f"captions_{self.split}.json")
        generated_captions = load_json_file(generated_captions_file)
        return generated_captions

    def _get_generated_captions(self, retrieval_results_img_paths):
        generated_text = []
        for img_path in retrieval_results_img_paths:
            generated_text.append(self.generated_captions[os.path.basename(img_path)])
        return generated_text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        captions = item["captions"]
        random_idx = random.randint(0, self.num_all_captions - 1)
        query = captions[random_idx]
        ground_truth = [caption for i, caption in enumerate(captions) if i != random_idx]
        ground_truth = ground_truth[:self.num_all_captions - 1]

        # Collect feedback from the first stage of retrieval for each textual query:
        # 1. Get the retrieved images and their paths
        retrieval_results_img_paths = self.retrieval_results[random_idx][os.path.basename(item["image_path"])]
        retrieval_results_img_paths = [
            os.path.join(self.data_dir, item["filepath"], img_path)
            for img_path in retrieval_results_img_paths
        ]
        retrieval_results_images = []
        for img_path in retrieval_results_img_paths:
            retrieved_image = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                retrieved_image = self.transform(retrieved_image)
            retrieval_results_images.append(retrieved_image)

        # 2. Get captions generated with LLaVA for the retrieved images
        generated_text = self._get_generated_captions(retrieval_results_img_paths) if (
            self.use_generated_captions
        ) else None

        return {
            "image": image,
            "img_path": item["image_path"],
            "class_label": item["imgid"], # we want to retrieve the correct image id based on caption
            "query": query,
            "ground_truth": ground_truth,
            "retrieval_results_images": retrieval_results_images,
            "retrieval_results_img_paths": retrieval_results_img_paths,
            "generated_text": generated_text,
        }
