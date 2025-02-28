import os
from PIL import Image
import random

import numpy as np
from torch.utils.data import Dataset
import torch
from transformers import AutoProcessor

from src.utils.utils import load_json_file
from src.datasets.data_collator import ImageTextDataCollator


FLICKR_NUM_ALL_CAPTIONS = 5


def load_flickr_data(config, split, processor, process_images=True, summarizer=False):
    assert split in ["train", "val", "test"]

    data_dir = config.get("data_dir", None)
    assert data_dir is not None, "data_dir is required"

    data_file = config.get("data_file", None)
    assert data_file is not None, "data_file is required"

    transform = config.get("transform", None)

    num_captions_to_use = config.get("num_captions_to_use", 5)
    assert 1 <= num_captions_to_use <= 5, "num_captions_to_use must be between 1 and 5"

    process_images = config.get("process_images", False) or process_images

    if summarizer:
        embeddings_path = config.get("embeddings_path", None)
        assert embeddings_path is not None, "embeddings_path is required"
        embeddings_path = os.path.join(embeddings_path, split)

        topk = config.get("topk", 5)
        use_embeddings = config.get("use_embeddings", True)
        use_detected_objects = config.get("use_detected_objects", False)
        use_classified_objects = config.get("use_classified_objects", False)
        use_generated_captions = config.get("use_generated_captions", True)
        dataset = FlickrDatasetSummarizer(
            data_dir=data_dir,
            data_file=data_file,
            split=split,
            transform=transform,
            embeddings_path=embeddings_path,
            topk=topk,
            use_embeddings=use_embeddings,
            use_detected_objects=use_detected_objects,
            use_classified_objects=use_classified_objects,
            use_generated_captions=use_generated_captions,
        )

        collator = FlickrDatasetSummarizerCollator(
            processor=processor,
            process_images=process_images,
        )

    else:
        dataset = FlickrDataset(
            data_dir=data_dir,
            data_file=data_file,
            split=split,
            transform=transform,
            num_captions_to_use=num_captions_to_use
        )

        collator = FlickrCollator(
            processor=processor,
            process_images=process_images,
            num_captions=num_captions_to_use
        )

    return dataset, collator


class FlickrDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            data_file: str,
            split: str = None,
            transform=None,
            num_captions_to_use: int = 5,
    ):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.data_file = data_file
        assert 1 <= num_captions_to_use <= 5
        self.num_captions_to_use = num_captions_to_use
        self.data = self._load_data()

    def _load_data(self):
        data = load_json_file(self.data_file)["images"]
        dataset = []
        for item in data:
            if item["split"] == self.split:
                image_path = os.path.join(self.data_dir, "flickr30k-images", item["filename"])
                sentences = item["sentences"]
                captions = [sentences[i]["raw"] for i in range(len(sentences))]
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
        image = Image.open(item["image_path"])
        if self.transform is not None:
            image = self.transform(image)
        if self.num_captions_to_use < 5:
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


class FlickrCollator(ImageTextDataCollator):
    def __init__(
            self,
            processor: AutoProcessor = None,
            process_images: bool = True,
            num_captions: int = 5,
        ):
        super().__init__(processor)
        self.process_images = process_images
        self.num_captions = num_captions

    def __call__(self, batch):
        processed_batch = {}

        processed_batch['img_path'] = np.array([example['img_path'] for example in batch])
        processed_batch['class_label'] = torch.tensor([example['class_label'] for example in batch])

        if self.process_images and self.processor is not None:
            processed_img_text = self.processor(
                images=[example['image'] for example in batch],
                text=[example[f"caption_{0}"] for example in batch],
                return_tensors="pt",
                padding=True
            )
            processed_batch['image'] = processed_img_text['pixel_values']
        else:
            processed_batch['image'] = [example['image'] for example in batch]

        if self.processor is not None:
            for i in range(self.num_captions):
                processed_text = self.processor(
                    text=[example[f"caption_{i}"] for example in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                processed_batch[f"caption_{i}"] = processed_text['input_ids']
                processed_batch[f'caption_{i}_attention_mask'] = processed_text['attention_mask']
        return processed_batch


class FlickrDatasetSummarizer(Dataset):
    def __init__(
            self,
            data_dir: str,
            data_file: str,
            split: str = None,
            transform=None,
            embeddings_path: str = None,
            topk: int = 5,
            use_embeddings: bool = False,
            use_detected_objects: bool = True,
            use_classified_objects: bool = True,
            use_generated_captions: bool = True,
    ):
        self.num_all_captions = FLICKR_NUM_ALL_CAPTIONS

        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.data_file = data_file
        self.data = self._load_data()
        self.embeddings_path = embeddings_path
        self.use_embeddings = use_embeddings
        self.topk = topk
        self.caption_embeddings, self.retrieval_results = self._load_caption_embeddings_and_retrieval_results()
        assert not ((use_detected_objects or use_classified_objects) and use_generated_captions), \
            "Use textual feedback from yolo-based models or LLaVA-generated captions, but not both"
        self.use_detected_objects = use_detected_objects
        self.use_classified_objects = use_classified_objects
        self.use_generated_captions = use_generated_captions
        self.detected_objects = self._load_detected_objects() if use_detected_objects else None
        self.classified_objects = self._load_classified_objects() if use_classified_objects else None
        self.generated_captions = self._load_generated_captions() if use_generated_captions else None

    def _load_data(self):
        data = load_json_file(self.data_file)["images"]
        dataset = []
        for item in data:
            if item["split"] == self.split:
                image_path = os.path.join(self.data_dir, "flickr30k-images", item["filename"])
                sentences = item["sentences"]
                captions = [sentences[i]["raw"] for i in range(len(sentences))]
                dataset.append({
                    "image_path": image_path,
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

    def _load_detected_objects(self):
        od_json_file = os.path.join(self.data_dir, "yolo", "object_detection", f"{self.split}_yolo_text.json")
        object_detection_texts = load_json_file(od_json_file)

        return object_detection_texts

    def _load_classified_objects(self):
        cls_json_file = os.path.join(self.data_dir, "yolo", "classification", f"{self.split}_yolo_text.json")
        classification_texts = load_json_file(cls_json_file)

        return classification_texts

    def _load_generated_captions(self):
        generated_captions_file = os.path.join(self.data_dir, "captions", f"captions_{self.split}.json")
        generated_captions = load_json_file(generated_captions_file)
        return generated_captions

    def _generate_text_feedback(self, retrieval_results_img_paths, img_path):
        object_detection_string = ""
        classification_string = ""

        if self.use_detected_objects:
            topk_object_detection_texts = [
                self.detected_objects[os.path.basename(img_path)] for img_path in retrieval_results_img_paths
            ]
            objects = set()
            for text in topk_object_detection_texts:
                entities = text.split(", ")
                objects.update(''.join(c for c in entity if c.isalpha() or c.isspace()) for entity in entities)
            object_detection_string = " ".join(objects)

        if self.use_classified_objects:
            topk_classification_texts = [
                self.classified_objects[os.path.basename(img_path)] for img_path in retrieval_results_img_paths
            ]
            classified = set()
            for text in topk_classification_texts:
                entities = text.split(", ")
                classified.update(''.join(c for c in entity if c.isalpha() or c.isspace()) for entity in entities)
            classification_string = " ".join(classified)

        text_feedback = object_detection_string + " " + classification_string
        text_feedback = ' '.join(text_feedback.split())
        text_feedback = text_feedback.replace("_", " ")
        text_feedback = list(set(text_feedback.split()))
        return text_feedback

    def _get_generated_captions(self, retrieval_results_img_paths):
        generated_text = []
        for img_path in retrieval_results_img_paths:
            generated_text.append(self.generated_captions[os.path.basename(img_path)])
        return generated_text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"])
        if self.transform is not None:
            image = self.transform(image)

        captions = item["captions"]
        random_idx = random.randint(0, self.num_all_captions - 1)
        query = captions[random_idx]
        ground_truth = [caption for i, caption in enumerate(captions) if i != random_idx]

        # Collect feedback from the first stage of retrieval for each textual query:
        # 1. Get the retrieved images and their paths
        retrieval_results_img_paths = self.retrieval_results[random_idx][os.path.basename(item["image_path"])]
        retrieval_results_img_paths = [
            os.path.join(self.data_dir, "flickr30k-images", img_path)
            for img_path in retrieval_results_img_paths
        ]
        retrieval_results_images = []
        for img_path in retrieval_results_img_paths:
            retrieved_image = Image.open(img_path)
            if self.transform is not None:
                retrieved_image = self.transform(retrieved_image)
            retrieval_results_images.append(retrieved_image)

        # 2. Aggregate the text feedback from the retrieved images (from yolo-based object detection and classification)
        text_feedback = self._generate_text_feedback(retrieval_results_img_paths, item["image_path"]) if (
            self.use_detected_objects or self.use_classified_objects
        ) else None

        # 3. Get captions generated with LLaVA for the retrieved images
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
            "text_feedback": text_feedback,
            "generated_text": generated_text,
        }


class FlickrDatasetSummarizerCollator:
    def __init__(self, processor=None, process_images=True):
        self.processor = processor
        self.process_images = process_images

    def __call__(self, batch):
        processed_batch = {}

        # Process image paths and class labels
        processed_batch['img_path'] = np.array([example['img_path'] for example in batch])
        processed_batch['class_label'] = torch.tensor([example['class_label'] for example in batch])

        # Process ground truth images
        if self.process_images and self.processor is not None:
            processed_images = self.processor(
                images=[example['image'] for example in batch],
                return_tensors="pt",
                padding=True
            )
            processed_batch['image'] = processed_images['pixel_values']
        else:
            processed_batch['image'] = [example['image'] for example in batch]

        # Process text fields
        if self.processor is not None:
            # Process query
            processed_query = self.processor(
                text=[example['query'] for example in batch],
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            processed_batch['query_input_ids'] = processed_query['input_ids']
            processed_batch['query_attention_mask'] = processed_query['attention_mask']

            # Process ground truth captions
            processed_batch["num_ground_truth_captions"] = len(batch[0]["ground_truth"])
            all_ground_truth_captions = []
            # Flatten ground truth captions for each example along the batch dimension
            # Shape for input_ids and ground_truth_attention_mask:
            #   [bsz * (num_captions - 1), seq_len]
            #   seq_len will be padded to the longest ground truth caption in the batch
            for example in batch:
                all_ground_truth_captions.extend(example["ground_truth"])
            processed_ground_truth = self.processor(
                text=all_ground_truth_captions,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            processed_batch['ground_truth_input_ids'] = processed_ground_truth['input_ids']
            processed_batch['ground_truth_attention_mask'] = processed_ground_truth['attention_mask']

            # Process text feedback from yolo-based models
            # Shape for input_ids and attention_mask:
            #   [bsz, seq_len]
            if any(example['text_feedback'] for example in batch):
                processed_text_feedback = self.processor(
                    text=[' '.join(example['text_feedback']) for example in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                processed_batch['text_feedback_input_ids'] = processed_text_feedback['input_ids']
                processed_batch['text_feedback_attention_mask'] = processed_text_feedback['attention_mask']

            # Process LLaVA-generated captions
            # Flatten generated captions for each retrieved image in topk along the batch dimension
            # Shape for input_ids and attention_mask:
            #   [bsz * topk, seq_len]
            #   seq_len will be padded to the longest generated caption in the batch
            if any(example['generated_text'] for example in batch):
                processed_batch["num_generated_captions"] = len(batch[0]['generated_text'])
                all_generated_captions = []
                for example in batch:
                    all_generated_captions.extend(example['generated_text'])
                processed_generated_text = self.processor(
                    text=all_generated_captions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                processed_batch['generated_text_input_ids'] = processed_generated_text['input_ids']
                processed_batch['generated_text_attention_mask'] = processed_generated_text['attention_mask']

        # Process retrieval results images
        # Flatten topk retrieved images along the batch dimension
        # Shape for pixel_values:
        #   [bsz * topk, 3, 224, 224]
        if self.process_images and self.processor is not None:
            retrieved_images = [img for example in batch for img in example['retrieval_results_images']]
            processed_retrieval_images = self.processor(
                images=retrieved_images,
                return_tensors="pt",
                padding=True
            )
            processed_batch['retrieval_results_images'] = processed_retrieval_images['pixel_values']
        else:
            processed_batch['retrieval_results_images'] = [example['retrieval_results_images'] for example in batch]

        return processed_batch
