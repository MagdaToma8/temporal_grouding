import os
import random
import json
from typing import List, Dict, Union, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor

from src.utils.utils import load_txt_file, load_pkl_file
from src.datasets.data_collator import ImageTextDataCollator


def load_cub_data(config, split, processor, process_images=True):
    assert split in ["train", "val", "test"]

    data_dir = config.get("data_dir", None)
    assert data_dir is not None, "data_dir is required"

    random_per_class = config.get("random_per_class", None)
    if random_per_class is not None:
        assert random_per_class > 0, "random_per_class must be greater than 0"

    captions_file = config.get("captions_file", None)

    transform = config.get("transform", None)

    text_use_positive_attributes = config.get("text_use_positive_attributes", True)

    selected_attributes = load_selected_attributes(os.path.join(data_dir, "processed_data", 'attributes.txt'))
    textual_attributes = create_textual_attributes(selected_attributes)
    class_labels = load_class_labels(os.path.join(data_dir, 'classes.txt'))

    dataset = CUBDataset(
        data_dir=data_dir,
        split=split,
        selected_attributes_mapping=textual_attributes,
        text_class_labels=class_labels,
        transforms=transform,
        text_use_positive_attributes=text_use_positive_attributes,
        random_per_class=random_per_class,
        captions_file=captions_file
    )
    text_types = ['text_class_label', 'text_attributes', 'text_class_label_and_attribute']

    if captions_file is not None:
        text_types.append('text_caption')
        text_types.append('text_caption_and_attribute')

    dataset_collator = CUBDatasetCollator(
        processor=processor,
        text_type=text_types,
        process_images=process_images
    )

    return dataset, dataset_collator


def load_selected_attributes(file_path: str) -> List[List[str]]:
    attributes = load_txt_file(file_path)
    attributes = [attribute.split() for attribute in attributes]
    attributes = [attribute[1] for attribute in attributes]
    return attributes


def create_textual_attributes(
    selected_attributes: List[List[str]],
) -> List[str]:
    textual_attributes = {}
    for i in range(len(selected_attributes)):
        attr_split = selected_attributes[i].split("::")
        attribute_type = attr_split[0]
        attribute_property = attr_split[1]

        attribute_type = attribute_type.replace('has_', '').lower()
        attribute_type = attribute_type.replace('_', ' ').lower()
        attribute_property = attribute_property.replace('_', ' ').lower()

        text = f"{attribute_property} {attribute_type}"

        textual_attributes[i] = {}
        textual_attributes[i][1] = text
        textual_attributes[i][0] = f"not {text}"

    return textual_attributes


def load_class_labels(file_path: str) -> List[str]:
    class_labels = load_txt_file(file_path)
    text_class_labels = [label.split()[1][4:].replace('_', ' ') for label in class_labels]
    return text_class_labels


class CUBDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            split: str,
            selected_attributes_mapping: Dict[int, Dict[int, str]],
            text_class_labels: List[str],
            transforms=None,
            text_use_positive_attributes: bool = True,
            random_per_class: Optional[int] = None,
            captions_file: Optional[str] = None,
        ):
        self.data_dir = data_dir
        self.split = split
        self.data = load_pkl_file(os.path.join(data_dir, f'processed_data/{split}.pkl'))
        if random_per_class is not None:
            self.data = self._get_random_per_class(random_per_class)
        self.transforms = transforms
        self.selected_attributes_mapping = selected_attributes_mapping
        self.text_class_labels = text_class_labels
        self.text_use_positive_attributes = text_use_positive_attributes
        self.captions_file = captions_file
        self._load_captions(captions_file) if captions_file is not None else None

    def __len__(self):
        return len(self.data)

    def _load_and_process_image(self, image_path: str) -> Image.Image:
        image = Image.open(image_path)
        image = self.transforms(image) if self.transforms is not None else image
        return image

    def _load_and_process_attributes(self, attributes: List[List[str]], positive: bool = False) -> List[str]:
        if positive:
            attributes = [
                self.selected_attributes_mapping[i][attributes[i]]
                for i in range(len(attributes))
                if attributes[i] == 1
            ]
        else:
            attributes = [self.selected_attributes_mapping[i][attributes[i]] for i in range(len(attributes))]
        return attributes

    def _load_captions(self, captions_file: str) -> List[str]:
        with open(captions_file, 'r') as f:
            captions_json = json.load(f)
        captions = captions_json['captions']
        img_paths = captions_json['img_paths']
        captions_dict = {img_paths[i]: captions[i] for i in range(len(img_paths))}
        for i, item in enumerate(self.data):
            self.data[i]['caption'] = captions_dict[os.path.join(self.data_dir, 'images', item['img_path'])]

    def _get_random_per_class(self, random_per_class: int) -> List[Dict]:
        class_idx = {}
        data = []
        for item in self.data:
            if item['class_label'] not in class_idx:
                class_idx[item['class_label']] = []
            class_idx[item['class_label']].append(item)

        for _, items in class_idx.items():
            if len(items) >= random_per_class:
                data.extend(random.sample(items, random_per_class))
            else:
                data.extend(items)
        # print([item["img_path"] for item in data])
        return data

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, 'images', self.data[idx]['img_path'])
        class_label = self.data[idx]['class_label'] - 1
        binary_attributes = self.data[idx]['attribute_label']
        if self.text_use_positive_attributes:
            textual_attributes = self._load_and_process_attributes(binary_attributes, positive=True)
        else:
            textual_attributes = self._load_and_process_attributes(binary_attributes)
        # print(img_path, textual_attributes)
        # randomly choose one of the textual attributes
        # textual_attributes = random.choice(textual_attributes)
        textual_attributes = ', '.join(random.sample(textual_attributes, 3))
        text_class_label = f"{self.text_class_labels[class_label]}"
        text_class_label_and_attribute = f"{text_class_label} with {textual_attributes}"
        # print(text_class_label_and_attribute)
        img = self._load_and_process_image(img_path)

        if self.captions_file is not None:
            text_caption = self.data[idx]['caption']
            text_caption_and_attribute = f"{text_caption[:-1]}: {textual_attributes}"
        else:
            text_caption = None
            text_caption_and_attribute = None

        return {
            'image': img,
            'img_path': img_path,
            'class_label': class_label,
            'text_class_label': text_class_label,
            'text_class_label_and_attribute': text_class_label_and_attribute,
            'text_attributes': f"A bird with {textual_attributes}",
            'binary_attributes': binary_attributes,
            'text_caption': text_caption,
            'text_caption_and_attribute': text_caption_and_attribute
        }

class CUBDatasetCollator(ImageTextDataCollator):
    def __init__(
            self,
            processor: Optional[AutoProcessor] = None,
            text_type: Optional[Union[str, List[str]]] = None,
            process_images: bool = True,
        ):
        super().__init__(processor)
        if isinstance(text_type, str):
            assert text_type in [
                'text_class_label',
                'text_attributes',
                'text_class_label_and_attribute',
                'text_caption',
                'text_caption_and_attribute'
            ]
        elif isinstance(text_type, list):
            assert "text_class_label" in text_type
            assert all(
                t in [
                    'text_class_label',
                    'text_attributes',
                    'text_class_label_and_attribute',
                    'text_caption',
                    'text_caption_and_attribute'
                ]
                for t in text_type
            )
        if text_type is None:
            self.text_type = []
        else:
            self.text_type = text_type if isinstance(text_type, list) else [text_type]
        self.process_images = process_images

    def __call__(self, batch: List[Dict]) -> Dict:
        processed_batch = {}

        processed_batch['img_path'] = np.array([example['img_path'] for example in batch])
        processed_batch['class_label'] = torch.tensor([example['class_label'] for example in batch])
        processed_batch['binary_attributes'] = torch.stack(
            [torch.tensor(example['binary_attributes']) for example in batch]
        )

        text_processor = self.text_type[0]
        if self.process_images and self.processor is not None:
            processed_img_text = self.processor(
                images=[example['image'] for example in batch],
                text=[example[text_processor] for example in batch],
                return_tensors="pt",
                padding=True
            )
            processed_batch['image'] = processed_img_text['pixel_values']
        else:
            processed_batch['image'] = [example['image'] for example in batch]

        if self.processor is not None:
            for i in range(len(self.text_type)):
                processed_text_type = self.processor.tokenizer(
                    [example[self.text_type[i]] for example in batch],
                    padding=True,
                    return_tensors="pt"
                )

            processed_batch[self.text_type[i]] = processed_text_type['input_ids']
            processed_batch[f'{self.text_type[i]}_attention_mask'] = processed_text_type['attention_mask']
        return processed_batch
