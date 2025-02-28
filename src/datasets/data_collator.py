from typing import Dict, List, Optional

import torch
from transformers import AutoProcessor


class ImageTextDataCollator:
    def __init__(self, processor: Optional[AutoProcessor] = None):
        self.processor = processor

    def __call__(self, batch: List[Dict]) -> Dict:
        processed_batch = {}
        for key in batch[0].keys():
            if key != "image" and key != "text":
                processed_batch[key] = torch.stack([example[key] for example in batch])

        if self.processor is not None:
            processed_img_text = self.processor(
                images=[example['image'] for example in batch],
                text=[example['text_class_label'] for example in batch],
                return_tensors="pt",
                padding=True
            )
            processed_batch['image'] = processed_img_text['pixel_values']
            processed_batch['input_ids'] = processed_img_text['input_ids']
            processed_batch['attention_mask'] = processed_img_text['attention_mask']
        return processed_batch
